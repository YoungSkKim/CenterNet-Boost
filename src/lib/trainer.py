from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import numpy as np
from progress.bar import Bar

from model.data_parallel import DataParallel
from utils.utils import AverageMeter

import torch.nn.functional as F

from model.losses import FastFocalLoss, RegWeightedL1Loss
from model.losses import BinRotLoss, WeightedBCELoss, compute_rot_loss
from model.decode import generic_decode
from model.utils import _sigmoid, flip_tensor, flip_lr_off, flip_lr, _tranpose_and_gather_feat
from utils.debugger import Debugger
from utils.post_process import generic_post_process

from utils.depth import eval_depth

class GenericLoss(torch.nn.Module):
  def __init__(self, opt):
    super(GenericLoss, self).__init__()
    self.crit = FastFocalLoss(opt=opt)
    self.crit_reg = RegWeightedL1Loss()
    if 'rot' in opt.heads:
      self.crit_rot = BinRotLoss()
    if 'nuscenes_att' in opt.heads:
      self.crit_nuscenes_att = WeightedBCELoss()
    self.bceloss = torch.nn.BCEWithLogitsLoss(reduction='none')
    self.l1loss = torch.nn.L1Loss(reduction='none')
    self.opt = opt
    if self.opt.eval_depth:
      self.metrics_all = []
      self.metrics_obj = []

  def _sigmoid_output(self, output):
    if 'hm' in output:
      output['hm'] = _sigmoid(output['hm'])
    if 'hm_hp' in output:
      output['hm_hp'] = _sigmoid(output['hm_hp'])
    if 'dep' in output:
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
    if 'depconf' in output:
      output['depconf'] = torch.clamp(output['depconf'].sigmoid(), min=0.01, max=0.99)
    return output

  def forward(self, outputs, batch):
    opt = self.opt
    losses = {head: 0 for head in opt.heads}
    if opt.auxdep: losses.update({'auxdep': 0})

    for s in range(opt.num_stacks):
      output = outputs[s]
      output = self._sigmoid_output(output)

      if opt.eval_depth:
        self.metrics_all, self.metrics_obj = eval_depth(batch, output, self.metrics_all, self.metrics_obj)
        for items in losses:  # dump dummy losses
          losses[items] = torch.tensor(0, dtype=torch.float, device=opt.device)
        continue

      if 'hm' in output:
        losses['hm'] += self.crit(
          output['hm'], batch['hm'], batch['ind'], 
          batch['mask'], batch['cat']) / opt.num_stacks

      if 'dep' in output:
        dep_pred = _tranpose_and_gather_feat(output['dep'], batch['ind'])
        dep_gt = batch['dep']
        loss = self.l1loss(dep_pred * batch['dep_mask'], dep_gt * batch['dep_mask'])
        if opt.focaldep:
          dep_delta = torch.clamp(torch.abs(dep_pred - dep_gt), 0, opt.deperror_clamp[1])
          target = torch.pow((1 + opt.focaldep_alpha * dep_delta / (dep_gt + 1e-4)), opt.focaldep_gamma)
          loss = loss * target
        losses['dep'] += loss.sum() / (batch['dep_mask'].sum() + 1e-4)

      if opt.auxdep:
        loss = self.l1loss(output['dep'].squeeze()*batch['auxdep_mask'],
                           batch['auxdep'].squeeze()*batch['auxdep_mask']) / opt.num_stacks
        losses['auxdep'] += loss.sum() / (batch['auxdep_mask'].sum() + 1e-4)

      if opt.depconf:
        if opt.auxdep:
          dep_delta = torch.abs(output['dep'].squeeze()*batch['auxdep_mask'] - \
                                batch['auxdep'].squeeze()*batch['auxdep_mask'])
          target = torch.clamp(dep_delta, opt.deperror_clamp[0], opt.deperror_clamp[1])
          target = torch.exp(-1*opt.depconf_beta*(target-opt.deperror_clamp[0]))
          pred = output['depconf'].squeeze(1)
          loss = self.l1loss(pred * batch['auxdep_mask'], target * batch['auxdep_mask'])
          losses['depconf'] += loss.sum() / (batch['auxdep_mask'].sum() + 1e-4)
        else:
          dep_delta = torch.abs(_tranpose_and_gather_feat(output['dep'], batch['ind']) - batch['dep'])
          target = torch.clamp(dep_delta, opt.deperror_clamp[0], opt.deperror_clamp[1])
          target = torch.exp(-1*opt.depconf_beta*target)
          pred = _tranpose_and_gather_feat(output['depconf'], batch['ind'])
          loss = self.l1loss(pred * batch['dep_mask'], target * batch['dep_mask'])
          losses['depconf'] += loss.sum() / (batch['dep_mask'].sum() + 1e-4)

      regression_heads = [
        'reg', 'wh', 'tracking', 'ltrb', 'ltrb_amodel', 'hps',
        'dim', 'amodel_offset', 'velocity']

      for head in regression_heads:
        if head in output:
          losses[head] += self.crit_reg(
            output[head], batch[head + '_mask'],
            batch['ind'], batch[head]) / opt.num_stacks

      if 'hm_hp' in output:
        losses['hm_hp'] += self.crit(
          output['hm_hp'], batch['hm_hp'], batch['hp_ind'],
          batch['hm_hp_mask'], batch['joint']) / opt.num_stacks
        if 'hp_offset' in output:
          losses['hp_offset'] += self.crit_reg(
            output['hp_offset'], batch['hp_offset_mask'],
            batch['hp_ind'], batch['hp_offset']) / opt.num_stacks

      if 'rot' in output:
        losses['rot'] += self.crit_rot(
          output['rot'], batch['rot_mask'], batch['ind'], batch['rotbin'],
          batch['rotres']) / opt.num_stacks

      if 'nuscenes_att' in output:
        losses['nuscenes_att'] += self.crit_nuscenes_att(
          output['nuscenes_att'], batch['nuscenes_att_mask'],
          batch['ind'], batch['nuscenes_att']) / opt.num_stacks

    losses['tot'] = 0
    for head in opt.heads:
      if head in losses:
        losses['tot'] += opt.weights[head] * losses[head]
    if opt.auxdep:
      losses['tot'] += opt.weights['auxdep'] * losses['auxdep']

    return losses['tot'], losses


class ModleWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModleWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch):
    pre_img = batch['pre_img'] if 'pre_img' in batch else None
    pre_hm = batch['pre_hm'] if 'pre_hm' in batch else None
    outputs = self.model(batch['image'], pre_img, pre_hm, is_trainer=True)
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats

class Trainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModleWithLoss(model, self.loss)

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.model_with_loss = DataParallel(
        self.model_with_loss, device_ids=gpus, 
        chunk_sizes=chunk_sizes).to(device)
    else:
      self.model_with_loss = self.model_with_loss.to(device)
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader):
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    # avg_loss_stats = {l: AverageMeter() for l in self.loss_stats \
    #                   if l == 'tot' or opt.weights[l] > 0}
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)

      if self.opt.lr_warmup and epoch-1 < self.opt.lr_warmup_epoch:
        lr = (iter_id + (epoch-1)*num_iters + 1) / (num_iters * self.opt.lr_warmup_epoch) * self.opt.lr
        for param_group in self.optimizer.param_groups:
          param_group['lr'] = lr

      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)
      output, loss, loss_stats = model_with_loss(batch)
      loss = loss.mean()
      if phase == 'train':
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['image'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
        '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0: # If not using progress bar
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
      else:
        bar.next()
      
      if opt.debug > 0:
        self.debug(batch, output, iter_id, dataset=data_loader.dataset)
      
      del output, loss, loss_stats
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results
  
  def _get_losses(self, opt):
    loss_order = ['hm', 'cls', 'wh', 'reg', 'ltrb', 'hps', 'hm_hp', \
      'hp_offset', 'dep', 'dim', 'rot', 'amodel_offset', 'auxdep', 'extdep', 'depconf',\
      'ltrb_amodel', 'tracking', 'nuscenes_att', 'velocity']
    loss_states = ['tot'] + [k for k in loss_order if k in opt.heads]
    if opt.auxdep: loss_states += ['auxdep']
    loss = GenericLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id, dataset):
    opt = self.opt
    if 'pre_hm' in batch:
      output.update({'pre_hm': batch['pre_hm']})
    dets = generic_decode(output, K=opt.K, opt=opt)
    for k in dets:
      dets[k] = dets[k].detach().cpu().numpy()
    dets_gt = batch['meta']['gt_det']
    for i in range(1):
      debugger = Debugger(opt=opt, dataset=dataset)
      img = batch['image'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * dataset.std + dataset.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')

      if 'pre_img' in batch:
        pre_img = batch['pre_img'][i].detach().cpu().numpy().transpose(1, 2, 0)
        pre_img = np.clip(((
          pre_img * dataset.std + dataset.mean) * 255), 0, 255).astype(np.uint8)
        debugger.add_img(pre_img, 'pre_img_pred')
        debugger.add_img(pre_img, 'pre_img_gt')
        if 'pre_hm' in batch:
          pre_hm = debugger.gen_colormap(
            batch['pre_hm'][i].detach().cpu().numpy())
          debugger.add_blend_img(pre_img, pre_hm, 'pre_hm')

      debugger.add_img(img, img_id='out_pred')
      if 'ltrb_amodel' in opt.heads:
        debugger.add_img(img, img_id='out_pred_amodel')
        debugger.add_img(img, img_id='out_gt_amodel')

      # Predictions
      for k in range(len(dets['scores'][i])):
        if dets['scores'][i, k] > opt.vis_thresh:
          debugger.add_coco_bbox(
            dets['bboxes'][i, k] * opt.down_ratio, dets['clses'][i, k],
            dets['scores'][i, k], img_id='out_pred')

          if 'ltrb_amodel' in opt.heads:
            debugger.add_coco_bbox(
              dets['bboxes_amodel'][i, k] * opt.down_ratio, dets['clses'][i, k],
              dets['scores'][i, k], img_id='out_pred_amodel')

          if 'hps' in opt.heads and int(dets['clses'][i, k]) == 0:
            debugger.add_coco_hp(
              dets['hps'][i, k] * opt.down_ratio, img_id='out_pred')

          if 'tracking' in opt.heads:
            debugger.add_arrow(
              dets['cts'][i][k] * opt.down_ratio, 
              dets['tracking'][i][k] * opt.down_ratio, img_id='out_pred')
            debugger.add_arrow(
              dets['cts'][i][k] * opt.down_ratio, 
              dets['tracking'][i][k] * opt.down_ratio, img_id='pre_img_pred')

      # Ground truth
      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt['scores'][i])):
        if dets_gt['scores'][i][k] > opt.vis_thresh:
          debugger.add_coco_bbox(
            dets_gt['bboxes'][i][k] * opt.down_ratio, dets_gt['clses'][i][k],
            dets_gt['scores'][i][k], img_id='out_gt')

          if 'ltrb_amodel' in opt.heads:
            debugger.add_coco_bbox(
              dets_gt['bboxes_amodel'][i, k] * opt.down_ratio,
              dets_gt['clses'][i, k],
              dets_gt['scores'][i, k], img_id='out_gt_amodel')

          if 'hps' in opt.heads and \
            (int(dets['clses'][i, k]) == 0):
            debugger.add_coco_hp(
              dets_gt['hps'][i][k] * opt.down_ratio, img_id='out_gt')

          if 'tracking' in opt.heads:
            debugger.add_arrow(
              dets_gt['cts'][i][k] * opt.down_ratio, 
              dets_gt['tracking'][i][k] * opt.down_ratio, img_id='out_gt')
            debugger.add_arrow(
              dets_gt['cts'][i][k] * opt.down_ratio, 
              dets_gt['tracking'][i][k] * opt.down_ratio, img_id='pre_img_gt')

      if 'hm_hp' in opt.heads:
        pred = debugger.gen_colormap_hp(
          output['hm_hp'][i].detach().cpu().numpy())
        gt = debugger.gen_colormap_hp(batch['hm_hp'][i].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hmhp')
        debugger.add_blend_img(img, gt, 'gt_hmhp')


      if 'rot' in opt.heads and 'dim' in opt.heads and 'dep' in opt.heads:
        dets_gt = {k: dets_gt[k].cpu().numpy() for k in dets_gt}
        calib = batch['meta']['calib'].detach().numpy() \
                if 'calib' in batch['meta'] else None
        det_pred = generic_post_process(opt, dets, 
          batch['meta']['c'].cpu().numpy(), batch['meta']['s'].cpu().numpy(),
          output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
          calib)
        det_gt = generic_post_process(opt, dets_gt, 
          batch['meta']['c'].cpu().numpy(), batch['meta']['s'].cpu().numpy(),
          output['hm'].shape[2], output['hm'].shape[3], self.opt.num_classes,
          calib)

        debugger.add_3d_detection(
          batch['meta']['img_path'][i], batch['meta']['flipped'][i],
          det_pred[i], calib[i],
          vis_thresh=opt.vis_thresh, img_id='add_pred')
        debugger.add_3d_detection(
          batch['meta']['img_path'][i], batch['meta']['flipped'][i], 
          det_gt[i], calib[i],
          vis_thresh=opt.vis_thresh, img_id='add_gt')
        debugger.add_bird_views(det_pred[i], det_gt[i], 
          vis_thresh=opt.vis_thresh, img_id='bird_pred_gt')
        debugger.compose_vis_ddd(
          batch['meta']['img_path'][i], False, det_pred[i], calib[i],
          opt.vis_thresh,
          pred, 'bird_pred_gt', img_id='out')
      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(batch['meta']['img_id'][0].numpy()))
      else:
        debugger.show_all_imgs(pause=True)
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)
