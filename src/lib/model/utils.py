from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _sigmoid12(x):
  y = torch.clamp(x.sigmoid_(), 1e-12)
  return y

def _gather_feat(feat, ind):
  dim = feat.size(2)
  ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
  feat = feat.gather(1, ind)
  return feat

def _tranpose_and_gather_feat(feat, ind):
  feat = feat.permute(0, 2, 3, 1).contiguous()
  feat = feat.view(feat.size(0), -1, feat.size(3))
  feat = _gather_feat(feat, ind)
  return feat

def flip_tensor(x):
  return torch.flip(x, [3])
  # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def _nms(heat, kernel=3):
  pad = (kernel - 1) // 2

  hmax = nn.functional.max_pool2d(
      heat, (kernel, kernel), stride=1, padding=pad)
  keep = (hmax == heat).float()
  return heat * keep

def _topk_channel(scores, K=100):
  batch, cat, height, width = scores.size()
  
  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

  topk_inds = topk_inds % (height * width)
  topk_ys   = (topk_inds / width).int().float()
  topk_xs   = (topk_inds % width).int().float()

  return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=100):
  batch, cat, height, width = scores.size()
    
  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

  topk_inds = topk_inds % (height * width)
  topk_ys   = (topk_inds / width).int().float()
  topk_xs   = (topk_inds % width).int().float()
    
  topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
  topk_clses = (topk_ind / K).int()
  topk_inds = _gather_feat(
      topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

  return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
  """Calculate overlap between two set of bboxes.

  If ``is_aligned `` is ``False``, then calculate the overlaps between each
  bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
  pair of bboxes1 and bboxes2.

  Args:
      bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
      bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
          B indicates the batch dim, in shape (B1, B2, ..., Bn).
          If ``is_aligned `` is ``True``, then m and n must be equal.
      mode (str): "iou" (intersection over union), "iof" (intersection over
          foreground) or "giou" (generalized intersection over union).
          Default "iou".
      is_aligned (bool, optional): If True, then m and n must be equal.
          Default False.
      eps (float, optional): A value added to the denominator for numerical
          stability. Default 1e-6.

  Returns:
      Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
  """

  assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
  # Either the boxes are empty or the length of boxes's last dimenstion is 4
  assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
  assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

  # Batch dim must be the same
  # Batch dim: (B1, B2, ... Bn)
  assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
  batch_shape = bboxes1.shape[:-2]

  rows = bboxes1.size(-2)
  cols = bboxes2.size(-2)
  if is_aligned:
    assert rows == cols

  if rows * cols == 0:
    if is_aligned:
      return bboxes1.new(batch_shape + (rows,))
    else:
      return bboxes1.new(batch_shape + (rows, cols))

  area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
          bboxes1[..., 3] - bboxes1[..., 1])
  area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
          bboxes2[..., 3] - bboxes2[..., 1])

  if is_aligned:
    lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
    rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

    wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
    overlap = wh[..., 0] * wh[..., 1]

    if mode in ['iou', 'giou']:
      union = area1 + area2 - overlap
    else:
      union = area1
    if mode == 'giou':
      enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
      enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
  else:
    lt = torch.max(bboxes1[..., :, None, :2],
                   bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
    rb = torch.min(bboxes1[..., :, None, 2:],
                   bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

    wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
    overlap = wh[..., 0] * wh[..., 1]

    if mode in ['iou', 'giou']:
      union = area1[..., None] + area2[..., None, :] - overlap
    else:
      union = area1[..., None]
    if mode == 'giou':
      enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                              bboxes2[..., None, :, :2])
      enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                              bboxes2[..., None, :, 2:])

  eps = union.new_tensor([eps])
  union = torch.max(union, eps)
  ious = overlap / union
  if mode in ['iou', 'iof']:
    return ious
  # calculate gious
  enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
  enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
  enclose_area = torch.max(enclose_area, eps)
  gious = ious - (enclose_area - union) / enclose_area
  return gious


class ProposalTargetCreator(object):
  def __init__(self, opt,
               n_sample=100,
               pos_ratio=4,
               pos_iou_thresh=0.65,
               neg_iou_thresh_hi=0.3):
    self.opt = opt
    self.n_sample = n_sample
    self.pos_ratio = int(pos_ratio)
    self.OHEM_ratio = 0.5
    self.pos_iou_thresh = pos_iou_thresh
    self.neg_iou_thresh_hi = neg_iou_thresh_hi

  def __call__(self, output, batch):
    batch_size = batch['hm'].shape[0]
    sample = {'batch': {}, 'output': {}, 'num_pos': 0, 'num_neg': 0}
    batch_fakeneg = (self.n_sample - 1) * torch.ones(128, dtype=torch.int64, device=self.opt.device)
    heads_roi = ['dep', 'dim', 'rot', 'amodel_offset']
    if 'cls' in output:
      heads_roi.append('cls')
    for head in heads_roi:
      sample['batch'][head] = []
      sample['output'][head] = []
    if 'rot' in heads_roi:
      sample['output']['rot'] = []
      sample['batch'].pop('rot')
      sample['batch']['rotres'] = []
      sample['batch']['rotbin'] = []

    for b in range(batch_size):
      if output['keep_idx_pos'][b].shape[0] == 0:
        continue

      output_bboxes = output['bboxes'][b, output['keep_idx_pos'][b], :]
      ious = bbox_overlaps(output_bboxes, batch['bboxes'][b], mode='iou')
      gt_matched_ious, gt_assignment = ious.max(axis=1)
      gt_assignment = gt_assignment[gt_matched_ious > 0.1]
      gt_matched_ious = gt_matched_ious[gt_matched_ious > 0.1]

      # Remove 'ignore'
      ignore_idx = (batch['ignore'][b] == True).nonzero().view(-1)
      for gt_idx, gt in enumerate(gt_assignment):
        if gt in ignore_idx:
          gt_matched_ious[gt_idx] = 0

      # Remove rois if multiple rois are matched with gt
      gt_unique = torch.unique(gt_assignment)
      if gt_unique.shape != gt_assignment.shape:
        for gt in gt_assignment:
          repeated_idxs = torch.where(gt_assignment==gt)[0]
          if repeated_idxs.shape[0] > 1:
            repeated_ious = gt_matched_ious[repeated_idxs]
            _, repeated_max_idx = repeated_ious.max(axis=0)
            for repeated in repeated_idxs:
              if repeated == repeated_idxs[repeated_max_idx]:
                continue
              else:
                gt_matched_ious[repeated] = 0

      pos_idx_out = torch.where(gt_matched_ious >= self.pos_iou_thresh)[0]
      pos_idx_out_final = output['keep_idx_pos'][b][pos_idx_out]
      num_pos = pos_idx_out_final.shape[0]

      neg_idx_out = torch.where(gt_matched_ious < self.neg_iou_thresh_hi)[0]
      neg_idx_wo_ignore = (~batch['ignore'][b].bool() * output['keep_idx_neg'][b] == True).nonzero().view(-1)
      neg_idx_out_final = torch.cat((output['keep_idx_pos'][b][neg_idx_out].float(),
                                     neg_idx_wo_ignore.float())).long()
      neg_idx_out = neg_idx_out_final
      num_neg = neg_idx_out_final.shape[0]

      if num_neg > num_pos * self.pos_ratio:
        if num_pos > 0:
          num_neg = int(num_pos) * self.pos_ratio
        else:
          num_neg = min(num_neg, self.pos_ratio)
        neg_samples = torch.randperm(num_neg)[:num_neg]
        neg_idx_out_final = neg_idx_out_final[neg_samples]
        neg_idx_out = neg_idx_out_final

      pos_idx_bat = gt_assignment[pos_idx_out]
      neg_idx_bat = batch_fakeneg[neg_idx_out]
      sample['num_pos'] += num_pos
      sample['num_neg'] += num_neg

      for head in heads_roi:
        if head in 'cls':
          cls_idx_out = torch.cat((pos_idx_out_final, neg_idx_out_final), dim=0)
          cls_idx_bat = torch.cat((pos_idx_bat, neg_idx_bat), dim=0)
          sample['output'][head].append(output[head][b, cls_idx_out, :])
          sample['batch'][head].append(batch[head][b, cls_idx_bat].long())
        else:
          if num_pos > 0:
            sample['output'][head].append(output[head][b, pos_idx_out, :])
            if head == 'rot':
              sample['batch']['rotbin'].append(batch['rotbin'][b, pos_idx_bat, :])
              sample['batch']['rotres'].append(batch['rotres'][b, pos_idx_bat, :])
            else:
              sample['batch'][head].append(batch[head][b, pos_idx_bat, :])

    if sample['num_pos'] > 0:
      for head in sample['output']:
        sample['output'][head] = torch.cat(sample['output'][head], dim=0)
      for head in sample['batch']:
        sample['batch'][head] = torch.cat(sample['batch'][head], dim=0)

      if self.opt.OHEM:
        num_OHEM = max(int(sample['num_pos']*self.OHEM_ratio), 1)

        dep_diff = torch.abs(sample['output']['dep'] - sample['batch']['dep'])
        dep_score_descend = torch.sort(dep_diff, dim=0, descending=True)[1][:num_OHEM]
        sample['output']['dep'] = sample['output']['dep'][dep_score_descend]
        sample['batch']['dep'] = sample['batch']['dep'][dep_score_descend]

        cls_diff = torch.abs(sample['output']['cls'].sigmoid() - sample['batch']['cls'].view(-1, 1))
        cls_score_descend = torch.sort(cls_diff, dim=0, descending=True)[1][:int(num_OHEM*5)]
        sample['output']['cls'] = sample['output']['cls'][cls_score_descend]
        sample['batch']['cls'] = sample['batch']['cls'][cls_score_descend]

    return sample