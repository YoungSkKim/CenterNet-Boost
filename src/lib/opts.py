from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    # basic experiment setting
    self.parser.add_argument('--task', default='ddd',
                             help='ctdet | ddd | multi_pose '
                             '| tracking or combined with ,')
    self.parser.add_argument('--dataset', default='kitti',
                             help='see lib/dataset/dataset_facotry for ' + 
                            'available datasets')
    self.parser.add_argument('--test_dataset', default='kitti',
                             help='coco | kitti | coco_hp | pascal')
    self.parser.add_argument('--exp_id', default='ddd/ablation/ns/ddd_ns_focaldep_0.5_1_5')
    # self.parser.add_argument('--load_model', default='../models/ddd_3dop.pth',
    # self.parser.add_argument('--load_model', default='../models/nuScenes_3Ddetection_e140.pth',
    self.parser.add_argument('--load_model', default='../exp/ddd/ablation/trial1/ddd_aux_dprop_1.0/model_60.pth',
    # self.parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    self.parser.add_argument('--test', action='store_true')
    self.parser.add_argument('--debug', type=int, default=0,
                             help='level of visualization.'
                                  '1: only show the final detection results'
                                  '2: show the network output features'
                                  '3: use matplot to display' # useful when lunching training with ipython notebook
                                  '4: save all visualizations to disk')
    self.parser.add_argument('--no_pause', action='store_true')
    self.parser.add_argument('--demo', default='', 
                             help='path to image/ image folders/ video. '
                                  'or "webcam"')
    self.parser.add_argument('--resume', type=self.str2bool, default=False,
                             help='resume an experiment. '
                                  'Reloaded the optimizer parameter and '
                                  'set load_model to model_last.pth '
                                  'in the exp dir if load_model is empty.')

    # system
    self.parser.add_argument('--gpus', default='0',
                             help='-1 for CPU, use comma for multiple gpus')
    self.parser.add_argument('--num_workers', type=int, default=16,
                             help='dataloader threads. 0 for single-thread.')
    self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                             help='disable when the input size is not fixed.')
    self.parser.add_argument('--seed', type=int, default=317, 
                             help='random seed') # from CornerNet
    self.parser.add_argument('--not_set_cuda_env', action='store_true',
                             help='used when training in slurm clusters.')

    # log
    self.parser.add_argument('--print_iter', type=int, default=0, 
                             help='disable progress bar and print to screen.')
    self.parser.add_argument('--save_all', default=True, action='store_true',
                             help='save model to disk every 5 epochs.')
    self.parser.add_argument('--vis_thresh', type=float, default=-1,
                             help='visualization threshold.')
    self.parser.add_argument('--debugger_theme', default='white', 
                             choices=['white', 'black'])
    self.parser.add_argument('--eval_val', action='store_true')
    self.parser.add_argument('--save_imgs', default='out, generic, ddd_pred', help='')
    self.parser.add_argument('--save_img_suffix', default='', help='')
    self.parser.add_argument('--skip_first', type=int, default=-1, help='')
    self.parser.add_argument('--save_video', action='store_true')
    self.parser.add_argument('--save_framerate', type=int, default=30)
    self.parser.add_argument('--resize_video', action='store_true')
    self.parser.add_argument('--video_h', type=int, default=512, help='')
    self.parser.add_argument('--video_w', type=int, default=512, help='')
    self.parser.add_argument('--transpose_video', action='store_true')
    self.parser.add_argument('--show_track_color', action='store_true')
    self.parser.add_argument('--not_show_bbox', action='store_true')
    self.parser.add_argument('--not_show_number', action='store_true')
    self.parser.add_argument('--not_show_txt', action='store_true')
    self.parser.add_argument('--qualitative', default=True, action='store_true')
    self.parser.add_argument('--tango_color', action='store_true')
    self.parser.add_argument('--only_show_dots', action='store_true')
    self.parser.add_argument('--show_trace', action='store_true')

    # model
    self.parser.add_argument('--arch', default='dla_34',
                             help='model architecture. Currently tested'
                                  'res_18 | res_101 | resdcn_18 | resdcn_101 |'
                                  'dlav0_34 | dla_34 | dlaDH_34 | hourglass')
    self.parser.add_argument('--dla_node', default='dcn') 
    self.parser.add_argument('--head_conv', type=int, default=-1,
                             help='conv layer channels for output head'
                                  '0 for no conv layer'
                                  '-1 for default setting: '
                                  '64 for resnets and 256 for dla.')
    self.parser.add_argument('--num_head_conv', type=int, default=1)
    self.parser.add_argument('--head_kernel', type=int, default=3, help='')
    self.parser.add_argument('--down_ratio', type=int, default=4,
                             help='output stride. Currently only supports 4.')
    self.parser.add_argument('--not_idaup', action='store_true')
    self.parser.add_argument('--num_classes', type=int, default=-1)
    self.parser.add_argument('--num_layers', type=int, default=101)
    self.parser.add_argument('--backbone', default='dla34')
    self.parser.add_argument('--neck', default='dlaup')
    self.parser.add_argument('--msra_outchannel', type=int, default=256)
    self.parser.add_argument('--efficient_level', type=int, default=0)
    self.parser.add_argument('--prior_bias', type=float, default=-4.6)

    # input
    self.parser.add_argument('--input_res', type=int, default=-1, 
                             help='input height and width. -1 for default from '
                             'dataset. Will be overriden by input_h | input_w')
    self.parser.add_argument('--input_h', type=int, default=-1, 
                             help='input height. -1 for default from dataset.')
    self.parser.add_argument('--input_w', type=int, default=-1, 
                             help='input width. -1 for default from dataset.')
    self.parser.add_argument('--dataset_version', default='')

    # train
    self.parser.add_argument('--optim', default='adam')
    self.parser.add_argument('--lr', type=float, default=1.25e-4,
                             help='learning rate for batch size 32.')
    self.parser.add_argument('--weight_decay', type=float, default=0)
    self.parser.add_argument('--lr_warmup', type=self.str2bool, default=False,
                             help='warmup learning rate.')
    self.parser.add_argument('--lr_warmup_epoch', type=float, default=1,
                             help='warmup epoch.')
    self.parser.add_argument('--lr_step', type=str, default='60',
                             help='drop learning rate at step.')
    self.parser.add_argument('--lr_factor', type=float, default=0.1,
                             help='drop learning rate by factor.')
    self.parser.add_argument('--save_point', type=str, default='70',
                             help='when to save the model to disk.')
    self.parser.add_argument('--num_epochs', type=int, default=70,
                             help='total training epochs.')
    self.parser.add_argument('--batch_size', type=int, default=1,
                             help='batch size')
    self.parser.add_argument('--master_batch_size', type=int, default=-1,
                             help='batch size on the master gpu.')
    self.parser.add_argument('--num_iters', type=int, default=-1,
                             help='default: #samples / batch_size.')
    self.parser.add_argument('--val_intervals', type=int, default=999,
                             help='number of epochs to run validation.')
    self.parser.add_argument('--trainval', type=self.str2bool, default=False,
                             help='include validation in training and test on test set')
    self.parser.add_argument('--ltrb', action='store_true', help='')
    self.parser.add_argument('--ltrb_weight', type=float, default=0.1, help='')
    self.parser.add_argument('--reset_hm', action='store_true')
    self.parser.add_argument('--use_kpt_center', action='store_true')
    self.parser.add_argument('--add_05', action='store_true')
    self.parser.add_argument('--dense_reg', type=int, default=1, help='')

    # test
    self.parser.add_argument('--flip_test', type=self.str2bool, default=False,
                             help='flip data augmentation.')
    self.parser.add_argument('--test_scales', type=str, default='1',
                             help='multi scale test augmentation.')
    self.parser.add_argument('--nms', action='store_true',
                             help='run nms in testing.')
    self.parser.add_argument('--K', type=int, default=100,
                             help='max number of output objects.')
    self.parser.add_argument('--max_objs', type=int, default=50,
                             help='max number of ground truth objects.')
    self.parser.add_argument('--not_prefetch_test', action='store_true',
                             help='not use parallal data pre-processing.')
    self.parser.add_argument('--fix_short', type=int, default=-1)
    self.parser.add_argument('--keep_res', action='store_true',
                             help='keep the original resolution'
                                  ' during validation.')
    self.parser.add_argument('--map_argoverse_id', action='store_true',
                             help='if trained on nuscenes and eval on kitti')
    self.parser.add_argument('--out_thresh', type=float, default=0.4, help='')
    self.parser.add_argument('--depth_scale', type=float, default=1, help='')
    self.parser.add_argument('--save_results', action='store_true')
    self.parser.add_argument('--load_results', default='')
    self.parser.add_argument('--use_loaded_results', action='store_true')
    self.parser.add_argument('--ignore_loaded_cats', default='')
    self.parser.add_argument('--model_output_list', action='store_true',
                             help='Used when convert to onnx')
    self.parser.add_argument('--non_block_test', action='store_true')
    # self.parser.add_argument('--vis_gt_bev', default='../vis/bird_pred_gt', help='')
    self.parser.add_argument('--vis_gt_bev', default='', help='')
    self.parser.add_argument('--kitti_split', default='3dop',
                             help='different validation split for kitti: 3dop | subcnn')
    self.parser.add_argument('--test_focal_length', type=int, default=-1)

    # dataset
    self.parser.add_argument('--not_rand_crop', type=self.str2bool, default=False,
                             help='not use the random crop data augmentation'
                                  'from CornerNet.')
    self.parser.add_argument('--not_max_crop', action='store_true',
                             help='used when the training dataset has'
                                  'inbalanced aspect ratios.')
    self.parser.add_argument('--shift', type=float, default=0,
                             help='when not using random crop, 0.1'
                                  'apply shift augmentation.')
    self.parser.add_argument('--scale', type=float, default=0,
                             help='when not using random crop, 0.4'
                                  'apply scale augmentation.')
    self.parser.add_argument('--aug_rot', type=float, default=0, 
                             help='probability of applying '
                                  'rotation augmentation.')
    self.parser.add_argument('--rotate', type=float, default=0,
                             help='when not using random crop'
                                  'apply rotation augmentation.')
    self.parser.add_argument('--flip', type=float, default=0.5,
                             help='probability of applying flip augmentation.')
    self.parser.add_argument('--no_color_aug', action='store_true',
                             help='not use the color augmenation from CornerNet')
    self.parser.add_argument('--nuscenes_interval', type=int, default=1,
                             help='data sampling ratio for faster training')
    self.parser.add_argument('--nuscenes_CBGS', type=self.str2bool, default=True,
                             help='data sampling ratio for faster training')
    # tracking
    self.parser.add_argument('--tracking', action='store_true')
    self.parser.add_argument('--pre_hm', action='store_true')
    self.parser.add_argument('--same_aug_pre', action='store_true')
    self.parser.add_argument('--zero_pre_hm', action='store_true')
    self.parser.add_argument('--hm_disturb', type=float, default=0)
    self.parser.add_argument('--lost_disturb', type=float, default=0)
    self.parser.add_argument('--fp_disturb', type=float, default=0)
    self.parser.add_argument('--pre_thresh', type=float, default=-1)
    self.parser.add_argument('--track_thresh', type=float, default=0.3)
    self.parser.add_argument('--new_thresh', type=float, default=0.3)
    self.parser.add_argument('--max_frame_dist', type=int, default=3)
    self.parser.add_argument('--ltrb_amodel', action='store_true')
    self.parser.add_argument('--ltrb_amodel_weight', type=float, default=0.1)
    self.parser.add_argument('--public_det', action='store_true')
    self.parser.add_argument('--no_pre_img', action='store_true')
    self.parser.add_argument('--zero_tracking', action='store_true')
    self.parser.add_argument('--hungarian', action='store_true')
    self.parser.add_argument('--max_age', type=int, default=-1)

    # loss
    self.parser.add_argument('--tracking_weight', type=float, default=1)
    self.parser.add_argument('--reg_loss', default='l1',
                             help='regression loss: sl1 | l1 | l2')
    self.parser.add_argument('--hm_weight', type=float, default=1,
                             help='loss weight for keypoint heatmaps.')
    self.parser.add_argument('--off_weight', type=float, default=1,
                             help='loss weight for keypoint local offsets.')
    self.parser.add_argument('--wh_weight', type=float, default=0.1,
                             help='loss weight for bounding box size.')
    self.parser.add_argument('--hp_weight', type=float, default=1,
                             help='loss weight for human pose offset.')
    self.parser.add_argument('--hm_hp_weight', type=float, default=1,
                             help='loss weight for human keypoint heatmap.')
    self.parser.add_argument('--amodel_offset_weight', type=float, default=1,
                             help='Please forgive the typo.')
    self.parser.add_argument('--dep_weight', type=float, default=1,
                             help='loss weight for depth.')
    self.parser.add_argument('--dim_weight', type=float, default=1,
                             help='loss weight for 3d bounding box size.')
    self.parser.add_argument('--rot_weight', type=float, default=1,
                             help='loss weight for orientation.')
    self.parser.add_argument('--nuscenes_att', action='store_true')
    self.parser.add_argument('--nuscenes_att_weight', type=float, default=1)
    self.parser.add_argument('--velocity', action='store_true')
    self.parser.add_argument('--velocity_weight', type=float, default=1)

    # CenterNet-Boost configs
    self.parser.add_argument('--set_amodal_center', type=self.str2bool, default=True,
                             help='set projected 3D center as center point if True,'
                                  'use center of 2D bounding box if False.')
    self.parser.add_argument('--discard_distant', type=float, default=60,
                             help='discard distant instance farther than certain range, set -1 to disable')

    self.parser.add_argument('--auxdep', type=self.str2bool, default=True,
                             help='enable auxiliary depth prediction loss.')
    self.parser.add_argument('--auxdep_weight', type=float, default=1,
                             help='loss weight for auxiliary depth.')
    self.parser.add_argument('--auxdep_offset', type=float, default=1.5,
                             help='add offset to depth annotation.')
    self.parser.add_argument('--auxdep_ratio', type=float, default=0.5,
                             help='random point sampling ratio.')
    self.parser.add_argument('--auxdep_mask', type=self.str2bool, default=True,
                             help='foreground object mask')

    self.parser.add_argument('--focaldep', type=self.str2bool, default=True,
                             help='enable focal depth loss.')
    self.parser.add_argument('--focaldep_alpha', type=float, default=0.5,
                             help='balancing parameter.')
    self.parser.add_argument('--focaldep_gamma', type=float, default=2,
                             help='balancing parameter.')

    self.parser.add_argument('--depconf', type=self.str2bool, default=True,
                             help='enable depth uncertainty prediction loss.')
    self.parser.add_argument('--depconf_weight', type=float, default=0.5,
                             help='loss weight for depth uncertainty prediction.')
    self.parser.add_argument('--depconf_beta', type=float, default=2.0,
                             help='balancing parameter.')

    self.parser.add_argument('--deperror_clamp', type=str, default='0.1,1',
                             help='clamp depth error for training stability.')

    # configs for experiments
    self.parser.add_argument('--skip_load', default=[],
                             help='skip loading parameters from pre-trained model.'
                                  "e.g., ['dep', 'dim'] to train dep, dim from init")

    self.parser.add_argument('--freeze_weight', type=self.str2bool, default=False)
    self.parser.add_argument('--freeze_weight_skip', nargs='+', default=[],
                             help='skip freeze branch from pre-trained model.'
                                  "e.g., ['dep'] not to freeze dep branch")

    self.parser.add_argument('--reuse_hm', type=self.str2bool, default=True)
    self.parser.add_argument('--reuse_hm_order', type=str, default='5,0,7',
                             help='reuse heatmap branch from pre-trained model on nuScenes.'
                                  "e.g., '5,0,7' to load ['ped','car','bicycle'] from"
                                  "['car','truck','bus','trailer','const_vehicle',"
                                  " 'ped','motorcycle','bicycle','traffic_cone','barrier']")

    self.parser.add_argument('--eval_depth', default=False,
                             help='evaluate depth prediction performance.')

    # custom dataset
    self.parser.add_argument('--custom_dataset_img_path', default='')
    self.parser.add_argument('--custom_dataset_ann_path', default='')

  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)
  
    if opt.test_dataset == '':
      opt.test_dataset = opt.dataset
    
    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
    opt.save_point = [int(i) for i in opt.save_point.split(',')]
    opt.test_scales = [float(i) for i in opt.test_scales.split(',')]
    opt.save_imgs = [i for i in opt.save_imgs.split(',')] \
      if opt.save_imgs != '' else []
    opt.ignore_loaded_cats = \
      [int(i) for i in opt.ignore_loaded_cats.split(',')] \
      if opt.ignore_loaded_cats != '' else []
    opt.deperror_clamp = [float(i) for i in opt.deperror_clamp.split(',')]
    opt.reuse_hm_order = [int(i) for i in opt.reuse_hm_order.split(',')]

    opt.pre_img = False
    if 'tracking' in opt.task:
      print('Running tracking')
      opt.tracking = True
      opt.out_thresh = max(opt.track_thresh, opt.out_thresh)
      opt.pre_thresh = max(opt.track_thresh, opt.pre_thresh)
      opt.new_thresh = max(opt.track_thresh, opt.new_thresh)
      opt.pre_img = not opt.no_pre_img
      print('Using tracking threshold for out threshold!', opt.track_thresh)
      if 'ddd' in opt.task:
        opt.show_track_color = True

    opt.fix_res = not opt.keep_res
    print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')

    if opt.head_conv == -1: # init default head_conv
      opt.head_conv = 256 if 'dla' in opt.arch else 64

    opt.pad = 127 if 'hourglass' in opt.arch else 31
    opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

    if opt.master_batch_size == -1:
      opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
      slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
      if i < rest_batch_size % (len(opt.gpus) - 1):
        slave_chunk_size += 1
      opt.chunk_sizes.append(slave_chunk_size)
    print('training chunk_sizes:', opt.chunk_sizes)

    if opt.debug > 0:
      opt.num_workers = 0
      opt.batch_size = 1
      opt.gpus = [opt.gpus[0]]
      opt.master_batch_size = -1

    # log dirs
    opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    opt.data_dir = os.path.join(opt.root_dir, 'data')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    
    if opt.resume and opt.load_model == '':
      opt.load_model = os.path.join(opt.save_dir, 'model_last.pth')

    if opt.dataset == 'nuscenes':
      opt.reuse_hm = False

    return opt


  def update_dataset_info_and_set_heads(self, opt, dataset):
    opt.num_classes = dataset.num_categories \
                      if opt.num_classes < 0 else opt.num_classes
    # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
    input_h, input_w = dataset.default_resolution
    input_h = opt.input_res if opt.input_res > 0 else input_h
    input_w = opt.input_res if opt.input_res > 0 else input_w
    opt.input_h = opt.input_h if opt.input_h > 0 else input_h
    opt.input_w = opt.input_w if opt.input_w > 0 else input_w
    opt.output_h = opt.input_h // opt.down_ratio
    opt.output_w = opt.input_w // opt.down_ratio
    opt.input_res = max(opt.input_h, opt.input_w)
    opt.output_res = max(opt.output_h, opt.output_w)
  
    opt.heads = {'hm': opt.num_classes, 'reg': 2, 'wh': 2}

    if 'tracking' in opt.task:
      opt.heads.update({'tracking': 2})

    if 'ddd' in opt.task:
      opt.heads.update({'dep': 1, 'rot': 8, 'dim': 3, 'amodel_offset': 2})
    
    if 'multi_pose' in opt.task:
      opt.heads.update({
        'hps': dataset.num_joints * 2, 'hm_hp': dataset.num_joints,
        'hp_offset': 2})

    if opt.ltrb:
      opt.heads.update({'ltrb': 4})
    if opt.ltrb_amodel:
      opt.heads.update({'ltrb_amodel': 4})
    if opt.nuscenes_att:
      opt.heads.update({'nuscenes_att': 8})
    if opt.velocity:
      opt.heads.update({'velocity': 3})
    if opt.depconf:
      opt.heads.update({'depconf': 1})

    weight_dict = {'hm': opt.hm_weight, 'wh': opt.wh_weight,
                   'reg': opt.off_weight, 'hps': opt.hp_weight,
                   'hm_hp': opt.hm_hp_weight, 'hp_offset': opt.off_weight,
                   'dep': opt.dep_weight, 'rot': opt.rot_weight,
                   'dim': opt.dim_weight,
                   'amodel_offset': opt.amodel_offset_weight,
                   'ltrb': opt.ltrb_weight,
                   'tracking': opt.tracking_weight,
                   'ltrb_amodel': opt.ltrb_amodel_weight,
                   'nuscenes_att': opt.nuscenes_att_weight,
                   'velocity': opt.velocity_weight,
                   'auxdep': opt.auxdep_weight,
                   'depconf': opt.depconf_weight}
    opt.weights = {head: weight_dict[head] for head in opt.heads}
    if opt.auxdep:
      opt.weights.update({'auxdep': weight_dict['auxdep']})
    for head in opt.weights:
      if opt.weights[head] == 0:
        del opt.heads[head]
    opt.head_conv = {head: [opt.head_conv \
      for i in range(opt.num_head_conv if head != 'reg' else 1)] for head in opt.heads}
    
    print('input h w:', opt.input_h, opt.input_w)
    print('heads', opt.heads)
    print('weights', opt.weights)
    print('head conv', opt.head_conv)

    return opt

  def init(self, args=''):
    # only used in demo
    default_dataset_info = {
      'ctdet': 'coco', 'multi_pose': 'coco_hp', 'ddd': 'nuscenes',
      'tracking,ctdet': 'coco', 'tracking,multi_pose': 'coco_hp', 
      'tracking,ddd': 'nuscenes'
    }
    opt = self.parse()
    from dataset.dataset_factory import dataset_factory
    train_dataset = default_dataset_info[opt.task] \
      if opt.task in default_dataset_info else 'coco'
    dataset = dataset_factory[train_dataset]
    opt = self.update_dataset_info_and_set_heads(opt, dataset)
    return opt

  def str2bool(self, v):
      if isinstance(v, bool):
          return v
      if v.lower() in ('yes', 'true', 't', 'y', '1'):
          return True
      elif v.lower() in ('no', 'false', 'f', 'n', '0'):
          return False
      else:
          raise argparse.ArgumentTypeError('Boolean value expected.')