from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
sys.path.append("./lib/model/networks/DCNv2")

import torch
import torch.utils.data
from opts import opts
from model.model import create_model, load_model, save_model
from model.data_parallel import DataParallel
from logger import Logger
from dataset.dataset_factory import get_dataset
from trainer import Trainer

def get_optimizer(opt, model):
  optimizer = torch.optim.Adam(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
  return optimizer

def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  if not opt.not_set_cuda_env:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  logger = Logger(opt)

  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
  optimizer = get_optimizer(opt, model)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, opt, optimizer)

  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up validation data...')
  val_loader = torch.utils.data.DataLoader(
    Dataset(opt, 'val'), batch_size=1, shuffle=False, num_workers=1,
    pin_memory=True)

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  print('Starting eval...')
  opt.eval_depth = True
  with torch.no_grad():
    log_dict_val, preds = trainer.val(1, val_loader)
    if opt.eval_val:
      val_loader.dataset.run_eval(preds, opt.save_dir)


if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
