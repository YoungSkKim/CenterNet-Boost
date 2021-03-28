from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from torchvision.ops import roi_align, nms, roi_pool

from model.decode import generic_decode
from model.utils import _sigmoid

def _sigmoid_output(output):
    if 'hm' in output:
        output['hm'] = _sigmoid(output['hm'])
    # if 'cls' in output:
    #     output['cls'] = _sigmoid(output['cls'])
    if 'dep' in output:
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
    return output

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

class BaseModel(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModel, self).__init__()
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        self.num_stacks = num_stacks
        self.heads = heads

        if opt.twostage:
            self.avgpool_2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.maxpool = nn.MaxPool2d(kernel_size=7, stride=7)
            self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)

            for head in self.heads:
                if head in ['hm', 'reg', 'wh', 'auxdep']:  # RPN
                    head_conv = [256]
                elif head in ['dep', 'rot', 'dim', 'amodel_offset']:  # RoI
                    head_conv = [256]
                else:
                    head_conv = []

                classes = self.heads[head]
                if len(head_conv) > 0:
                    conv = nn.Conv2d(last_channel, head_conv[0],
                                     kernel_size=head_kernel,
                                     padding=head_kernel // 2, bias=True)
                    convs = [conv]

                    for k in range(1, len(head_conv)):
                        convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k],
                                               kernel_size=1, bias=True))
                    out = nn.Conv2d(head_conv[-1], classes,
                                    kernel_size=1, stride=1, padding=0, bias=True)
                    if len(convs) == 1:
                      fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
                    elif len(convs) == 2:
                      fc = nn.Sequential(convs[0], nn.ReLU(inplace=True),
                                         convs[1], nn.ReLU(inplace=True))

                else:  # cls (2 fc)
                    out = nn.Sequential(nn.Linear(last_channel, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                        nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                        nn.Linear(512, classes))
                    if 'cls' in head:
                        # out[-1].bias.data.fill_(opt.prior_bias)
                        out[-1].bias.data.fill_(0.0)
                    self.__setattr__(head, out)
                    continue

                if 'hm' in head:
                    fc[-1].bias.data.fill_(opt.prior_bias)
                else:
                    fill_fc_weights(fc)

                self.__setattr__(head, fc)

        else:
            for head in self.heads:
                classes = self.heads[head]
                head_conv = head_convs[head]
                if len(head_conv) > 0:
                  out = nn.Conv2d(head_conv[-1], classes,
                        kernel_size=1, stride=1, padding=0, bias=True)
                  conv = nn.Conv2d(last_channel, head_conv[0],
                                   kernel_size=head_kernel,
                                   padding=head_kernel // 2, bias=True)
                  convs = [conv]
                  for k in range(1, len(head_conv)):
                      convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k],
                                   kernel_size=1, bias=True))
                  if len(convs) == 1:
                    fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
                  elif len(convs) == 2:
                    fc = nn.Sequential(
                      convs[0], nn.ReLU(inplace=True),
                      convs[1], nn.ReLU(inplace=True), out)
                  elif len(convs) == 3:
                    fc = nn.Sequential(
                        convs[0], nn.ReLU(inplace=True),
                        convs[1], nn.ReLU(inplace=True),
                        convs[2], nn.ReLU(inplace=True), out)
                  elif len(convs) == 4:
                    fc = nn.Sequential(
                        convs[0], nn.ReLU(inplace=True),
                        convs[1], nn.ReLU(inplace=True),
                        convs[2], nn.ReLU(inplace=True),
                        convs[3], nn.ReLU(inplace=True), out)
                  if 'hm' in head:
                    fc[-1].bias.data.fill_(opt.prior_bias)
                  elif 'depconf' in head:
                    fc[-1].bias.data.fill_(0)
                  else:
                    fill_fc_weights(fc)
                else:
                  fc = nn.Conv2d(last_channel, classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
                  if 'hm' in head:
                    fc.bias.data.fill_(opt.prior_bias)
                  else:
                    fill_fc_weights(fc)
                self.__setattr__(head, fc)


    def img2feats(self, x):
      raise NotImplementedError
    
    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
      raise NotImplementedError

    def forward(self, x, pre_img=None, pre_hm=None, is_trainer=None):
      if (pre_hm is not None) or (pre_img is not None):
        feats = self.imgpre2feats(x, pre_img, pre_hm)
      else:
        feats = self.img2feats(x)
      out = []
      if self.opt.model_output_list:
        for s in range(self.num_stacks):
          z = []
          for head in sorted(self.heads):
              z.append(self.__getattr__(head)(feats[s]))
          out.append(z)

      elif self.opt.twostage:
        for s in range(self.num_stacks):
          z = {}
          head_RPN = ['hm', 'wh', 'reg', 'amodel_offset', 'rot', 'dim']
          for head in self.heads:
              if head in head_RPN:
                z[head] = self.__getattr__(head)(feats[s])
          z = _sigmoid_output(z)
          dets = generic_decode(z, K=self.opt.K, opt=self.opt)
          if is_trainer:
            for det in ['bboxes']:
              z[det] = dets[det]
          else:
            for det in dets:
              z[det] = dets[det]
          batch_size = z['hm'].shape[0]
          idx_batch = torch.arange(0, batch_size).float().cuda().repeat_interleave(self.opt.K).view(-1, 1)
          z['keep_idx_pos'], z['keep_idx_neg'] = {}, {}
          for b in range(batch_size):
            if self.opt.RPN_nms:
                keep_idx = torch.zeros_like(dets['scores'][b], dtype=torch.bool)
                keep_idx_nms = nms(dets['bboxes'][b], dets['scores'][b], iou_threshold=self.opt.RPN_nms_threshold)
                keep_idx[[keep_idx_nms]] = True
                # Remove low score
                keep_idx_score = dets['scores'][b]
                keep_idx_score_valid = keep_idx_score > self.opt.RPN_score_threshold
                # Remove small boxes
                bboxes_width = dets['bboxes'][b][:, 2] - dets['bboxes'][b][:, 0]
                keep_idx_width_valid = bboxes_width > self.opt.RPN_width_threshold / self.opt.down_ratio
                keep_idx_valid = keep_idx_score_valid * keep_idx_width_valid * keep_idx
                z['keep_idx_pos'][b] = (keep_idx_valid == True).nonzero().view(-1)
                z['keep_idx_neg'][b] = ~keep_idx_score_valid * keep_idx_width_valid * keep_idx
            else:
                # Remove low score
                keep_idx_score = dets['scores'][b]
                keep_idx_score_valid = keep_idx_score > self.opt.RPN_score_threshold
                # Remove small boxes
                bboxes_width = dets['bboxes'][b][:, 2] - dets['bboxes'][b][:, 0]
                keep_idx_width_valid = bboxes_width > self.opt.RPN_width_threshold / self.opt.down_ratio
                keep_idx_valid = keep_idx_score_valid * keep_idx_width_valid
                z['keep_idx_pos'][b] = (keep_idx_valid == True).nonzero().view(-1)
                z['keep_idx_neg'][b] = ~keep_idx_score_valid * keep_idx_width_valid
          rois = roi_align(feats[s],
                           boxes=torch.cat((idx_batch, dets['bboxes'].view(-1, 4)), dim=1),
                           output_size=self.opt.RPN_RoI_size)
          if self.opt.RPN_RoI_pool == 'avg':
              rois = self.avgpool(rois)
          elif self.opt.RPN_RoI_pool == 'max':
              rois = self.maxpool(rois)
          for head in self.heads:
              if head in ['dep']:
                  if head == 'cls':
                      rois = rois.view(rois.shape[0], -1)
                  tmp = self.__getattr__(head)(rois)
                  z[head] = tmp.view(batch_size, self.opt.K, -1)
          out.append(z)

      else:
        for s in range(self.num_stacks):
          z = {}
          for head in self.heads:
              z[head] = self.__getattr__(head)(feats[s])
          out.append(z)
      return out
