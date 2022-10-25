import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init
from torch.nn.parameter import Parameter
from torch.nn import init

from math import ceil

from mmdet2.core import auto_fp16, force_fp32
from mmdet2.ops import ConvModule, DepthwiseSeparableConvModule, build_upsample_layer
from ..registry import HEADS

DLFSE_max = 3
DDPC_max = 24
REN_CH = [8, 128]

def clamp(after_v, x_padded):
    after_v = torch.min(torch.ones(after_v.shape).cuda(after_v.device)*((x_padded.shape[-2])*(x_padded.shape[-1])-1),
                        torch.max(torch.zeros(after_v.shape).cuda(after_v.device), after_v))
    return after_v

def get_val(x_padded, after_v):
    B, C, H, W = x_padded.shape[0], x_padded.shape[1], x_padded.shape[-2], x_padded.shape[-1]
    after_v = torch.cat(x_padded.shape[1] * [after_v.view(-1, (x_padded.shape[-1]-2) * (x_padded.shape[-2]-2) * 9).unsqueeze(1)], 1).long()
    x_padded = torch.gather(x_padded.view(x_padded.shape[0], x_padded.shape[1],-1), 2, after_v)
    return x_padded.view(B, C, (H-2)*(W-2), 9)

def shift_x(x, x_range, D_max, range_out):
    x_range_x = x_range[:, 0:1, ...]
    x_range_y = x_range[:, 1:2, ...]
    offset_x = D_max * ((x_range_x - x_range_x.min()) / (x_range_x.max()-x_range_x.min()))
    offset_y = D_max * ((x_range_y - x_range_y.min()) / (x_range_y.max()-x_range_y.min()))
    B, H, W = x.shape[0], x.shape[-2], x.shape[-1]

    x_off = torch.from_numpy(np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])).cuda(x.device)
    y_off = torch.from_numpy(np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])).cuda(x.device)

    v, _ = torch.meshgrid([torch.arange(0, (x.shape[-2] + 2 ) *(x.shape[-1] + 2)), torch.arange(0, 9)])
    v = v.cuda(x.device) 
    pre_v = v + x_off + (y_off * x.shape[-1] + 2)
    pre_v = pre_v.view(x.shape[-2] + 2, x.shape[-1] + 2, 9)[1:x.shape[-2] + 1,1:x.shape[-1] + 1].contiguous().view(-1,9)
    pre_v = torch.cat(x.shape[0] * [pre_v.unsqueeze(0)], 0)

    offset_v = torch.matmul(offset_x.view(B, H*W, 1), x_off.float().view(1,9)) \
                            + torch.matmul(offset_y.view(B, H*W, 1), y_off.float().view(1, 9) * (x.shape[-1] + 2))
    x_padded = F.pad(x, (1, 1 , 1, 1), mode='constant')

    after_v = pre_v + offset_v
    after_v_f = clamp(pre_v + torch.floor(offset_v), x_padded)
    after_v_f1 = clamp(after_v_f + x_off,x_padded)
    after_v_c = clamp(pre_v + torch.ceil(offset_v), x_padded)
    after_v_c1 = clamp(after_v_c + x_off, x_padded)

    val_f = get_val(x_padded, after_v_f)
    val_f1 = get_val(x_padded, after_v_f1)
    val_c = get_val(x_padded, after_v_c)
    val_c1 = get_val(x_padded, after_v_c1)

    f1 = (torch.abs(after_v - after_v_f)).unsqueeze(1) * val_f +(torch.abs(after_v_f1 - after_v)).unsqueeze(1) * val_f1
    f2 = (torch.abs(after_v - after_v_c1)).unsqueeze(1) * val_c1 +(torch.abs(after_v_c - after_v)).unsqueeze(1) * val_c
    x_padded = torch.abs((after_v - after_v_f) / x_padded.shape[-1]).unsqueeze(1) * f1 \
                         + torch.abs((after_v_c1 - after_v) / (x_padded.shape[-1])).unsqueeze(1) * f2

    x_padded = torch.sum(x_padded * range_out.view(1, x.shape[1], 1, 9), -1).view(x_padded.shape[0], x.shape[1], x.shape[-2], x.shape[-1]).cuda(x.device)    
    return x_padded

class MC(torch.nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg):
        super().__init__()
        self.in_channels = in_channels
        self.conv_out_channels = out_channels
        self.conv_kernel_size = 3
        self.convs = nn.ModuleList()
        for i in range(2):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                DepthwiseSeparableConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        x = self.convs(x)
        x = F.interpolate(x, size=(x.shape[-2]*2, x.shape[-1]*2), 
                          mode='bilinear', align_corners=False)

        return x


class LSFE(torch.nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg, flag=False):
        super().__init__()
        self.in_channels = in_channels
        self.conv_out_channels = out_channels
        self.conv_kernel_size = 3
        self.flag = flag

        if self.flag:
            self.range_out = Parameter(torch.Tensor(in_channels, 1, *(3, 3)))
            init.kaiming_uniform_(self.range_out, a=math.sqrt(5))
            self.range_conv = ConvModule(
                    self.in_channels,
                    self.conv_out_channels,
                    1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            self.in_channels = out_channels

        self.convs = nn.ModuleList()
        for i in range(2):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                DepthwiseSeparableConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.convs = nn.Sequential(*self.convs)

    def forward(self, x, x_range):
        if self.flag:
            x = shift_x(x, x_range, DLFSE_max, self.range_out)
            x = self.range_conv(x)
        x = self.convs(x)
        return x


class DPC(torch.nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg, flag=False):
        super().__init__()
        self.in_channels = in_channels
        self.concat_channels = in_channels * 5
        self.conv_out_channels = out_channels
        self.conv_kernel_size = 3
        self.flag = flag

        if self.flag:
            self.range_out_a = Parameter(torch.Tensor(in_channels, 1, *(3, 3)))
            self.range_out_b = Parameter(torch.Tensor(in_channels, 1, *(3, 3)))
            init.kaiming_uniform_(self.range_out_a, a=math.sqrt(5))
            init.kaiming_uniform_(self.range_out_b, a=math.sqrt(5))
            self.range_conv_a = ConvModule(
                    self.in_channels,
                    self.in_channels // 2,
                    1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            self.range_conv_b = ConvModule(
                    self.in_channels,
                    self.in_channels // 2,
                    1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            self.merge = ConvModule(
                    self.in_channels,
                    self.in_channels,
                    1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            self.concat_channels = in_channels * 6


        self.convs = nn.ModuleList()
        dilations = [(1,6), (1,1), (6,21), (18,15), (6,3)]

        for i in range(5):
            padding = dilations[i]
            self.convs.append(
                DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.in_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    dilation=dilations[i]))
        self.conv = ConvModule(
                    self.concat_channels,
                    self.conv_out_channels,
                    1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)

    def forward(self, x, x_range):
        x_u = self.convs[0](x)
        x1 = self.convs[1](x_u)
        x2 = self.convs[2](x_u)
        x3 = self.convs[3](x_u)
        x4 = self.convs[4](x3)
        x_f = torch.cat([
            x,
            x1,
            x2,
            x3,
            x4 
            ], dim=1)

        if self.flag:
            x_a = shift_x(x, x_range, DDPC_max, self.range_out_a)
            x_a = self.range_conv_a(x_a)
            x_b = shift_x(x_u, x_range, DDPC_max, self.range_out_b)
            x_b = self.range_conv_a(x_b)
            x_m = self.merge(torch.cat([x_a, x_b], dim=1))
            x_f = torch.cat([x_f, x_m], dim=1)

        x = self.conv(x_f) 
        return x



@HEADS.register_module
class EfficientLPSSemanticHead(nn.Module):
    def __init__(self,
                 in_channels=256,
                 conv_out_channels=128,
                 num_classes=183,
                 ignore_label=255,
                 loss_weight=1.0,
                 ohem = 0.25,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(EfficientLPSSemanticHead, self).__init__()
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.ohem = ohem
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.fp16_enabled = False
    
        if self.ohem is not None:
            assert (self.ohem >= 0 and self.ohem < 1)

        self.range_offset = nn.ModuleList()
        self.lateral_convs_ss = nn.ModuleList()
        self.lateral_convs_ls = nn.ModuleList()
        self.aligning_convs = nn.ModuleList()
        self.ss_idx = [3,2]
        self.ls_idx = [1,0]
        self.flag = [True, False] 

        for i in range(2):
            self.range_offset.append(nn.Conv2d(REN_CH[i], 2, 3, padding=1))

        for i in range(2):
            self.lateral_convs_ss.append(
                DPC(
                    self.in_channels,
                    self.conv_out_channels,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    flag=self.flag[i]))

        for i in range(2):
            self.lateral_convs_ls.append(
                LSFE(
                    self.in_channels,
                    self.conv_out_channels,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    flag=self.flag[i]))

        for i in range(2):
            self.aligning_convs.append(
                  MC(
                    self.conv_out_channels,
                    self.conv_out_channels,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.conv_logits = nn.Conv2d(conv_out_channels * 4, self.num_classes, 1)

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction='none')

    def init_weights(self):
        kaiming_init(self.conv_logits)

    def forward(self, feats, range_feats):
        feats = list(feats)
        range_feats = list(range_feats)
        ref_size = tuple(feats[0].shape[-2:])

        r_off = self.range_offset[1](range_feats[3])
        for idx, lateral_conv_ss in zip(self.ss_idx, self.lateral_convs_ss):
            feats[idx] = lateral_conv_ss(feats[idx], r_off)
            r_off = None
       
        x = self.aligning_convs[0](feats[self.ss_idx[1]] + F.interpolate(
                    feats[self.ss_idx[0]], size=tuple(feats[self.ss_idx[1]].shape[-2:]), 
                    mode='bilinear', align_corners=False))

        r_off = self.range_offset[0](range_feats[1])

        for idx, lateral_conv_ls in zip(self.ls_idx, self.lateral_convs_ls):
            feats[idx] = lateral_conv_ls(feats[idx], r_off)
            feats[idx] = feats[idx] + F.interpolate(
                      x, size=tuple(feats[idx].shape[-2:]), 
                      mode='bilinear', align_corners=False)
            if idx != 0:
                x = self.aligning_convs[1](feats[idx])
                r_off = None

        for i in range(1,4):
            feats[i] = F.interpolate(
                      feats[i], size=ref_size, 
                      mode='bilinear', align_corners=False)

        x = torch.cat(feats, dim=1)
        x = self.conv_logits(x)
        x = F.interpolate(
                      x, size=(ref_size[0]*4, ref_size[1]*4), 
                      mode='bilinear', align_corners=False)

        return x


    def loss(self, mask_pred, labels):
        loss = dict()
        labels = labels.squeeze(1).long()
        loss_semantic_seg = self.criterion(mask_pred, labels)
        loss_semantic_seg = loss_semantic_seg.view(-1)

        if self.ohem is not None:
            top_k = int(ceil(loss_semantic_seg.numel() * self.ohem))
            if top_k != loss_semantic_seg.numel():
                    loss_semantic_seg, _ = loss_semantic_seg.topk(top_k)
 
        loss_semantic_seg = loss_semantic_seg.mean()
        loss_semantic_seg *= self.loss_weight
        loss['loss_semantic_seg'] = loss_semantic_seg
        return loss
