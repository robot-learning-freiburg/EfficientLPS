from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

import geffnet
import os

from mmdet2.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, multiclass_nms)
from .. import builder
from ..registry import EFFICIENTLPS
from .base import BaseDetector
from .KNN import KNN
try:
    from mmdet2.ops.norm import norm_cfg
    from mmdet2.ops.roi_sampling import roi_sampling, invert_roi_bbx
except ImportError:
    from mmdet.ops.norm import norm_cfg
    from mmdet.ops.roi_sampling import roi_sampling, invert_roi_bbx

import time
import cv2
import numpy as np

import PIL.Image as Image




@EFFICIENTLPS.register_module
class EfficientLPS(BaseDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 semantic_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        assert backbone is not None
        assert rpn_head is not None
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert mask_roi_extractor is not None           
        assert mask_head is not None
        assert semantic_head is not None

        super(EfficientLPS, self).__init__()

        self.eff_backbone_flag = False if 'efficient' not in backbone['type'] else True

        if self.eff_backbone_flag == False:
            self.backbone = builder.build_backbone(backbone)
        else:
            self.backbone = geffnet.create_model(backbone['type'], 
                                                 pretrained=True if pretrained is not None else False,
                                                 se=False,
                                                 in_chans=backbone['in_chans'], 
                                                 act_layer=backbone['act_cfg']['type'],
                                                 norm_layer=norm_cfg[backbone['norm_cfg']['type']][1],
                                                 proximity=True)

            self.ren = geffnet.create_model('tf_efficientnet_ren', 
                                            pretrained=False,
                                            se=False,
                                            in_chans=1, 
                                            act_layer=backbone['act_cfg']['type'],
                                            norm_layer=norm_cfg[backbone['norm_cfg']['type']][1],
                                            proximity=False)

        
        self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        self.rpn_head = builder.build_head(rpn_head)

        self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
        self.bbox_head = builder.build_head(bbox_head)

        self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
        self.share_roi_extractor = True
        self.mask_head = builder.build_head(mask_head)

        self.semantic_head = builder.build_head(semantic_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
       
        self.num_classes = semantic_head['num_classes']
        self.num_stuff = self.num_classes - bbox_head['num_classes'] + 1
        self.post = KNN(self.num_classes)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if self.eff_backbone_flag == False:
            self.backbone.init_weights(pretrained=pretrained)

        self.neck.init_weights()

        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)

        self.rpn_head.init_weights()
        self.bbox_roi_extractor.init_weights()
        self.bbox_head.init_weights()
        self.mask_head.init_weights()
        self.mask_roi_extractor.init_weights()
        self.semantic_head.init_weights() 

    def extract_feat(self, img, mean=None, std=None):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img, mean, std)
        x_range = self.ren(img[:, 0:1, ...], None, None)
        x = self.neck(x, x_range)
        return x, x_range

    def forward_dummy(self, img): #leave it for now
        """Used for computing network flops.

        See `mmdet2ection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(device=img.device)
        # bbox head
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            outs = outs + (cls_score, bbox_pred)
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], mask_rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            outs = outs + (cls_score, bbox_pred)
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            outs = outs + (mask_pred, )
        return outs

    def assign_result(self, x, proposal_list,
                      img, gt_bboxes, gt_labels, gt_bboxes_ignore):
        bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
        bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
        num_imgs = img.size(0)
        if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
            sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        return sampling_results

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None):

        x, x_range = self.extract_feat(img, 
                              img_metas[0]['sensor_img_means'][0], 
                              img_metas[0]['sensor_img_stds'][0])
        losses = dict()
        semantic_logits = self.semantic_head(x[:4], x_range[:4])
        loss_seg = self.semantic_head.loss(semantic_logits, gt_semantic_seg)
        losses.update(loss_seg)
        instance_flags = [len(gt_label)!= 0 for gt_label in gt_labels] 

        if np.sum(instance_flags) == 0:
            loss_i = sum(x_i.sum() for x_i in x) * 0
            instance_losses = dict(loss_rpn_cls=loss_i,
                                   loss_rpn_bbox=loss_i,
                                   loss_cls=loss_i,
                                   loss_bbox=loss_i,
                                   loss_mask=loss_i)
            losses.update(instance_losses)
            return losses

        rpn_outs = self.rpn_head(x)
        rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas,
                                          self.train_cfg.rpn)
        rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(rpn_losses)

        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
        proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

        sampling_results =  self.assign_result(x, proposal_list, img,
                               gt_bboxes, gt_labels, gt_bboxes_ignore)

        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_targets = self.bbox_head.get_target(sampling_results,
                                                 gt_bboxes, gt_labels,
                                                 self.train_cfg.rcnn)
        loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                        *bbox_targets)
        losses.update(loss_bbox)


        pos_rois = bbox2roi(
                [res.pos_bboxes for res in sampling_results])
        mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], pos_rois)
        if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)

        if mask_feats.shape[0] > 0:
            mask_pred = self.mask_head(mask_feats)
            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        return losses



    def simple_test(self, img, img_metas, proposals=None, rescale=False, eval=None, return_pred=False):
        x, x_range = self.extract_feat(img, 
                              img_metas[0]['sensor_img_means'][0], 
                              img_metas[0]['sensor_img_stds'][0])
        semantic_logits = self.semantic_head(x[:4], x_range[:4])
        result = []
        if semantic_logits.shape[0] == 1:
            proposal_list = self.simple_test_rpn(x, img_metas,
                                     self.test_cfg.rpn)

            det_bboxes, det_labels = self.simple_test_bboxes(x, 
                img_metas, proposal_list, self.test_cfg.rcnn, rescale=rescale)
            if eval is not None:
                if return_pred:
                    pred_ = self.simple_test_mask_(
                        x, img_metas, det_bboxes, det_labels, semantic_logits, rescale=rescale, return_pred=return_pred) 
                    result.append([pred_, img_metas])
                else:      
                    panoptic_mask, cat_ = self.simple_test_mask_(
                        x, img_metas, det_bboxes, det_labels, semantic_logits, rescale=rescale, return_pred=return_pred)
                    result.append([panoptic_mask, cat_, img_metas])
        
            else:          
                bbox_results = bbox2result(det_bboxes, det_labels,
                                           self.bbox_head.num_classes)
                mask_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, semantic_logits, rescale=rescale)

                return bbox_results, mask_results
        else:
            for i in range(len(img_metas)):
                new_x = []
                for x_i in x:
                    new_x.append(x_i[i:i+1])
                proposal_list = self.simple_test_rpn(new_x, [img_metas[i]],
                                     self.test_cfg.rpn)

                assert eval is not None

                det_bboxes, det_labels = self.simple_test_bboxes(new_x, 
                    [img_metas[i]], proposal_list, self.test_cfg.rcnn, rescale=rescale)
                if return_pred:
                    pred_ = self.simple_test_mask_(
                        new_x, [img_metas[i]], det_bboxes, det_labels, semantic_logits[i:i+1], rescale=rescale, return_pred=return_pred) 
                    result.append([pred_, [img_metas[i]]])
                else:
                    panoptic_mask, cat_ = self.simple_test_mask_(
                        new_x, [img_metas[i]], det_bboxes, det_labels, semantic_logits[i:i+1], rescale=rescale, return_pred=return_pred)
                    result.append([panoptic_mask, cat_, [img_metas[i]]])

        return result

    def aug_test(self,):
        pass


    def simple_test_rpn(self, x, img_metas, rpn_test_cfg):
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_metas, rpn_test_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def simple_test_bboxes(self,
                    x,
                    img_metas,
                    proposals,
                    rcnn_test_cfg,
                    rescale=False):

        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def simple_test_mask(self,
                  x,
                  img_metas,
                  det_bboxes,
                  det_labels,
                  semantic_logits, 
                  rescale=False):
        print("I have reached simple_test mask' function")
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)

            segm_result = self.mask_head.get_seg_masks(mask_pred, _bboxes,
                                                       det_labels,
                                                       self.test_cfg.rcnn,
                                                       ori_shape, scale_factor,
                                                       rescale)
        return segm_result

    def simple_test_mask_(self,
                  x,
                  img_metas,
                  det_bboxes,
                  det_labels,
                  semantic_logits, 
                  rescale=False,
                  return_pred=False):

        stuff_id = img_metas[0]['stuff_id']
        start = 0
        end = self.num_stuff
        start_t = self.num_stuff
        end_t = self.num_classes
        offset = 0
        if stuff_id != 0:
            start = stuff_id
            end = stuff_id + self.num_stuff 
            offset = self.num_stuff
            start_t = 0
            end_t = self.num_classes - self.num_stuff 

        ori_shape = img_metas[0]['ori_shape'][1:]
        scale_factor = img_metas[0]['scale_factor']
        ref_size = (np.int(np.round(ori_shape[0]*scale_factor[1])), 
                    np.int(np.round(ori_shape[1]*scale_factor[0])))
        semantic_logits = F.interpolate(semantic_logits, size=ref_size, 
                                   mode="bilinear", align_corners=False)   
        sem_pred = torch.argmax(torch.cat([semantic_logits[0:1, start:end], 
                                semantic_logits[0:1, start_t:end_t]], dim=1), dim=1)[0]
        panoptic_mask = torch.zeros_like(sem_pred, dtype=torch.long)
        cat = [255]

        if det_bboxes.shape[0] == 0:
            intermediate_logits = semantic_logits[0, start:end] 
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
            mask_pred = self.mask_head(mask_feats)
            confidence = det_bboxes[:,4]
            idx = torch.argsort(confidence, descending=True)
            bbx_inv = invert_roi_bbx(_bboxes, 
                      tuple(mask_pred.shape[-2:]), ref_size)
            bbx_idx = torch.arange(0, det_bboxes.size(0), 
                      dtype=torch.long, device=det_bboxes.device)
            
            mask_pred = roi_sampling(mask_pred, bbx_inv, bbx_idx, 
                        ref_size, padding="zero")

            ML_A = mask_pred.new_zeros(mask_pred.shape[0], mask_pred.shape[-2], 
                                             mask_pred.shape[-1])
            ML_B = ML_A.clone()             
            occupied = torch.zeros_like(sem_pred, dtype=torch.bool)
            i = 0 
            for id_i in idx:
                label_i = det_labels[id_i] 
                mask_pred_i = mask_pred[id_i, label_i+1, :, :]
                mask_i = (mask_pred_i.sigmoid() > self.test_cfg.rcnn.mask_thr_binary) 
                mask_i = mask_i.type(torch.bool)
                intersection = occupied & mask_i
                if intersection.float().sum() / mask_i.float().sum() > self.test_cfg.panoptic.overlap_thr:
                    continue

                mask_i = mask_i ^ intersection
                occupied += mask_i

                y0 = max(int(det_bboxes[id_i, 1] + 1), 0)
                y1 = min(int((det_bboxes[id_i, 3] - 1).round() + 1), ref_size[0])
                x0 = max(int(det_bboxes[id_i, 0] + 1), 0)
                x1 = min(int((det_bboxes[id_i, 2] - 1).round() + 1), ref_size[1])

                ML_A[i] = 4 * mask_pred_i
                ML_B[i, y0: y1, x0: x1] = semantic_logits[0, label_i + self.num_stuff - offset, y0: y1, x0: x1]
                cat.append(label_i.item() + self.num_stuff - offset)
                i = i + 1 

            ML_A = ML_A[:i]
            ML_B = ML_B[:i]
            FL = (ML_A.sigmoid() + ML_B.sigmoid())*(ML_A + ML_B)
            intermediate_logits = torch.cat([semantic_logits[0, start:end], FL], dim=0)

        cat = torch.tensor(cat, dtype=torch.long)
        intermediate_mask = torch.argmax(F.softmax(intermediate_logits, dim=0), dim=0) + 1
        intermediate_mask = intermediate_mask - self.num_stuff
        intermediate_mask[intermediate_mask <= 0] = 0         
        unique = torch.unique(intermediate_mask) 
        ignore_val = intermediate_mask.max().item() + 1
        ignore_arr = torch.ones((ignore_val,), dtype=unique.dtype, device=unique.device) * ignore_val
        total_unique = unique.shape[0]
        ignore_arr[unique] = torch.arange(total_unique).cuda(ignore_arr.device)  
        panoptic_mask = ignore_arr[intermediate_mask]
        panoptic_mask[intermediate_mask == ignore_val] = 0 

        cat_ = cat[unique].long()
        sem_pred[panoptic_mask > 0] = self.num_stuff
        sem_pred[sem_pred >= self.num_stuff] = self.num_stuff
        cls_stuff, area = torch.unique(sem_pred, return_counts=True)
        cls_stuff[area < self.test_cfg.panoptic.min_stuff_area] = self.num_stuff
        cls_stuff = cls_stuff[cls_stuff!=self.num_stuff]     

        tmp = torch.ones((self.num_stuff + 1,), dtype=cls_stuff.dtype, device=cls_stuff.device) * self.num_stuff
        tmp[cls_stuff] = torch.arange(cls_stuff.shape[0]).cuda(tmp.device)  
        new_sem_pred = tmp[sem_pred]
        bool_mask = new_sem_pred != self.num_stuff   
        panoptic_mask[bool_mask] = new_sem_pred[bool_mask] + total_unique 
        #remap
        cat_ = cat_ - start_t
        cat_ = torch.cat((cat_, (cls_stuff + stuff_id).cpu().long()), -1)   
        #back to 3D
        prediction_img = Image.fromarray(np.uint8(panoptic_mask.cpu().numpy()))
        prediction_img = prediction_img.resize(ori_shape[::-1], resample=Image.NEAREST)
        prediction_img = np.array(prediction_img, dtype=np.uint32, copy=False)
        cat = cat_.cpu().numpy()
        prediction_img, cat = self.compact_labels(prediction_img, cat)
        cat = torch.from_numpy(cat.astype(np.long))
        prediction_img = torch.from_numpy(prediction_img.astype(np.long))
        device = panoptic_mask.device
        proj_argmax = prediction_img.cuda(device)
        n_points = img_metas[0]['unproj_n_points']
        unproj_argmax = self.post(img_metas[0]['proj_range'][:n_points].cuda(device), 
                                  img_metas[0]['unproj_range'][:n_points].cuda(device), proj_argmax, 
                                  img_metas[0]['proj_x'][:n_points].cuda(device), 
                                  img_metas[0]['proj_y'][:n_points].cuda(device), 
                                  torch.max(proj_argmax) + 1)
        unprojsem_argmax = cat[unproj_argmax] 
        pred_np = unprojsem_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.uint32)
        pred_np = np.uint32(unproj_argmax.cpu().numpy().reshape((-1)).astype(np.uint32) << 16 | pred_np)
        if return_pred:
            return pred_np
        path = self.get_path(img_metas)
        pred_np.tofile(path)

        return panoptic_mask.cpu(), cat_.cpu()

    def compact_labels(self, msk, cat):
        ids = np.unique(msk)
        if 0 not in ids:
            ids = np.concatenate((np.array([0], dtype=np.int32), ids), axis=0)

        ids_to_compact = np.zeros((ids.max() + 1,), dtype=np.int32)
        ids_to_compact[ids] = np.arange(0, ids.size, dtype=np.int32)

        msk = ids_to_compact[msk]
        cat = cat[ids]

        return msk, cat

    def get_path(self, metas):
        filename = metas[0]['filename']
        tmpDir = 'tmpDir'
        os.makedirs(tmpDir, exist_ok = True)
        base_path = os.path.join(tmpDir, filename.split('/')[-3])
        os.makedirs(base_path, exist_ok = True)
        pred_base_path = os.path.join(base_path, 'predictions')
        os.makedirs(pred_base_path, exist_ok = True)
        final_path = os.path.join(pred_base_path, filename.split('/')[-1].replace('.bin', '.label'))
        return final_path
