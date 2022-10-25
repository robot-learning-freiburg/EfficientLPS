import os.path as osp

import mmcv
import torch
import numpy as np
import pycocotools.mask as maskUtils

from ..registry import PIPELINES
from ..laserscan_unfolding import LaserScan, SemLaserScan
import cv2

@PIPELINES.register_module
class LoadImageFromFile(object):

    def __init__(self, to_float32=False, color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img = mmcv.imread(filename, self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        return '{} (to_float32={}, color_type={})'.format(
            self.__class__.__name__, self.to_float32, self.color_type)

@PIPELINES.register_module
class LoadLidarFromFile(object):

    def __init__(self, project=True, H=64, W=2048, fov_up=3.0, fov_down=-25.0, gt=False, max_points=150000,
                sensor_img_means = [12.12, 10.88, 0.23, -1.04, 0.21], sensor_img_stds = [12.32, 11.47, 6.91, 0.86, 0.16]):
        self.gt_flag = gt
        self.max_points = max_points
        self.sensor_img_means = torch.tensor(sensor_img_means,
                                         dtype=torch.float).view(-1,1,1)
        self.sensor_img_stds = torch.tensor(sensor_img_stds,
                                        dtype=torch.float).view(-1,1,1)

        if self.gt_flag:
            self.scan = SemLaserScan(project=project, H=H, W=W, fov_up=fov_up, fov_down=fov_down)
        else:
            self.scan = LaserScan(project=project, H=H, W=W, fov_up=fov_up, fov_down=fov_down)


    def __call__(self, results):
        filename = results['img_info']['filename']
        self.scan.open_scan(filename)
        unproj_n_points = self.scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(self.scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(self.scan.unproj_range)
        unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(self.scan.remissions)

        # get points and labels
        proj_range = torch.from_numpy(self.scan.proj_range).clone()
        proj_xyz = torch.from_numpy(self.scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(self.scan.proj_remission).clone()
        proj_mask = torch.from_numpy(self.scan.proj_mask)
        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(self.scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(self.scan.proj_y)
        proj = torch.cat([proj_range.unsqueeze(0).clone(),
                      proj_xyz.clone().permute(2, 0, 1),
                      proj_remission.unsqueeze(0).clone()])
        proj = (proj - self.sensor_img_means
            ) / self.sensor_img_stds
        proj = proj * proj_mask.float()


        results['filename'] = filename
        results['img'] = proj
        results['img_shape'] = proj.shape
        results['ori_shape'] = proj.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = proj.shape
        results['scale_factor'] = 1.0
        
        results['proj_x'] = proj_x
        results['proj_y'] = proj_y
        results['proj_msk'] = proj_mask
        results['proj_range'] = proj_range
        results['unproj_range'] = unproj_range
        results['unproj_n_points'] = unproj_n_points
        if self.gt_flag:
            class_lut = results['class_lut']
            self.scan.open_label(results['img_info']['ann'])
            sem = class_lut[self.scan.proj_sem_label]
            sem[sem == 0] = 255
            sem[sem != 255] = sem[sem != 255] - 1
            self._load_semantic_seg(results, sem)
            self._load_instance_seg(results, self.scan.proj_inst_label, sem)

        results['sensor_img_means'] = self.sensor_img_means
        results['sensor_img_stds'] = self.sensor_img_stds

        return results

    def _load_semantic_seg(self, results, sem):
        results['gt_semantic_seg'] = sem
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def _load_instance_seg(self, results, inst, sem):
        mapped_inst = sem * 1000 + inst
        gt_labels = []
        gt_masks = []
        bboxes = []
        for inst_i in np.unique(mapped_inst):
            if inst_i != 255000 and inst_i // 1000 < results['stuff_id']:                
                gt_labels.append((inst_i // 1000) + 1)
                mask = mapped_inst == inst_i
                gt_masks.append(np.uint8(mask))
                # bbox computation for a segment
                hor = np.sum(mask, axis=0)
                hor_idx = np.nonzero(hor)[0]
                x = hor_idx[0]
                width = hor_idx[-1] - x + 1
                vert = np.sum(mask, axis=1)
                vert_idx = np.nonzero(vert)[0]
                y = vert_idx[0]
                height = vert_idx[-1] - y + 1
                bbox = [int(x), int(y), int(x + width), int(y + height)]
                bboxes.append(bbox)            

        results['gt_bboxes'] = bboxes
        results['gt_labels'] = gt_labels
        results['bbox_fields'].append('gt_bboxes')
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')

        return results              


@PIPELINES.register_module
class LoadMultiChannelImageFromFiles(object):
    """ Load multi channel images from a list of separate channel files.
    Expects results['filename'] to be a list of filenames
    """

    def __init__(self, to_float32=True, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = [
                osp.join(results['img_prefix'], fname)
                for fname in results['img_info']['filename']
            ]
        else:
            filename = results['img_info']['filename']
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        return '{} (to_float32={}, color_type={})'.format(
            self.__class__.__name__, self.to_float32, self.color_type)


@PIPELINES.register_module
class LoadAnnotations(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes']

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        results['gt_semantic_seg'] = mmcv.imread(
            osp.join(results['seg_prefix'], results['ann_info']['seg_map']),
            flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={},'
                     ' with_seg={})').format(self.with_bbox, self.with_label,
                                             self.with_mask, self.with_seg)
        return repr_str


@PIPELINES.register_module
class LoadProposals(object):

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                'but found {}'.format(proposals.shape))
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0]], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(num_max_proposals={})'.format(
            self.num_max_proposals)
