import os.path as osp

import os
import mmcv
import yaml
import numpy as np
from torch.utils.data import Dataset


from mmdet2.core import eval_map, eval_recalls
from mmdet2.utils import print_log
from .pipelines import Compose
from .registry import DATASETS
from .eval_np import PanopticEval

WIDTH = 64
HEIGHT = 2048
STUFF_START_ID = 8 # mapped starting stuff id when the mapped id for thing classes
                   # starts from 0.

@DATASETS.register_module
class SemanticKITTIDataset(Dataset):
    """Custom dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 config='',
                 split='',
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=False):

        self.ann_file = ann_file
        self.data_root = data_root
        self.config = config
        self.split = split
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt

        self.cfg = yaml.safe_load(open(self.config, 'r'))

        seqs = self.cfg["split"][self.split]
        seq_names = []
        for seq in seqs:
            seq = '{0:02d}'.format(int(seq))
            velodyne_paths = os.path.join(self.ann_file, seq, "velodyne")
            seq_label_names = sorted([os.path.join(velodyne_paths, fn) for fn in os.listdir(velodyne_paths) if fn.endswith(".bin")])
            seq_names.extend(seq_label_names)

        self.vel_seq_infos = seq_names 

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        self.class_remap = self.cfg["learning_map"]
        self.class_inv_remap = self.cfg["learning_map_inv"]
        self.class_ignore = self.cfg["learning_ignore"]
        self.nr_classes = len(self.class_inv_remap)
        self.class_strings = self.cfg["labels"]

        # make lookup table for mapping
        # class
        maxkey = max(self.class_remap.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        self.class_lut = np.zeros((maxkey + 100), dtype=np.int32)
        self.class_lut[list(self.class_remap.keys())] = list(self.class_remap.values())

        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.vel_seq_infos)

    def load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.ones(len(self), dtype=np.uint8)

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None or (len(data['gt_labels'].data) == 0):
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        label_info = self.vel_seq_infos[idx].replace('velodyne', 'labels').replace('.bin', '.label')
        img_info = {
                    'filename': self.vel_seq_infos[idx],
                    'width': WIDTH,
                    'height': HEIGHT,
                    'ann': label_info
                    }
        ann_info =  label_info
        results = dict(img_info=img_info, ann_info=ann_info, class_lut=self.class_lut, stuff_id=STUFF_START_ID)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        label_info = self.vel_seq_infos[idx].replace('velodyne', 'labels').replace('.bin', '.label')
        img_info = {
                    'filename': self.vel_seq_infos[idx],
                    'width': WIDTH,
                    'height': HEIGHT,
                    'ann': label_info
                    }
        results = dict(img_info=img_info, class_lut=self.class_lut, stuff_id=STUFF_START_ID)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, **kwargs):
        pass

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 min_points=50):

        ignore_class = [cl for cl, ignored in self.class_ignore.items() if ignored] 
        class_evaluator = PanopticEval(self.nr_classes, None, ignore_class, min_points=min_points)
        test_sequences = self.cfg['split'][self.split]
        label_names = []
        for sequence in test_sequences:
            sequence = '{0:02d}'.format(int(sequence))
            label_paths = os.path.join(self.ann_file, sequence, "labels")
            seq_label_names = sorted([os.path.join(label_paths, fn) for fn in os.listdir(label_paths) if fn.endswith(".label")])
            label_names.extend(seq_label_names)
        
        # get predictions paths
        pred_names = []
        for sequence in test_sequences:
            sequence = '{0:02d}'.format(int(sequence))
            pred_paths = os.path.join('tmpDir', sequence, "predictions")
            # populate the label names
            seq_pred_names = sorted([os.path.join(pred_paths, fn) for fn in os.listdir(pred_paths) if fn.endswith(".label")])
            pred_names.extend(seq_pred_names)
        # check that I have the same number of files
        assert (len(label_names) == len(pred_names))

        print("Evaluating sequences: ", end="", flush=True)
        # open each file, get the tensor, and make the iou comparison

        complete = len(label_names)
        count = 0
        percent = 10
        for label_file, pred_file in zip(label_names, pred_names):
            count = count + 1
            if 100 * count / complete > percent:
                print("{}% ".format(percent), end="", flush=True)
                percent = percent + 10

            label = np.fromfile(label_file, dtype=np.uint32)

            u_label_sem_class = self.class_lut[label & 0xFFFF]  # remap to xentropy format
            u_label_inst = label # unique instance ids.

            label = np.fromfile(pred_file, dtype=np.uint32)

            u_pred_sem_class = label & 0xFFFF  # remap to xentropy format
            u_pred_sem_class += 1
            u_pred_sem_class[u_pred_sem_class == 256] = 0 
            u_pred_inst = label # unique instance ids.

            class_evaluator.addBatch(u_pred_sem_class, u_pred_inst, u_label_sem_class, u_label_inst)

        print("100%")

        # when I am done, print the evaluation
        class_PQ, class_SQ, class_RQ, class_all_PQ, class_all_SQ, class_all_RQ = class_evaluator.getPQ()
        class_IoU, class_all_IoU = class_evaluator.getSemIoU()

        # now make a nice dictionary
        output_dict = {}

        # make python variables
        class_PQ = class_PQ.item()
        class_SQ = class_SQ.item()
        class_RQ = class_RQ.item()
        class_all_PQ = class_all_PQ.flatten().tolist()
        class_all_SQ = class_all_SQ.flatten().tolist()
        class_all_RQ = class_all_RQ.flatten().tolist()
        class_IoU = class_IoU.item()
        class_all_IoU = class_all_IoU.flatten().tolist()

        things = ['car', 'truck', 'bicycle', 'motorcycle', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist']
        stuff = [
            'road', 'sidewalk', 'parking', 'other-ground', 'building', 'vegetation', 'trunk', 'terrain', 'fence', 'pole',
            'traffic-sign'
        ]
        all_classes = things + stuff

        # class

        output_dict["all"] = {}
        output_dict["all"]["PQ"] = class_PQ
        output_dict["all"]["SQ"] = class_SQ
        output_dict["all"]["RQ"] = class_RQ
        output_dict["all"]["IoU"] = class_IoU

        classwise_tables = {}

        for idx, (pq, rq, sq, iou) in enumerate(zip(class_all_PQ, class_all_RQ, class_all_SQ, class_all_IoU)):
            class_str = self.class_strings[self.class_inv_remap[idx]]
            output_dict[class_str] = {}
            output_dict[class_str]["PQ"] = pq
            output_dict[class_str]["SQ"] = sq
            output_dict[class_str]["RQ"] = rq
            output_dict[class_str]["IoU"] = iou

        msg = "{:14s}| {:>5s}  {:>5s}  {:>5s}".format("Category", "PQ", "SQ", "RQ")
        print_log(msg, logger=logger)

        PQ_all = np.mean([float(output_dict[c]["PQ"]) for c in all_classes])
        PQ_dagger = np.mean([float(output_dict[c]["PQ"]) for c in things] + [float(output_dict[c]["IoU"]) for c in stuff])
        RQ_all = np.mean([float(output_dict[c]["RQ"]) for c in all_classes])
        SQ_all = np.mean([float(output_dict[c]["SQ"]) for c in all_classes])
        self.log_results('All', PQ_all, RQ_all, SQ_all, logger)

        PQ_things = np.mean([float(output_dict[c]["PQ"]) for c in things])
        RQ_things = np.mean([float(output_dict[c]["RQ"]) for c in things])
        SQ_things = np.mean([float(output_dict[c]["SQ"]) for c in things])
        self.log_results('Things', PQ_things, RQ_things, SQ_things, logger)

        PQ_stuff = np.mean([float(output_dict[c]["PQ"]) for c in stuff])
        RQ_stuff = np.mean([float(output_dict[c]["RQ"]) for c in stuff])
        SQ_stuff = np.mean([float(output_dict[c]["SQ"]) for c in stuff])
        self.log_results('Stuff', PQ_stuff, RQ_stuff, SQ_stuff, logger)

        mIoU = output_dict["all"]["IoU"]

        codalab_output = {}
        codalab_output["pq_mean"] = float(PQ_all)
        codalab_output["pq_dagger"] = float(PQ_dagger)
        codalab_output["sq_mean"] = float(SQ_all)
        codalab_output["rq_mean"] = float(RQ_all)
        codalab_output["iou_mean"] = float(mIoU)
        codalab_output["pq_stuff"] = float(PQ_stuff)
        codalab_output["rq_stuff"] = float(RQ_stuff)
        codalab_output["sq_stuff"] = float(SQ_stuff)
        codalab_output["pq_things"] = float(PQ_things)
        codalab_output["rq_things"] = float(RQ_things)
        codalab_output["sq_things"] = float(SQ_things)
        table = []
        for cl in all_classes:
            entry = output_dict[cl]
            table.append({
                "class": cl,
                "pq": "{:.3}".format(entry["PQ"]),
                "sq": "{:.3}".format(entry["SQ"]),
                "rq": "{:.3}".format(entry["RQ"]),
                "iou": "{:.3}".format(entry["IoU"])
            })
        print('Generating output files in tmpDir')
        # save to yaml
        output_filename = os.path.join('tmpDir', 'scores.txt')
        with open(output_filename, 'w') as outfile:
            yaml.dump(codalab_output, outfile, default_flow_style=False)

    def log_results(self, metric, pq, rq, sq, logger):
        msg = "{:14s}| {:5.1f}  {:5.1f}  {:5.1f}".format(
                    metric,
                    100 * pq,
                    100 * rq,
                    100 * sq,
                )
        print_log(msg, logger=logger)

