import os
import json
import cv2
import numpy as np
from PIL import Image
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList


def DAVISDataset():
    return DAVISDatasetClass().get_sequence_list()


class DAVISDatasetClass(BaseDataset):
    """DAVIS dataset

    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.davis_path
        self.seq_path = os.path.join(self.base_path, 'JPEGImages/480p')
        self.anno_path = os.path.join(self.base_path, 'Annotations/480p')
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        dataset = self._prepare_groundtruth()
        sequence_list = []
        for seq in dataset:
            sequence_list.append(Sequence(seq['sequence_name'], seq['frames'],
                                          np.array(seq['ground_truth_rect'])))
        return SequenceList(sequence_list)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = os.listdir(self.seq_path)
        return sequence_list

    def _prepare_groundtruth(self):
        dataset = []
        meta_path = os.path.join(self.base_path, 'meta_pytracking_gt.json')
        if os.path.isfile(meta_path):
            with open(meta_path, 'r') as f:
                dataset = json.load(f)
                return dataset

        start_frame = 0
        nz = 5
        ext = 'jpg'
        for seq_name in self.sequence_list:
            anno_files = os.listdir(os.path.join(self.anno_path, seq_name))
            anno_files.sort()
            ground_truth_rect = [self.mask_to_box(os.path.join(self.anno_path, seq_name, x))
                                 for x in anno_files]

            frame_num = len(ground_truth_rect)
            frames = ['{seq_path}/{seq_name}/{frame:0{nz}}.{ext}'.format(seq_path=self.seq_path,
                      seq_name=seq_name, frame=frame+start_frame, nz=nz, ext=ext)
                      for frame in range(frame_num)]

            dataset.append({'sequence_name': seq_name, 'frames': frames,
                            'ground_truth_rect': ground_truth_rect})

        with open(meta_path, 'w') as f:
            json.dump(dataset, f)

        return dataset

    @staticmethod
    def mask_to_box(mask_path, mode='rectangle'):
        """
        Calculate bounding box according to mask, return information of object 1 if
        it have multiple objects in mask.
        :param mask_path: mask file path
        :param mode: rectangle(default) or polygon
        :return: box: rectangle: [x0, y0, w, h].
                      polygon: [[xy0],[xy1],[xy2],[xy3]], counter-clock
                 None: object does not exist in this mask.
        """
        assert os.path.isfile(mask_path), '{} do not exists'.format(mask_path)
        mask = np.array(Image.open(mask_path))

        idx = np.where(mask == 1)                   # (2,n)
        if idx[0].size == 0:
            return [0,0,0,0]

        if mode == 'polygon':
            idx = np.array(idx).transpose(1, 0)     # (y,x)*n
            rect = cv2.minAreaRect(idx)             # tuple((cy,cx), (h,w), angle)
            box = cv2.boxPoints(rect)               # (4,2)
            # box = np.int0(box)
            box = np.array([p[-1::-1] for p in box])
        elif mode == 'rectangle':
            box = np.array([idx[1].min(), idx[0].min(),             # (x,y,w,h)
                            idx[1].max() - idx[1].min(),
                            idx[0].max() - idx[0].min()])

        box = box.tolist()
        return box
