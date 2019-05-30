import torch
import os
import os.path
import numpy as np
from PIL import Image
from collections import OrderedDict
import json

from ltr.data.image_loader import default_image_loader
from .base_dataset import BaseDataset
from ltr.admin.environment import env_settings


class VOT2016(BaseDataset):
    """ VOT2016 dataset.

    Publication:
        VOT2016: Densely Annotated Video Segmentation.
        The 2017 DAVIS Challenge on Video Object Segmentation
        J. Pont-Tuset, F. Perazzi, S. Caelles, P. Arbel¨¢ez, A. Sorkine-Hornung, and L. Van Gool
        arXiv:1704.00675, 2017

    Download the dataset in https://davischallenge.org/davis2017/code.html.
    """
    def __init__(self, root=None, image_loader=default_image_loader):
        """
        args:
            root        - The path to the DAVIS folder, containing the training sets.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            set_ids (None) - List containing the ids of the TrackingNet sets to be used for training. If None, all the
                            sets (0 - 11) will be used.
        """
        root = os.path.join(env_settings().dataset_dir, 'VOT2016') if root is None else root
        super().__init__(root, image_loader)
        self.seq_path = os.path.join(root, 'sequences')
        self.mask_path = os.path.join(root, 'anno_seg')

        self.sequence_list = self._list_sequences(root)

    def get_name(self):
        return 'VOT2016'

    def get_sequence_len(self, seq_id):
        return self.sequence_list[seq_id]['length']

    def get_sequence_info(self, seq_id):
        return torch.Tensor(self.sequence_list[seq_id]['target_visible'])

    def _get_frame(self, seq_id, frame_id):
        return self.image_loader(self.sequence_list[seq_id]['images'][frame_id])

    def _get_mask(self, seq_id, frame_id):
        mask_path = self.sequence_list[seq_id]['masks'][frame_id]
        mask = np.array(Image.open(mask_path))
        mask[mask != 0] = 1
        return mask

    def get_frames(self, seq_id, frame_ids, anno=None):
        frame_list = [self._get_frame(seq_id, f) for f in frame_ids]
        mask_frames = [self._get_mask(seq_id, f) for f in frame_ids]
        anno_frames = [torch.Tensor(self.sequence_list[seq_id]['annos'][f]) for f in frame_ids]

        object_meta = OrderedDict({'object_class': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, mask_frames, anno_frames, object_meta

    def _list_sequences(self, root):
        """ Lists all the videos in the input set_ids. Returns a list of tuples (set_id, video_name)

        args:
            root: Root directory to DAVIS
            set_ids: Sets (0-11) which are to be used

        returns:
            list - list of tuples (set_id, video_name) containing the set_id and video_name for each sequence
        """
        # Return dataset if exist
        if os.path.isfile(os.path.join(root, 'meta_pytracking.json')):
            with open(os.path.join(root, 'meta_pytracking.json'), 'r') as f:
                sequence_list = json.load(f)
                return sequence_list

        # List all sequence names
        with open(os.path.join(self.seq_path, 'list.txt'), 'r') as f:
            sequences_names = f.readlines()
            sequences_names = [x.strip() for x in sequences_names]

        # Construct dataset: sequence_list
        image_ext = 'jpg'
        mask_ext = 'png'
        nz = 8
        start_frame = 1

        sequence_list = []
        for seq_name in sequences_names:

            anno_path = os.path.join(self.seq_path, seq_name, 'groundtruth.txt')
            try:
                annos = np.loadtxt(anno_path, dtype=np.float64)
            except:
                annos = np.loadtxt(anno_path, delimiter=',', dtype=np.float64)

            end_frame = annos.shape[0]

            image_path = os.path.join(self.seq_path, seq_name)
            images = ['{sequence_path}/{frame:0{nz}}.{ext}'.format(
                      sequence_path=image_path, frame=frame_idx, nz=nz, ext=image_ext)
                      for frame_idx in range(start_frame, end_frame + 1)]

            mask_path = os.path.join(self.mask_path, seq_name)
            masks = ['{sequence_path}/{frame:0{nz}}.{ext}'.format(
                     sequence_path=mask_path, frame=frame_idx, nz=nz, ext=mask_ext)
                     for frame_idx in range(start_frame - 1, end_frame)]

            if annos.shape[1] > 4:
                gt_x_all = annos[:, [0, 2, 4, 6]]
                gt_y_all = annos[:, [1, 3, 5, 7]]

                x1 = np.amin(gt_x_all, 1).reshape(-1, 1)
                y1 = np.amin(gt_y_all, 1).reshape(-1, 1)
                x2 = np.amax(gt_x_all, 1).reshape(-1, 1)
                y2 = np.amax(gt_y_all, 1).reshape(-1, 1)

                annos = np.concatenate((x1, y1, x2 - x1, y2 - y1), 1)

            annos = annos.tolist()  # Convert to Python format for json

            # decide if target is visible
            target_visible = []
            image_size = []
            for mask in masks:
                mask_path = os.path.join(self.mask_path, seq_name, mask)
                mask = np.array(Image.open(mask_path))
                mask[mask == 255] = 1
                if mask.max() == 0:
                    target_visible.append(False)
                else:
                    target_visible.append(True)
                image_size = mask.shape
            new_sequence = {'name': seq_name, 'image_size': image_size,
                            'length': end_frame, 'target_visible': target_visible,
                            'images': images, 'masks': masks, 'annos': annos}
            sequence_list.append(new_sequence)

        with open(os.path.join(root, 'meta_pytracking.json'), 'w') as f:
            json.dump(sequence_list, f)

        return sequence_list
