import torch
import os
import os.path
import sys
import numpy as np
import pandas
from PIL import Image
from collections import OrderedDict
import json

from ltr.data.image_loader import default_image_loader
from .base_dataset import BaseDataset
from ltr.admin.environment import env_settings


def mask2box(mask):
    """
    Calculate a bounding box according to the mask

    :param mask: 2 dimensional np array, elements' value indicate different objects, eg. 0:background 1:obj1 2:obj2
    :return: bbox in the form of [x1,y1,w,h]
    """
    mask_idx = np.where(mask == 1)
    return torch.Tensor([mask_idx[1].min(), mask_idx[0].min(),
                        mask_idx[1].max() - mask_idx[1].min() + 1,
                        mask_idx[0].max() - mask_idx[0].min() + 1])


class DAVIS(BaseDataset):
    """ DAVIS dataset.

    Publication:
        DAVIS: Densely Annotated Video Segmentation.
        The 2017 DAVIS Challenge on Video Object Segmentation
        J. Pont-Tuset, F. Perazzi, S. Caelles, P. Arbel¨¢ez, A. Sorkine-Hornung, and L. Van Gool
        arXiv:1704.00675, 2017

    Download the dataset in https://davischallenge.org/davis2017/code.html.
    """
    def __init__(self, root=None, image_loader=default_image_loader, mode='train', train_ratio=1):
        """
        args:
            root        - The path to the DAVIS folder, containing the training sets.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            mode - The current dataset mode, train or val.
            train_ratio - The ratio of train data.
        """
        assert ((train_ratio >= 0) and (train_ratio <= 1)), 'train ratio of dataset must be in the range of 0-1'

        root = env_settings().davis_dir if root is None else root
        super().__init__(root, image_loader)
        self.anno_path = os.path.join(root, 'Annotations/480p')
        self.seq_path = os.path.join(root, 'JPEGImages/480p')

        self.sequence_list = self._list_sequences(root)
        train_num = int(len(self.sequence_list) * train_ratio)
        if mode == 'train':
            self.sequence_list = self.sequence_list[:train_num]
        elif mode == 'val':
            self.sequence_list = self.sequence_list[train_num:]

    def get_name(self):
        return 'davis'

    def get_sequence_len(self, seq_id):
        return self.sequence_list[seq_id]['length']

    def get_sequence_info(self, seq_id):
        return torch.Tensor(self.sequence_list[seq_id]['target_visible'])

    def _read_anno(self, seq_id, frame_id):
        mask_path = self.sequence_list[seq_id]['masks'][frame_id]
        mask = np.array(Image.open(mask_path))
        object_id = self.sequence_list[seq_id]['object_id']
        mask[mask != object_id] = 0
        mask[mask == object_id] = 1
        return mask

    def _get_frame(self, seq_id, frame_id):
        return self.image_loader(self.sequence_list[seq_id]['images'][frame_id])

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_name = self.sequence_list[seq_id]['name']
        frame_list = [self._get_frame(seq_id, f) for f in frame_ids]
        mask_frames = [self._read_anno(seq_id, f) for f in frame_ids]
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

        sequences_names = []
        with open(os.path.join(self.root, 'ImageSets/2017/train.txt'), 'r') as f:
            tmp = f.readlines()
            sequences_names.extend([x.strip() for x in tmp])
        with open(os.path.join(self.root, 'ImageSets/2017/val.txt'), 'r') as f:
            tmp = f.readlines()
            sequences_names.extend([x.strip() for x in tmp])

        sequence_list = []
        for seq_id, seq_name in enumerate(sequences_names):     # sequence
            files = os.listdir(os.path.join(self.seq_path, seq_name))

            images = []
            masks = []
            annos = []
            target_visible = []
            image_size = None

            for frame_id in range(len(files)):                  # frame
                image_name = "{:0>5d}.jpg".format(frame_id)
                image_path = os.path.join(self.seq_path, seq_name, image_name)
                mask_name = "{:0>5d}.png".format(frame_id)
                mask_path = os.path.join(self.anno_path, seq_name, mask_name)
                mask = np.array(Image.open(mask_path))
                if not image_size:
                    image_size = mask.shape
                if mask.max() == 0:
                    continue
                if mask.max() > len(images):
                    for _ in range(mask.max() - len(images)):
                        images.append([])
                        masks.append([])
                        annos.append([])
                        target_visible.append([])

                for idx in range(mask.max()):
                    object_id = idx + 1
                    images[idx].append(image_path)
                    masks[idx].append(mask_path)

                    mask_idx = np.where(mask == object_id)
                    if mask_idx[0].size == 0:
                        target_visible[idx].append(False)
                        annos[idx].append(None)
                    else:
                        target_visible[idx].append(True)
                        annos[idx].append([int(mask_idx[1].min()), int(mask_idx[0].min()),
                                          int(mask_idx[1].max() - mask_idx[1].min() + 1),
                                          int(mask_idx[0].max() - mask_idx[0].min() + 1)])

            for idx in range(len(images)):
                if not images[idx]:
                    continue

                new_sequence = {'name': seq_name, 'image_size': image_size, 'object_id': idx + 1,
                                'length': len(images[idx]), 'target_visible': target_visible[idx],
                                'images': images[idx], 'masks': masks[idx], 'annos': annos[idx]}
                sequence_list.append(new_sequence)

        with open(os.path.join(root, 'meta_pytracking.json'), 'w') as f:
            json.dump(sequence_list, f)

        return sequence_list


