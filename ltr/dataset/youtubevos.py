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


class YouTubeVOS_2018(BaseDataset):
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
            set_ids (None) - List containing the ids of the TrackingNet sets to be used for training. If None, all the
                            sets (0 - 11) will be used.
        """
        assert ((train_ratio >= 0) and (train_ratio <= 1)), 'train ratio of dataset must be in the range of 0-1'

        root = os.path.join(env_settings().dataset_dir, 'YouTubeVOS_2018/train') if root is None else root
        super().__init__(root, image_loader)
        self.seq_path = os.path.join(root, 'JPEGImages')
        self.anno_path = os.path.join(root, 'Annotations')

        self.sequence_list = self._list_sequences(root)
        train_num = int(len(self.sequence_list) * train_ratio)
        if mode == 'train':
            self.sequence_list = self.sequence_list[:train_num]
        elif mode == 'val':
            self.sequence_list = self.sequence_list[train_num:]

    def get_name(self):
        return 'YouTubeVOS_2018'

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

        # Open official meta file and construct our own meta file for pytracking
        with open(os.path.join(root, 'meta.json')) as f:
            sequence_list_dict = json.load(f)

        # Collect image, mask, annotation information sequence by sequence,
        # and consider each object in same image as different sequence
        sequence_list = []
        for key, seq in sequence_list_dict['videos'].items():   # seq name: '003234408d'
            seq_name = key

            for sub_key, sub_seq in seq['objects'].items():     # object_id: 'x'
                object_id = int(sub_key)
                seq_length = len(sub_seq['frames'])

                images = ['{seq_path}/{seq}/{frame}.{ext}'.format(seq_path=self.seq_path, seq=seq_name,
                          frame=frame, ext='jpg') for frame in sub_seq['frames']]
                masks = ['{seq_path}/{seq}/{frame}.{ext}'.format(seq_path=self.anno_path, seq=seq_name,
                         frame=frame, ext='png') for frame in sub_seq['frames']]

                annos = []
                target_visible = []
                image_size = (0, 0)
                for mask in masks:
                    mask = np.array(Image.open(mask))
                    mask_idx = np.where(mask == object_id)
                    if mask_idx[0].size == 0:
                        target_visible.append(False)
                        annos.append(None)
                    else:
                        target_visible.append(True)
                        annos.append([int(mask_idx[1].min()), int(mask_idx[0].min()),
                                      int(mask_idx[1].max() - mask_idx[1].min() + 1),
                                      int(mask_idx[0].max() - mask_idx[0].min() + 1)])

                    image_size = mask.shape

                new_sequence = {'name': seq_name, 'image_size': image_size, 'object_id': object_id,
                                'length': seq_length, 'target_visible': target_visible,
                                'images': images, 'masks': masks, 'annos': annos}
                sequence_list.append(new_sequence)

        with open(os.path.join(root, 'meta_pytracking.json'), 'w') as f:
            json.dump(sequence_list, f)

        return sequence_list


