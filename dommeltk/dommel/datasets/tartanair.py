import torch
import numpy as np
from numpy import genfromtxt
from skimage import io
from pathlib import Path

from dommel.datastructs import TensorDict, stack
from dommel.datasets import SequenceDataset


class TartanAirDataset(SequenceDataset):
    def __init__(self, directory, sequence_length,
                 sequence_stride=1, transform=None,
                 keys=None, num_workers=0, **kwargs):
        self._keys = keys
        if not self._keys:
            self._keys = ["image_left", "image_right",
                          "depth_left", "depth_right",
                          "seg_left", "seg_right",
                          "pose_left", "pose_right"]
        self._root_path = Path(directory)
        self._sequence_dirs = [
            x for x in self._root_path.iterdir() if x.is_dir()]

        SequenceDataset.__init__(self,
                                 sequence_length=sequence_length,
                                 sequence_stride=sequence_stride,
                                 transform=transform,
                                 num_workers=num_workers)

    def _load_sequences(self):
        lengths = [self._get_length(d) for d in self._sequence_dirs]
        return self._sequence_dirs, lengths

    def _load_sequence(self, key, indices):
        items = []
        for i in indices:
            items.append(self._load_index(key, i).unsqueeze(0))

        stacked = stack(*items)
        return stacked

    def _load_index(self, sequence_dir, index):
        item = TensorDict({})
        for key in self._keys:
            if "pose" in key:
                pose_file = Path(sequence_dir) / (key + ".txt")
                poses = genfromtxt(pose_file, delimiter=' ')
                item[key] = torch.as_tensor(
                    poses[index, [0, 1, 2, 6, 3, 4, 5]])
            elif "image" in key:
                img_path = list((Path(sequence_dir) / key).glob(
                    "{:06d}".format(index) + "*"))[0]
                img = io.imread(img_path)
                item[key] = torch.as_tensor(img)
            else:
                data_path = list((Path(sequence_dir) / key).glob(
                    "{:06d}".format(index) + "*"))[0]
                data = np.load(data_path)
                item[key] = torch.as_tensor(data)
        return item

    def _get_length(self, d):
        image_dir = Path(d) / "image_left"
        images = [x for x in image_dir.iterdir() if x.is_file()]
        return len(images)
