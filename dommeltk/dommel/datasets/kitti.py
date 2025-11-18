import os
from pathlib import Path
import numpy as np

import torch
import torch.utils.data as data
from torchvision.transforms import ToTensor

from skimage import io
from datetime import datetime
from random import randint

from dommel.datastructs import TensorDict


class KITTIDataset(data.Dataset):
    def __init__(self, directory, keys, sequence_length, transform=None):
        data.Dataset.__init__(self)
        self.root_path = Path(directory)
        self.keys = keys
        self.to_tensor = ToTensor()
        self.subsequence_length = sequence_length
        self.recs = [x for x in self.root_path.iterdir() if x.is_dir()]
        self.length = len(self.recs)
        self.start_index = None
        if transform is None:
            self.transforms = {key: [] for key in keys}
        else:
            self.transforms = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        path = self.recs[index]
        sequences = {}
        for key in self.keys:
            if "image" in key:
                sequence = self.load_img_recording(path, key)
            elif "velodyne" in key:
                sequence = self.load_velodyne(path, key)
            elif "oxts" in key:
                sequence = self.load_oxts(path, key)
            sequences.update(sequence)
        self.start_index = None
        return TensorDict(sequences)

    def _prepare(self, path, key):
        data = path / key / "data"
        timestamps = path / key / "timestamps.txt"
        subsequence_length = self.subsequence_length
        files = sorted(os.listdir(data))
        if self.start_index is None:
            self.start_index = randint(0, len(files) - subsequence_length)
        files = files[self.start_index: self.start_index + subsequence_length]

        with open(timestamps, "r") as times:
            timesteps = times.readlines()
        t_str = "%Y-%m-%d %H:%M:%S.%f"
        timesteps = [
            datetime.strptime(t[:-4], t_str).timestamp() for t in timesteps
        ]
        timesteps = torch.tensor(
            timesteps[self.start_index: self.start_index + subsequence_length]
        )

        return files, timesteps

    def load_img_recording(self, path, key):
        files, timesteps = self._prepare(path, key)
        imgs = None

        for file in files:
            img_path = data / file
            img = io.imread(img_path)
            img = self.to_tensor(img)
            if imgs is None:
                imgs = img
                imgs = imgs.unsqueeze(0)
            else:
                imgs = torch.cat((imgs, img.unsqueeze(0)), dim=0)

        for trans in self.transforms[key]:
            imgs = trans(imgs)

        return {key: imgs, f"{key}_time": timesteps}

    def load_velodyne(self, path, key):
        files, timesteps = self._prepare(path, key)
        tensors = None

        for file in files:
            velo_path = data / file
            arr = np.fromfile(velo_path)
            tens = torch.tensor(arr)
            if tensors is None:
                tensors = tens.unsqueeze(0)
            else:
                tensors = torch.cat((tensors, tens.unsqueeze(0)), dim=0)

        for trans in self.transforms[key]:
            tensors = trans(tensors)

        return {key: tensors, f"{key}_time": timesteps}

    def load_oxts(self, path, key):
        files, timesteps = self._prepare(path, key)
        tensors = None

        for file in files:
            oxts_path = data / file
            with open(oxts_path, "r") as oxts_file:
                oxts_data = oxts_file.readline()
            # throw away the gps config info
            oxts_data = oxts_data.split()[:-5]
            oxts_data = [float(x) for x in oxts_data]
            oxts_tensor = torch.tensor(oxts_data)
            if tensors is None:
                tensors = oxts_tensor.unsqueeze(0)
            else:
                tensors = torch.cat((tensors, oxts_tensor.unsqueeze(0)), dim=0)

        for trans in self.transforms[key]:
            tensors = trans(tensors)

        return {key: tensors, f"{key}_time": timesteps}
