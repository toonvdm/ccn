import numpy as np
import torch
import h5py as h5
from pathlib import Path

from dommel.datasets.dataset import retry, Dataset
from dommel.datastructs import TensorDict

from ccn.geometry import relative_transform
from ccn.action import Action

import logging

logger = logging.getLogger(__name__)


class SphericalObjectDataset(Dataset):
    def __init__(
        self,
        directory,
        object_file,
        considered_object_files,
        interval=None,
        anchor_interval=None,
        device="cuda:0",
        load_device="cuda:0",
        transform=None,
        seed=0,
        ds_length=10000,
        **kwargs,
    ):
        """
        :param directory: location to find the data
        :param object_file: the file for which the dataset is created
        :param considered_object_files: a list of all considered objects in this set
            - to be used for the anchored files, the object file is filtered out
        :param interval: interval to sample data from (train/test splits)
        :param anchor_interval: interval to sample data from for anchors (train/test splits)
        :param device: device to store data
        :param dtype: datatype to use (legacy, not used)
        :param transform: transform to apply to a dataset
        :param seed: seed to set (necessary for valid test/train splits)
        :param ds_length: length of the dataset
        :param load_depth: whether to load depth
        :param kwargs:
        """
        Dataset.__init__(self, transform, **kwargs)

        self._dtype = torch.float32

        # indices to shuffle data before splitting!
        np.random.seed(seed)
        # TODO: automatically determine the size of the dataset
        self._indices = np.arange(0, ds_length)
        np.random.shuffle(self._indices)

        if interval is None:
            interval = [0, 1]

        # Keys for data access
        self._im_key = "rgb"
        self._d_key = "depth"
        self._k_key = "intrinsic_matrix"
        self._p_key = "pose_matrix"
        self._mask_key = "mask"

        self._load_keys = [self._im_key, self._p_key]

        self._device = device

        self._buffer = self._read(
            str(Path(directory) / object_file), load_device, interval
        )

        self._buffer[self._mask_key] = [
            (i.sum(dim=0) > 0).to(dtype=torch.float32, device=i.device)
            for i in self._buffer[self._im_key]
        ]

        # load anchors
        if anchor_interval is None:
            anchor_interval = [0, 0.25]

        self._anchor_buffer = None
        anchor_files = [f for f in considered_object_files if f != object_file]
        for af in anchor_files:
            anchors = self._read_key(
                str(Path(directory) / af),
                load_device,
                anchor_interval,
                self._im_key,
            )
            if self._anchor_buffer is None:
                self._anchor_buffer = anchors
            else:
                self._anchor_buffer = torch.cat(
                    [self._anchor_buffer, anchors], dim=0
                )

        # add 5% black images
        black_ims = torch.zeros(
            int(0.05 * len(self._anchor_buffer)),
            *self._anchor_buffer.shape[1:],
        ).to(dtype=self._dtype, device=load_device)

        self._empty_image = torch.zeros_like(self._buffer[self._im_key][0]).to(
            dtype=self._dtype, device=load_device
        )
        self._anchor_buffer = torch.cat(
            [self._anchor_buffer, black_ims], dim=0
        )

    @retry
    def _read_key(self, path, device, interval, key):
        with h5.File(path, "r") as f:
            begin = int(interval[0] * len(f[key]))
            end = int(interval[1] * len(f[key]))
            x = torch.as_tensor(
                f[key][:][self._indices[begin:end]],
                device=device,
                dtype=self._dtype,
            )
            # immediately convert to torch format
            if key == self._im_key:
                x = x.permute((0, 3, 1, 2))
        return x

    @retry
    def _read(self, path, device, interval):
        d = {}
        with h5.File(path, "r") as f:
            begin = int(interval[0] * len(f[self._im_key]))
            end = int(interval[1] * len(f[self._im_key]))

            for key in f.keys():
                # Check whether the key contains a tensor
                if len(f[key].shape) > 0 and key in self._load_keys:
                    # b = time.time()
                    x = torch.as_tensor(
                        f[key][:][self._indices[begin:end]],
                        device=device,
                        dtype=self._dtype,
                    )
                    if key == self._im_key:
                        x = x.permute((0, 3, 1, 2))
                    d[key] = x
        return TensorDict(d)

    def __len__(self):
        # small hack to allow for batch sizes bigger than len(self._buffer[self._im_key])
        # requires a % operation at _get_item()
        length = len(self._buffer[self._im_key])
        return np.max([length, 500])

    def _get_item(self, index):
        """
        Will take a fixed initial sample, and pair it with a randomly selected one
        from the dataset
        :param index: index to query as input view
        :return: a paired data tensordict
        """
        random_index = np.random.randint(len(self._buffer[self._im_key]))
        index = index % len(self._buffer[self._im_key])
        return self.get_pair(index, random_index)

    def get_pair(self, idx_1, idx_2):
        image_1 = self._buffer[self._im_key][idx_1]
        image_2 = self._buffer[self._im_key][idx_2]

        image_a = self._anchor_buffer[
            np.random.randint(len(self._anchor_buffer))
        ]

        # Only keep the azimuth and elevation for training on this dataset
        # Fixed distance
        action = Action.from_matrix(
            relative_transform(
                self._buffer[self._p_key][idx_1].unsqueeze(0),
                self._buffer[self._p_key][idx_2].unsqueeze(0),
            )
        ).nn_action[0]

        return TensorDict(
            {
                "image_in": image_1,
                "image_out": image_2,
                "negative_anchor": image_a,
                "action": action,
            }
        ).to(self._device)
