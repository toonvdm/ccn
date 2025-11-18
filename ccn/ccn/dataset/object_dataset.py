import time
import numpy as np
import torch
import h5py as h5
from pathlib import Path

from pytorch3d.transforms import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    matrix_to_rotation_6d,
)

from dommel.datasets.dataset import retry, Dataset
from dommel.datastructs import TensorDict

from ccn.geometry import relative_transform

import logging

logger = logging.getLogger(__name__)


def matrix_to_6d(x):
    return matrix_to_rotation_6d(x[:3, :3])


def matrix_to_quat(x):
    return matrix_to_quaternion(x[:3, :3])


def cartesian_to_spherical(x):
    return torch.tensor(
        [
            torch.atan2(x[1], x[0]),
            torch.atan2(torch.sqrt(x[0] ** 2 + x[1] ** 2), x[2]),
        ],
        device=x.device,
        dtype=x.dtype,
    ) / (2 * np.pi)


class ObjectDataset(Dataset):
    def __init__(
        self,
        directory,
        object_file,
        considered_object_files,
        interval=None,
        anchor_interval=None,
        device="cuda:0",
        load_device="cuda:0",
        dtype=torch.float32,
        transform=None,
        seed=0,
        ds_length=10000,
        load_depth=True,
        orientation_mode="quat",
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
        :param orientation_mode: options are "quat", "6do", "spherical", "spherical-rot"]
        :param kwargs:
        """
        Dataset.__init__(self, transform, **kwargs)

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
        self._alpha_key = "alpha"

        self._load_keys = [self._im_key, self._p_key]

        if load_depth:
            self._load_keys += [self._d_key]

        # Variables for data
        self._device = device
        self._orientation_mode = orientation_mode

        if self._orientation_mode == "6do":
            self._translation_to_representation = lambda x: x
            self._matrix_to_representation = matrix_to_6d
        elif self._orientation_mode == "quat":
            self._translation_to_representation = lambda x: x
            self._matrix_to_representation = matrix_to_quat
        elif self._orientation_mode == "spherical":
            self._translation_to_representation = cartesian_to_spherical
            self._matrix_to_representation = lambda x: torch.zeros(
                (0), dtype=x.dtype, device=x.device
            )
        elif self._orientation_mode == "spherical-rot":
            self._load_keys += [self._alpha_key]
            self._translation_to_representation = cartesian_to_spherical
            self._matrix_to_representation = lambda x: torch.tensor(
                [x], dtype=torch.float32, device=x.device
            )
        else:
            raise NotImplementedError

        # cast to correct dtype
        self._dtype = dtype
        if isinstance(self._dtype, str):
            self._dtype = eval(f"torch.{self._dtype}")

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

        self._positive_target = torch.tensor(
            [1.0], dtype=self._dtype, device=load_device
        )
        self._negative_target = torch.tensor(
            [0.0], dtype=self._dtype, device=load_device
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
        mask_1 = self._buffer[self._mask_key][idx_1]
        image_2 = self._buffer[self._im_key][idx_2]
        mask_2 = self._buffer[self._mask_key][idx_2]

        image_a = self._anchor_buffer[
            np.random.randint(len(self._anchor_buffer))
        ]

        # Get a negative anchor
        rt = relative_transform(
            self._buffer[self._p_key][idx_1].unsqueeze(0),
            self._buffer[self._p_key][idx_2].unsqueeze(0),
        )[0]

        # x y z format
        translation = self._translation_to_representation(rt[:3, -1])
        # w x y z format
        if self._orientation_mode == "spherical-rot":
            rt = (
                self._buffer[self._alpha_key][idx_2]
                - self._buffer[self._alpha_key][idx_1]
            ) / (2 * np.pi)
        orientation = self._matrix_to_representation(rt)

        return TensorDict(
            {
                "image_in": image_1,
                "image_out": image_2,
                "mask_in": mask_1,
                "mask_out": mask_2,
                "negative_anchor": image_a,
                "negative_mask": self._empty_image[0],
                "translation": translation,
                "orientation": orientation,
                "positive_target": self._positive_target,
                "negative_target": self._negative_target,
                # Not for training, but useful for evaluation
                # i.e. render imagination against ground truth poses etc.
                "p_in": self._buffer[self._p_key][idx_1],
                "p_out": self._buffer[self._p_key][idx_2],
            }
        ).to(self._device)
