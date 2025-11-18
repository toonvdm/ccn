import collections
from pathlib import Path

import torch
import gym
import minerl

from dommel.datasets.dataset import SequenceDataset, retry
from dommel.datastructs import TensorDict

import logging

logger = logging.getLogger(__name__)


class MineRLDataset(SequenceDataset):
    """MineRL dataset downloads the given dataset and returns it
    in Tensordict shape.
    """

    def __init__(
        self,
        dataset="MineRLNavigate-v0",
        directory=None,
        transform=None,
        keys=None,
        num_workers=0,
        compression=9,
        interval=None,
        sequence_length=-1,
        sequence_stride=1,
        shuffle=False,
        cutoff=True,
        **kwargs,
    ):
        if not directory:
            directory = "/tmp"
        self._dir = Path(directory)
        self._keys = keys
        self._compression = compression
        if not self._dir.exists():
            self._dir.mkdir(parents=True)

        self._dataset = dataset
        minerl.data.download(self._dir, environment=self._dataset)
        self._data_pipeline = minerl.data.make(self._dataset)
        self._env = gym.make(self._dataset)
        self._parent_key = {
            "obs": 0,
            "action": 1,
            "reward": 2,
            "next_obs": 3,
            "done": 4,
        }
        self._key_translate = None

        # list all trajectories in directory
        self._all_trajectories = self._data_pipeline.get_trajectory_names()

        # Train val test split
        if interval is None:
            interval = [0, 1.0]

        ind_min = int(interval[0] * len(self._all_trajectories))
        ind_max = int(interval[1] * len(self._all_trajectories))
        self._all_trajectories = self._all_trajectories[ind_min:ind_max]

        SequenceDataset.__init__(
            self,
            sequence_length,
            sequence_stride,
            shuffle,
            cutoff,
            transform,
            num_workers,
        )

    def _load_sequences(self):
        lengths = [self._get_length(file) for file in self._all_trajectories]
        return self._all_trajectories, lengths

    def _load_sequence(self, key, indices):
        return self._read(key, indices)

    @retry
    def _read(self, path, indices):
        data = []
        data_loader = self._data_pipeline.load_data(path)
        for data_tuple in data_loader:
            dtuple, ktranslate = self._data_flatten(data_tuple)
            data.append(dtuple)

        if not self._key_translate:
            self._key_translate = ktranslate
        if not self._keys:
            self._keys = list(self._key_translate)

        d = {}
        for key in self._keys:
            if self._sequence_length > 0:
                value = [data[i][self._key_translate[key]] for i in indices]
            else:
                if key not in self._key_translate.keys():
                    raise Exception("Key " + key + " not in " + str(path))
                value = [y[self._key_translate[key]] for y in data]

            # if value.dtype == "uint16":
            #     value = value.astype(np.int32)
            # print(key, value)
            d[key] = torch.as_tensor(value)
        return TensorDict(d)

    @retry
    def _get_length(self, file):
        data_loader = self._data_pipeline.load_data(file)
        traj_len = 0
        for _ in data_loader:
            traj_len += 1
        return traj_len

    def _data_flatten(self, data_tuple):
        new_data_tuple = []
        new_key_translate = {}
        counter = 0

        for i, data in enumerate(data_tuple):
            key = list(self._parent_key)[i]
            if key == "action":
                act = self._env.action_space.flat_map(data)
                new_data_tuple.append(act)
                new_key_translate[key] = counter
                counter += 1
            elif isinstance(data, dict):
                new_data = self._flatten(data, parent_key=key)
                for k, v in new_data.items():
                    new_data_tuple.append(v)
                    new_key_translate[k] = counter
                    counter += 1
            else:
                new_data_tuple.append(data)
                new_key_translate[key] = counter
                counter += 1

        return tuple(new_data_tuple), new_key_translate

    def _flatten(self, d, parent_key="", sep="_"):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if v is not None and isinstance(v, collections.MutableMapping):
                items.extend(self._flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
