import torch
import h5py as h5

from pathlib import Path


from dommel.datastructs import TensorDict
from dommel.datasets.dataset import retry, Dataset


class H5Dataset(Dataset):
    def __init__(
        self, directory, interval=None, device="cpu", transform=None, **kwargs
    ):
        """
        :param directory: directory that contains h5 files
        :param interval: interval to use from files, default [0, 1]
        :param device: device to load dataset on (file, cpu or cuda:i)
        :param transform: optional transform for the data
        """
        Dataset.__init__(self, transform, **kwargs)

        self._files = sorted(Path(directory).glob("*.h5"))
        if interval:
            begin = int(interval[0] * len(self._files))
            end = int(interval[1] * len(self._files))
            self._files = self._files[begin:end]

        self._buffer = None
        self._device = "cpu"

        if device != "file":
            self._device = device
            self._load(self._device)
            self._get_item = self._read_buffer
        else:
            self._get_item = self._read_file

    def __len__(self):
        return len(self._files)

    def _load(self, device):
        self._buffer = []
        for p in self._files:
            self._buffer.append(self._read(p, device))

    def _read_buffer(self, index):
        return self._buffer[index].to(self._device)

    def _read_file(self, index):
        path = self._files[index]
        return self._read(path, self._device)

    @retry
    def _read(self, path, device):
        with h5.File(path, "r") as f:
            d = {}
            for key in f.keys():
                # Check whether the key contains a tensor
                if len(f[key].shape) > 0:
                    d[key] = torch.as_tensor(f[key][:], device=device)
            return TensorDict(d)
