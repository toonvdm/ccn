from dommel.datasets import Dataset
from dommel.datastructs import TensorDict


class PytorchDataset(Dataset):
    """
    Wrap pytorch datasets in a dict dataset
    """

    def __init__(self, dataset, keys=None, transform=None, **kwargs):
        self._keys = keys
        if not self._keys:
            self._keys = ["input", "label"]
        self._ds = dataset
        Dataset.__init__(self, transform, **kwargs)

    def __len__(self):
        return len(self._ds)

    def _get_item(self, idx):
        sample = self._ds[idx]
        return TensorDict({k: sample[i] for i, k in enumerate(self._keys)})
