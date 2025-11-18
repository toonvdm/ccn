from dommel.datasets.dataset_factory import dataset_factory

from dommel.datasets.dataset import (
    Dataset,
    SequenceDataset,
    ConcatDataset
)

from dommel.datasets.pytorch_dataset import PytorchDataset
from dommel.datasets.h5_dataset import H5Dataset

from dommel.datasets.memory_pool import MemoryPool
from dommel.datasets.file_pool import FilePool

from dommel.datasets.kitti import KITTIDataset
from dommel.datasets.euroc import EuRoCDataset
from dommel.datasets.tartanair import TartanAirDataset

__all__ = [
    "dataset_factory",
    "Dataset",
    "SequenceDataset",
    "ConcatDataset",
    "PytorchDataset",
    "H5Dataset",
    "MemoryPool",
    "FilePool",
    "KITTIDataset",
    "EuRoCDataset",
    "TartanAirDataset",
]

try:
    import minerl  # noqa: F401
    from dommel.datasets.minerl_dataset import MineRLDataset  # noqa: F401
    __all__.append("MineRLDataset")
except ImportError:
    pass
