import torch
from pathlib import Path
from numpy import genfromtxt
from skimage import io

from dommel.datastructs import TensorDict, stack
from dommel.datasets import SequenceDataset


class EuRoCDataset(SequenceDataset):
    def __init__(self, directory, sequence_length,
                 sequence_stride=1, transform=None,
                 keys=None, num_workers=0, **kwargs):
        self._keys = keys
        if not self._keys:
            self._keys = ["camera", "imu", "position", "timestamp"]
        self._root_path = Path(directory)

        imu_path = self._root_path / "mav0" / "imu0" / "data.csv"
        self._imu = genfromtxt(imu_path, delimiter=',')

        camera_path = self._root_path / "mav0" / "cam0" / "data.csv"
        self._camera = genfromtxt(camera_path, delimiter=',')

        position_path = self._root_path / "mav0" / "leica0" / "data.csv"
        if not position_path.is_file():
            position_path = self._root_path / "mav0" / \
                "state_groundtruth_estimate0" / "data.csv"
        self._position = genfromtxt(position_path, delimiter=',')

        start_imu = self._imu[0, 0]
        start_camera = self._camera[0, 0]
        start_position = self._position[0, 0]
        start = max(start_imu, start_camera, start_position)

        self._camera_offset = self._get_offset(self._camera, start)
        start = self._camera[self._camera_offset, 0]
        self._imu_offset = self._get_offset(self._imu, start)
        self._camera_offset = self._get_offset(self._camera, start)
        self._position_offset = self._get_offset(self._position, start)

        end_imu = self._imu[-1, 0]
        end_camera = self._camera[-1, 0]
        end_position = self._position[-1, 0]

        end = min(end_imu, end_camera, end_position)
        self._camera_end = self._camera.shape[0] - 1
        while self._camera[self._camera_end, 0] > end:
            self._camera_end -= 1

        # camera has lowest sampling rate, detect stride for imu and position
        self._imu_stride = self._get_offset(
            self._imu,
            self._camera[self._camera_offset + 1, 0]) - self._imu_offset
        self._position_stride = self._get_offset(
            self._position,
            self._camera[self._camera_offset + 1, 0]) - self._position_offset

        SequenceDataset.__init__(self, sequence_length, sequence_stride,
                                 False, transform, num_workers)

    def _load_sequences(self):
        keys = ["mav0"]  # for now I load sequence per sequence
        lengths = [self._camera_end - self._camera_offset]
        return keys, lengths

    def _load_sequence(self, key, indices):
        sequence = []
        for i in indices:
            d = TensorDict({})

            camera_idx = self._camera_offset + i
            timestamp = self._camera[camera_idx, 0]
            camera = self._camera[camera_idx, 0]

            if "imu" in self._keys:
                imu_idx = self._imu_offset + i * self._imu_stride
                imu = self._imu[imu_idx, 1:]
                d["imu"] = torch.as_tensor(imu)

            if "position" in self._keys:
                # position rate is not always constant, so we fetch the
                # closest index every time
                position_idx = self._get_offset(self._position, timestamp)
                position = self._position[position_idx, 1:]
                d["position"] = torch.as_tensor(position)

            if "camera" in self._keys:
                img_path = self._root_path / "mav0" / \
                    "cam0" / "data" / f"{camera:.0f}.png"
                img = io.imread(img_path)
                d["camera"] = torch.as_tensor(img).unsqueeze(-1)

            if "timestamp" in self._keys:
                d["timestamp"] = torch.as_tensor([timestamp])

            sequence.append(d.unsqueeze(0))
        return stack(*sequence)

    def _get_offset(self, array, start):
        for i in range(array.shape[0]):
            if array[i, 0] >= start:
                if i == 0:
                    return i
                elif array[i, 0] - start < start - array[i - 1, 0]:
                    return i
                else:
                    return i - 1
        return None
