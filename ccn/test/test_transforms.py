import unittest
import numpy as np
import torch
from torch.testing import assert_close

from ccn.util import read_config
from ccn.util import get_data_path as get_ccn_data_path

from dommel.datasets import dataset_factory

from ccn.util import to_img
from ccn.dataset.transforms import RandomBlackFrame, RandomComposeTransform
from ccn.dataset.transforms import (
    RandomBackgroundTransform,
    TransformSequence,
    RandomAnchorTransform,
    ExtractSpatialTransformCrop,
)
from scene_environments.util import get_data_path

import matplotlib.pyplot as plt

store_path = get_ccn_data_path() / "test/transforms"
store_path.mkdir(exist_ok=True, parents=True)


class TransformsTestCase(unittest.TestCase):
    def setUp(self):

        dp = (
            get_ccn_data_path()
            / "../experiments/scene-ccn/spatial-transformer-ccn/train/"
        )
        self.config = read_config(dp / "local_dataset.yml")
        location = (
            get_data_path()
            / "output/Single-Distance-0.5k-up-fibonacci-sphere/"
        )
        self.config["train_dataset"].update(
            {
                "location": str(location),
                "object_file": "002_master_chef_can_textured.h5",
                "load_device": "cpu",
                "device": "cpu",
            }
        )

    def test_black_frame_tf(self):
        rbf = RandomBlackFrame(p=1)
        self.ds = dataset_factory(
            clean=False, transform=rbf, **self.config["train_dataset"]
        )

        for d in self.ds:
            img = d["image_in"]
            self.assertEqual(img.sum(), 0)

        rbf = RandomBlackFrame(p=0)
        self.ds = dataset_factory(
            clean=False, transform=rbf, **self.config["train_dataset"]
        )
        for d in self.ds:
            img = d["image_in"]
            self.assertEqual(img.sum() > 0, True)

    def test_mask_augmentation_transform(self):
        rbf = RandomBlackFrame(p=0.10)
        rct = RandomComposeTransform(s_range=[0.4 / np.sqrt(2), 4.0])
        rbt = RandomBackgroundTransform("image_composed")
        rat = RandomAnchorTransform()
        est = ExtractSpatialTransformCrop()
        tf = TransformSequence(rbf, rct, rat, rbt, est)

        self.ds = dataset_factory(
            clean=False, transform=tf, **self.config["train_dataset"]
        )

        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        for a, d in zip(ax.flatten(), self.ds):
            # img = d["image_augmented_32"]
            img = d["image_augmented_64"]
            # img = d["image_composed"]
            a.imshow(to_img(img))
        plt.savefig(store_path / "test_augmentation.png", bbox_inches="tight")

        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        for a, d in zip(ax.flatten(), self.ds):
            img = d["image_st_gt"]
            a.imshow(to_img(img))
        plt.savefig(store_path / "test_recrop.png", bbox_inches="tight")


if __name__ == "__main__":
    # Set to float64 to avoid rounding errors for testing
    torch.set_default_dtype(torch.float64)
    unittest.main()
