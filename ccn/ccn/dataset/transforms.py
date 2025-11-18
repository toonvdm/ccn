import numpy as np

import torch
from torch.distributions.uniform import Uniform
import torchvision

from ccn.models.spatial_transformers import SpatialTransformer
from ccn.models.spatial_transformer_ccn import Interpolate


def pixel_mask(x):
    m = torch.zeros_like(x, device=x.device, dtype=torch.float32)
    m = (
        (torch.isclose(m, x, rtol=1e-8, atol=5.0 / 255))
        .prod(dim=0)
        .unsqueeze(0)
        .repeat(3, 1, 1)
    )
    return m.to(torch.float32)


class TransformSequence:
    def __init__(self, *args):
        self.transforms = args

    def __call__(self, batch):
        for tf in self.transforms:
            batch = tf(batch)
        return batch


class RandomBlackFrame:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, batch):
        img = batch["image_in"]
        is_black = torch.rand(1) <= self.p
        if is_black:
            batch["image_in"] = torch.zeros_like(img)
        batch["presence"] = torch.tensor(
            [~is_black], dtype=img.dtype, device=img.device
        )
        return batch


class RandomAnchorTransform:
    def __init__(self) -> None:

        low = torch.tensor([0.4 / np.sqrt(2), -1, -1])
        high = torch.tensor([4.0, 1, 1])

        self.spatial_transformer = SpatialTransformer(480)
        self.prior = Uniform(low=low, high=high)
        self.interpolate = Interpolate((64, 64), mode="bilinear")

    def __call__(self, batch):
        xi = batch["negative_anchor"].unsqueeze(0)
        sampled_sxy = self.prior.sample((xi.shape[0],)).to(xi.device)
        sampled_sxy[:, 1:] *= sampled_sxy[:, 0]
        anchor = self.spatial_transformer.inverse(sampled_sxy, xi)[0]

        m = pixel_mask(anchor)

        batch["image_composed"] = (
            anchor * (1 - m) + m * batch["image_composed"]
        )

        # overwrite this guy
        batch["image_augmented_64"] = self.interpolate(
            batch["image_composed"].unsqueeze(0)
        )[0]
        return batch


class RandomBackgroundTransform:
    def __init__(self, image_key="image_composed"):
        self.image_key = image_key
        self.spatial_transformer = SpatialTransformer(480)

        # self.interpolate = Interpolate((32, 32), mode="bilinear")
        self.interpolate = Interpolate((64, 64), mode="bilinear")

        self.randomrotate = torchvision.transforms.RandomRotation(90, fill=1)

    def apply_random_background(self, x):
        m = pixel_mask(x)

        bg1 = torch.rand(3, 1, 1).to(device=x.device, dtype=x.dtype)
        bg2 = torch.rand(3, 1, 1).to(device=x.device, dtype=x.dtype)
        # bg2 = torch.rand_like(x)

        _, w, h = x.shape
        w_min, w_max = sorted([int(i) for i in list(w * np.random.random(2))])
        h_min, h_max = sorted([int(i) for i in list(h * np.random.random(2))])

        m_bg = torch.ones_like(x, device=x.device, dtype=x.dtype)
        m_bg[:, w_min:w_max, h_min:h_max] = 0
        m_bg = self.randomrotate(m_bg)
        bg = bg1 * m_bg + (1 - m_bg) * bg2

        return m * bg + (1 - m) * x

    def __call__(self, batch):
        batch[self.image_key] = self.apply_random_background(
            batch[self.image_key]
        )

        batch["image_st_gt"] = self.spatial_transformer.forward(
            batch["sxy"], batch[self.image_key].unsqueeze(0)
        )[0]

        batch["image_augmented_64"] = self.interpolate(
            batch[self.image_key].unsqueeze(0)
        )[0]
        return batch


class RandomComposeTransform:
    """
    Apply a random affine transform on the observation, creating a
    480x480 image, where the object is randomly positioned and should be
    extracted somehow
    """

    def __init__(self, s_range=None, curriculum=False):

        if not s_range:
            s_range = [0.4 / np.sqrt(2), 8.0]
        self.s_range = s_range
        low = torch.tensor([self.s_range[0], -1, -1])
        high = torch.tensor([self.s_range[1], 1, 1])
        self.prior = Uniform(low=low, high=high)

        self.spatial_transformer = SpatialTransformer(480)
        self.interpolate = Interpolate((32, 32), mode="bilinear")
        self.interpolate64 = Interpolate((64, 64), mode="bilinear")

        self.div = 5

        self.curriculum = curriculum
        self.counter = 0
        self.update_prior()

    def update_prior(self):
        batch_size = 500
        epochs = 100
        if self.curriculum and self.counter % (batch_size * epochs) == 0:
            self.div = max(self.div - 1, 1.0)
            low = torch.tensor([self.s_range[0], -1 / self.div, -1 / self.div])
            high = torch.tensor([self.s_range[1], 1 / self.div, 1 / self.div])
            self.prior = Uniform(low=low, high=high)
            self.counter = 1

    def __call__(self, batch):

        self.counter += 1
        self.update_prior()

        xi = batch["image_in"].unsqueeze(0)
        sampled_sxy = self.prior.sample((xi.shape[0],)).to(xi.device)
        sampled_sxy[:, 1:] *= sampled_sxy[:, 0]
        result = self.spatial_transformer.inverse(sampled_sxy, xi)[0]
        batch["image_composed"] = result
        batch["image_composed_32"] = self.interpolate(result.unsqueeze(0))[0]
        batch["image_composed_64"] = self.interpolate64(result.unsqueeze(0))[0]
        batch["sxy"] = self.spatial_transformer.z_where_inv(sampled_sxy).to(
            torch.float32
        )
        return batch


class ExtractSpatialTransformCrop:
    def __init__(self) -> None:
        self.spatial_transformer = SpatialTransformer(480)

    def __call__(self, batch):
        batch["image_st_gt"] = self.spatial_transformer.forward(
            batch["sxy"], batch["image_composed"].unsqueeze(0)
        )[0]
        return batch
