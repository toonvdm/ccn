import torch
import torch.nn as nn

import numpy as np

# from torch.distributions import Normal
from dommel.distributions import MultivariateNormal

from dommel.datastructs import TensorDict
from ccn.models.spatial_transformers import SpatialTransformer
from ccn.models.medianpool import MedianPool2d

import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.reshape(input.size(0), size, 1, 1)


class MLPUnFlatten(nn.Module):
    def __init__(self, in_features, output_shape):
        nn.Module.__init__(self)

        out_features = output_shape[0] * output_shape[1] * output_shape[2]
        self.output_shape = output_shape
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input, *args, **kwargs):
        x = self.linear(input)
        x = x.reshape(input.size(0), *self.output_shape)
        return x


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        nn.Module.__init__(self)
        self.size = size
        self.mode = mode

    def forward(self, x):
        return F.interpolate(
            x, size=self.size, mode=self.mode, align_corners=False
        )


class ImageEncoder(nn.Module):
    def __init__(self, z_size=8):
        nn.Module.__init__(self)

        self.z_size = z_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(256, 2 * self.z_size),
        )

        self.softplus = nn.Softplus()

    def forward(self, x):
        h = self.encoder(x)
        mean = h[:, : self.z_size]
        scale = self.softplus(h[:, self.z_size :]) + 1e-4
        return MultivariateNormal(mean, scale)


class ImageDecoder(nn.Module):
    def __init__(self, z_size=8):
        nn.Module.__init__(self)

        self.z_size = z_size

        self.fc = nn.Linear(self.z_size, 128)
        self.unflatten = MLPUnFlatten(128, (64, 8, 8))
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1),
            nn.LeakyReLU(),
            Interpolate((17, 17), "bilinear"),
            nn.Conv2d(64, 64, kernel_size=5, stride=1),
            nn.LeakyReLU(),
            Interpolate((35, 35), "bilinear"),
            nn.Conv2d(64, 32, kernel_size=6, stride=1),
            nn.LeakyReLU(),
            Interpolate((69, 69), "bilinear"),
            nn.Conv2d(32, 16, kernel_size=6, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 3, kernel_size=1, stride=1),
        )

    def forward(self, z):
        z = self.unflatten(self.fc(z))
        x_hat = self.decoder(z)
        return x_hat


class Transition(nn.Module):
    def __init__(self, z_size=8, action_size=2):
        nn.Module.__init__(self)

        self.z_size = z_size
        self.action_size = action_size

        self.transition = nn.Sequential(
            nn.Linear(self.z_size + self.action_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2 * self.z_size),
        )

        self.softplus = nn.Softplus()

    def forward(self, z, a):
        z_a = torch.cat([z, a], dim=1)
        h = self.transition(z_a)
        mean = h[:, : self.z_size]
        scale = self.softplus(h[:, self.z_size :]) + 1e-4
        return MultivariateNormal(mean, scale)


class STEncoder(nn.Module):
    def __init__(self, r_max=8, activate=False, invert_scale=True):
        nn.Module.__init__(self)
        self.spatial_transformer = SpatialTransformer()

        self.encoder = nn.Sequential(
            Interpolate((32, 32), mode="bilinear"),
            Flatten(),
            nn.Linear(3 * 32 * 32, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 3),
        )

        self.activation = lambda x: x
        if activate:
            self.activation = nn.ELU()  # nn.LeakyReLU()

        # 1 / scale which is [1/r_max; np.sqrt(2)/0.4]
        # based on a 1x1 area that the agent can be in
        # and a base distance of .4 m
        # self.offset = 0.4 / np.sqrt(2)
        # self.range = r_max - self.offset
        self.offset_xy = torch.tensor([0, 100.0, 100.0]).unsqueeze(0)
        self.r_max = r_max

        self._invert_scale = lambda x: x
        if invert_scale:
            self._invert_scale = lambda x: 1 / ((self.r_max - 0.2) * x + 0.2)

    def forward(self, x):
        # making sure that the LeakyReLU is only applied the scale factor
        h = self.encoder(x)
        self.offset_xy = self.offset_xy.to(x.device)
        h = self.activation(h + self.offset_xy) - self.offset_xy
        h[:, 0] = self._invert_scale(h[:, 0])
        x_out = self.spatial_transformer.forward(h, x)
        return x_out, h


class ObjectMask(nn.Module):
    def __init__(self, median_pool=False):
        nn.Module.__init__(self)

        mask_modules = [
            nn.Conv2d(3, 16, kernel_size=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding="same"),
        ]
        if median_pool:
            mask_modules.append(MedianPool2d(5, same=True))
        mask_modules.append(nn.Sigmoid())
        self.mask = nn.Sequential(*mask_modules)

    def forward(self, x):
        m = self.mask(x)
        return m * x, m


class MSTN(nn.Module):
    def __init__(self, invert_scale):
        nn.Module.__init__(self)
        self.mask = ObjectMask(median_pool=True)
        self.stn = STEncoder(4, True, invert_scale)

        self.interpolate64 = Interpolate((64, 64), "bilinear")
        self.interpolate480 = Interpolate((480, 480), "bilinear")

        self.presence = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Spatial encode first
        x_masked_64, mask = self.mask(self.interpolate64(x))
        x_masked = (x * (self.interpolate480(mask))).clone().detach()
        crop, z_where = self.stn(x_masked)
        presence_hat = self.presence(crop)
        return crop, x_masked_64, presence_hat, z_where
        # return crop, x_masked, presence_hat, z_where


class VanillaCCN(nn.Module):
    def __init__(self, z_size, action_size, freeze=False):
        nn.Module.__init__(self)
        self.encoder = ImageEncoder(z_size)
        self.decoder = ImageDecoder(z_size)
        self.transition = Transition(z_size, action_size)

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, a, b, action):
        batch_size = a.shape[0]

        # Batch Encode for both before and after transition
        qs = self.encoder(torch.cat([a, b], dim=0))

        # Batch decode for before and after transition
        z = qs.sample()
        x_hats = self.decoder(z)

        # Only transition the first
        q_b_transitioned = self.transition(z[:batch_size], action)
        z_next_hat = q_b_transitioned.sample()
        x_b_transitioned_hat = self.decoder(z_next_hat)

        q_a_mean, q_b_mean = qs.mean[:batch_size], qs.mean[batch_size:]
        q_a_var, q_b_var = qs.variance[:batch_size], qs.variance[batch_size:]

        return (
            MultivariateNormal(q_a_mean, q_a_var),  # q_a
            MultivariateNormal(q_b_mean, q_b_var),  # q_b
            q_b_transitioned,  # q_b_transitioned
            x_hats[:batch_size],  # x_a
            x_hats[batch_size:],  # x_b
            x_b_transitioned_hat,  # x_transitioned
        )
