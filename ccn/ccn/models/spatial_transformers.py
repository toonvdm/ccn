import torch
import torch.nn as nn
from torch.nn.functional import grid_sample, affine_grid


def bounding_box(z_where, x_size=480):
    """This doesn't take into account interpolation, but it's close
    enough to be usable."""
    w = x_size / z_where[0]
    h = x_size / z_where[0]
    xtrans = -z_where[1] / z_where[0] * x_size / 2.0
    ytrans = -z_where[2] / z_where[0] * x_size / 2.0
    x = (x_size - w) / 2 + xtrans  # origin is top left
    y = (x_size - h) / 2 + ytrans
    return (x, y), w, h


def infer_uv(z_inv):
    """
    z_inv (w.r.t. inferred z)
    """
    (x, y), w, h = bounding_box(z_inv)
    u, v = y + h / 2, x + w / 2
    return u, v


def z_where_inv(z_where):
    n = z_where.size(0)
    out = torch.cat(
        (torch.ones([1, 1]).type_as(z_where).expand(n, 1), -z_where[:, 1:]), 1
    )
    out = out / z_where[:, 0:1]
    return out.to(device=z_where.device)


class SpatialTransformer:
    """
    Based on the Pyro tutorial for AIR: https://pyro.ai/examples/air.html
    Crucially; this model has no learnable parameters, as z-where is consistently
    profided as an input
    """

    def __init__(self, resolution=480):
        self.resolution = resolution

    def uv_to_xy(self, u, v):
        # 1, 1 is top right, -1, -1 is bottom left
        x = 1 - u / (0.5 * self.resolution)
        y = 1 - v / (0.5 * self.resolution)
        return x, y

    def xy_to_uv(self, x, y):
        u = (1 - x) * self.resolution * 0.5
        v = (1 - y) * self.resolution * 0.5
        return u, v

    @staticmethod
    def expand_z_where(z_where):
        """
        # Takes 3-dimensional vectors, and massages them into 2x3 matrices
        # with elements like so:
        # [s,x,y] -> [[s,0,x],
        #             [0,s,y]]
        """
        device = z_where.device
        n = z_where.size(0)
        expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3]).to(device)
        out = torch.cat(
            (torch.zeros([1, 1]).expand(n, 1).to(device), z_where), 1
        )
        return torch.index_select(out, 1, expansion_indices).view(n, 2, 3)

    @staticmethod
    def z_where_inv(z_where):
        """
        # Take a batch of z_where vectors, and compute their "inverse".
        # That is, for each row compute:
        # [s,x,y] -> [1/s,-x/s,-y/s]
        # These are the parameters required to perform the inverse of the
        # spatial transform performed in the generative model.
        """
        return z_where_inv(z_where)

    def forward(self, z_where, image):
        """
        :param z_where: latent describing the position of the object
        :param image: the observation from which to get the crop
        """
        n = image.size(0)
        theta_inv = self.expand_z_where(z_where)  # self.z_where_inv(z_where))
        grid = affine_grid(
            theta_inv, torch.Size((n, 3, 64, 64)), align_corners=False
        )
        out = grid_sample(
            image.view(n, 3, self.resolution, self.resolution),
            grid,
            align_corners=False,
        )
        return out  # .view(n, -1)

    def inverse(self, z_where, obj):
        """
        Inverse action of the spatial transformer, i.e.
        a `where` latent is decoded into an affine transform
        :param z_where: latent describing the position of the object
        :param obj: the rendered crop of the object
        """
        n = obj.size(0)
        theta = self.expand_z_where(z_where)
        grid = affine_grid(
            theta,
            torch.Size((n, 3, self.resolution, self.resolution)),
            align_corners=False,
        ).to(dtype=torch.float32, device=theta.device)
        out = grid_sample(obj.view(n, 3, 64, 64), grid, align_corners=False)
        return out.view(n, 3, self.resolution, self.resolution)
