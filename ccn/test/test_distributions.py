import unittest
import torch

from ccn.action import Action
from ccn.geometry import look_at_matrix
from ccn.distributions import empirical_truncated_normal

from torch.distributions import MultivariateNormal
from torch.testing import assert_close

import matplotlib.pyplot as plt
from ccn.visualization import plot_pdf, plot_camera_2d
from ccn.distributions.particle import ParticleFilter

from pathlib import Path

from ccn.util import get_data_path

test_store_dir = get_data_path() / "test/distributions"
test_store_dir.mkdir(exist_ok=True, parents=True)


class DistributionsTestCase(unittest.TestCase):
    def test_empirical_truncated_norm(self):
        # cam_pose = torch.eye(4)
        cam_point = torch.zeros(3)
        cam_point[0] = -0.25
        cam_pose = look_at_matrix(cam_point, torch.tensor([1.0, 1e-4, 1e-4]))

        belief = MultivariateNormal(torch.zeros(3), 0.01 * torch.eye(3))
        truncated_1 = empirical_truncated_normal(belief, cam_pose)
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        plot_pdf(ax[0], fig, belief)
        ax[0].set_title("Belief and camera before observing")
        plot_pdf(ax[1], fig, truncated_1)
        ax[1].set_title("Belief and camera after observing")
        [plot_camera_2d(cam_pose, a) for a in ax.flatten()]
        [a.plot([0], [0], marker="x", color="black") for a in ax.flatten()]
        plt.savefig(
            test_store_dir / "empirical.png",
            bbox_inches="tight",
        )

    def test_particle_filter_sampling(self):
        ranges = torch.rand((3, 2)) - 0.5
        pf = ParticleFilter(n_particles=3000, ranges=ranges)

        for i in range(ranges.size(0)):
            mini = pf.particles[:, i].min()
            maxi = pf.particles[:, i].max()

            self.assertEqual(mini >= ranges[i].min(), True)
            self.assertEqual(maxi <= ranges[i].max(), True)

    def test_particle_filter(self):
        pf = ParticleFilter()
        belief = MultivariateNormal(torch.rand(3) * 0.5, 0.01 * torch.eye(3))

        fig, ax = plt.subplots(1, 4, figsize=(12, 4))

        plot_pdf(ax[0], fig, belief)
        ax[0].scatter(
            *pf.particles[:, :2].T, marker="o", alpha=0.75, color="red"
        )

        ax[0].set_title("Step 0")
        ax[1].set_title("Step 1")
        ax[2].set_title("Step 10")

        pf.update(belief)
        pf.resample()
        pf.predict()

        plot_pdf(ax[1], fig, belief)
        ax[1].scatter(
            *pf.particles[:, :2].T, marker="o", alpha=0.75, color="red"
        )

        for _ in range(10):
            pf.update(belief)
            pf.resample()
            pf.predict()

        plot_pdf(ax[2], fig, belief)
        ax[2].scatter(
            *pf.particles[:, :2].T, marker="o", alpha=0.75, color="red"
        )

        belief = MultivariateNormal(torch.rand(3) * 0.5, 0.01 * torch.eye(3))
        for _ in range(10):
            pf.update(belief)
            pf.resample()
            pf.predict()

        plot_pdf(ax[3], fig, belief)
        ax[3].scatter(
            *pf.particles[:, :2].T, marker="o", alpha=0.75, color="red"
        )
        ax[3].set_title("10 steps after belief has changed")

        plt.savefig(
            test_store_dir / "particle_filter.png", bbox_inches="tight"
        )
        plt.close()


if __name__ == "__main__":
    # For testing - otherwise there are rounding errors
    torch.set_default_dtype(torch.float64)
    unittest.main()
