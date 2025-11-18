import unittest
import torch

from ccn.action import Action
from ccn.geometry import look_at_matrix
from ccn.distributions import empirical_truncated_normal

from torch.distributions import MultivariateNormal
from torch.testing import assert_close

import matplotlib.pyplot as plt
from ccn.visualization import plot_pdf, plot_camera_2d, plot_camera
from ccn.camera import PinholeModel, realsense_intrinsics
from ccn.camera import opengl2ee

from pathlib import Path

test_store_dir = Path(__file__).parent / "../data/test/camera"
test_store_dir.mkdir(exist_ok=True, parents=True)


class CameraTestCase(unittest.TestCase):
    def test_to_global_and_back(self):

        for opengl in [True, False]:
            pinhole = PinholeModel(realsense_intrinsics, opengl)

            cam_pos_global = torch.rand((3,))
            lookat_global = torch.rand((3,))
            # Defined in opengl format
            cam_pose_global = look_at_matrix(cam_pos_global, lookat_global)
            if not opengl:
                # Global pose in ee frame
                cam_pose_global = torch.mm(cam_pose_global, opengl2ee)

            points_cam_frame = torch.rand((1000, 3))
            points_glo_frame = pinhole.coords_to_global_frame(
                points_cam_frame, cam_pose_global
            )
            points_cam_frame_hat = pinhole.coords_to_camera_frame(
                points_glo_frame, cam_pose_global
            )
            assert_close(points_cam_frame, points_cam_frame_hat)

            # Test that lookat in camera-coordinates is a value on the [0, 0, -1] direction
            lookat_cam_frame = pinhole.coords_to_camera_frame(
                lookat_global.unsqueeze(0), cam_pose_global
            )[0]
            d = torch.linalg.norm(lookat_global - cam_pos_global)
            lookat_cam_frame_hat = torch.tensor([0, 0, d])
            assert_close(lookat_cam_frame, lookat_cam_frame_hat)

    def test_in_view_filter(self):
        pinhole = PinholeModel(realsense_intrinsics, opengl=True)

        cam_point = (torch.rand(3) - 0.5) * 0.5
        cam_point[-1] = 0
        cam_pose = look_at_matrix(cam_point, torch.tensor([1.0, 1e-4, 1e-4]))

        belief = MultivariateNormal(torch.zeros(3), 0.05 * torch.eye(3))
        coords = belief.sample((500,))

        in_view = pinhole.coords_in_view(coords, cam_pose)

        # Plotting
        # fig, ax = plt.subplots(1, 1, figsize=(8, 4), projection="3d")
        fig = plt.figure(figsize=(12, 3))
        ax = fig.add_subplot(1, 3, 1, projection="3d")

        # Plot samples
        ax.scatter(
            *coords[~in_view][:, :3].T, color="red", marker="o", alpha=0.25
        )
        ax.scatter(
            *coords[in_view][:, :3].T,
            color="green",
            marker="o",
        )
        ax.view_init(0, 30)

        # Plot camera
        plot_camera(cam_pose, ax)
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])

        ax = fig.add_subplot(1, 3, 2, projection="3d")
        ax.scatter(
            *coords[in_view][:, :3].T,
            color="green",
            marker="o",
        )
        ax.view_init(0, 135)
        plot_camera(cam_pose, ax)
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])

        ax = fig.add_subplot(1, 3, 3)
        ax.scatter(
            *coords[in_view][..., :2].T,
            marker=".",
            color="green",
            alpha=1.0,
            label="in view",
        )
        ax.scatter(
            *coords[~in_view][..., :2].T,
            marker=".",
            color="red",
            alpha=0.25,
            label="not in view",
        )
        plot_camera_2d(cam_pose, ax)
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Top View")
        ax.legend()

        plt.savefig(
            test_store_dir / "camera_in_view_filter.png",
            bbox_inches="tight",
        )


if __name__ == "__main__":
    # For testing - otherwise there are rounding errors
    torch.set_default_dtype(torch.float64)
    unittest.main()
