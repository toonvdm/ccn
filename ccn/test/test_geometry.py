import unittest
import torch
from torch.testing import assert_close

from ccn.geometry import relative_transform, look_at_matrix


class GeometryTestCase(unittest.TestCase):
    def _random_matrix(self):
        cart1 = torch.rand(1000, 3) * 100
        cart2 = torch.rand(1000, 3) * 100
        return look_at_matrix(cart1, cart2)

    def test_lookat_matrix(self):
        """
        Test that the lookat matrix, actually provides a
        matrix defining the pose for looking at a given
        object.
        """
        cart1 = torch.rand(1000, 3) * 100
        cart2 = torch.rand(1000, 3) * 100
        lats = look_at_matrix(cart1, cart2)

        # Assert the position of the camera is valid
        assert_close(lats[..., :3, -1], cart1)

        # Assert when stepping according to distance in the
        # negative z-direction (opengl format) of the camera
        # pose, you end up in the cart2 point
        distance = torch.linalg.norm(cart1 - cart2, dim=-1)
        direction = torch.zeros((1000, 4, 1))
        distance = distance.unsqueeze(-1)
        direction[:, 2, :] = -distance
        direction[:, 3, :] = 1
        res = torch.bmm(lats, direction).squeeze(1)[..., :3, 0]
        assert_close(res, cart2, rtol=1e-5, atol=1e-5)

    def test_lookat_matrix_batch(self):
        """
        Test that the lookat matrix provides the same
        results, batched or unbatched
        """
        cart1 = torch.rand(1000, 3) * 100
        cart2 = torch.rand(1000, 3) * 100
        lats = look_at_matrix(cart1, cart2)
        for c1, c2, lat in zip(cart1, cart2, lats):
            assert_close(lat, look_at_matrix(c1, c2))

    def test_relative_transform(self):
        """
        Test that applying the relative transform to the
        initial matrix, yields you the second matrix again
        """
        mat1 = self._random_matrix()
        mat2 = self._random_matrix()
        action = relative_transform(mat1, mat2)
        mat2_hat = torch.bmm(mat1, action)
        assert_close(mat2, mat2_hat, rtol=1e-5, atol=1e-3)


if __name__ == "__main__":
    # Set to float64 to avoid rounding errors for testing
    torch.set_default_dtype(torch.float64)
    unittest.main()
