import unittest
import torch
from torch.testing import assert_close

from ccn.models.spatial_transformers import SpatialTransformer


class SpatialTransformerTestCase(unittest.TestCase):
    def test_st_xy_to_uv_and_back(self):
        st = SpatialTransformer(480)
        x, y = 2 * torch.rand((100)) - 1, 2 * torch.rand((100)) - 1
        u, v = st.uv_to_xy(x, y)
        x_hat, y_hat = st.uv_to_xy(u, v)
        assert_close(x_hat, x)
        assert_close(y_hat, y)

        u, v = 480, 480
        u_hat, v_hat = st.xy_to_uv(-1, -1)
        self.assertEqual(u, u_hat)
        self.assertEqual(v, v_hat)

        u, v = 240, 240
        u_hat, v_hat = st.xy_to_uv(0, 0)
        self.assertEqual(u, u_hat)
        self.assertEqual(v, v_hat)

        u, v = 0, 0
        u_hat, v_hat = st.xy_to_uv(1, 1)
        self.assertEqual(u, u_hat)
        self.assertEqual(v, v_hat)


if __name__ == "__main__":
    # Set to float64 to avoid rounding errors for testing
    torch.set_default_dtype(torch.float64)
    unittest.main()
