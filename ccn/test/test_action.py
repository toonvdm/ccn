import unittest
import torch

from ccn.action import Action
from ccn.geometry import look_at_matrix

from ccn.dataset.object_dataset import cartesian_to_spherical as c2s_data

from torch.testing import assert_close

from scene_environments.environments.scene_environment import SceneEnvironment


class ActionTestCase(unittest.TestCase):
    def _random_spherical(self):
        spherical = torch.rand(1000, 3)
        spherical[..., 0] *= 2 * torch.pi
        spherical[..., 1] = (spherical[..., 1] - 0.5) * torch.pi  # / 2
        spherical[..., 2] *= 100
        return spherical

    def _random_matrix(self):
        cart1 = (torch.rand(1000, 3) - 0.5) * 100
        cart2 = (torch.rand(1000, 3) - 0.5) * 100
        return look_at_matrix(cart1, cart2)

    def test_from_to_matrix(self):
        matrices = self._random_matrix()
        action = Action.from_matrix(matrices)
        assert_close(action.matrix, matrices)

    def test_matrix_batch(self):
        matrices = self._random_matrix()
        for mat in matrices:
            action = Action.from_matrix(mat)
            assert_close(action.matrix, mat)

    def test_from_to_spherical(self):
        spherical = self._random_spherical()
        action = Action.from_spherical(spherical)
        assert_close(spherical, action.spherical)

    def test_from_to_spherical_batch(self):
        spherical = self._random_spherical()
        for s in spherical:
            assert_close(Action.from_spherical(s).spherical, s)

    def test_matrix_to_any_and_back(self):
        matrix = self._random_matrix()

        # only check translation for spherical
        s1 = Action.from_matrix(matrix).spherical
        s2 = Action.from_spherical(s1).spherical
        assert_close(s1[..., :3, -1], s2[..., :3, -1])

        cq1 = Action.from_matrix(matrix).cartesian_quaternion
        cq2 = Action.from_cartesian_quaternion(cq1).cartesian_quaternion
        assert_close(cq1, cq2, rtol=1e-4, atol=1e-4)

        c61 = Action.from_matrix(matrix).cartesian_6dof
        c62 = Action.from_cartesian_6dof(c61).cartesian_6dof
        assert_close(c61, c62, rtol=1e-5, atol=1e-5)

    def test_euler_and_back(self):
        matrices = self._random_matrix()
        actions = Action.from_matrix(matrices)
        andback = Action.from_cartesian_euler(actions.cartesian_euler)
        assert_close(actions.matrix, andback.matrix)

    def test_euler_scene_env_match(self):
        matrices = self._random_matrix()
        actions = Action(matrices)
        cart_euler = actions.cartesian_euler
        # Move to range -1, 1
        cart_euler[:, 3:] /= 2 * torch.pi
        cart_euler[:, 3:] += 1 / 2

        for a, m in zip(cart_euler, matrices):
            m_hat = SceneEnvironment.action_to_matrix(a)
            assert_close(m_hat, m.to(torch.float64), rtol=1e-5, atol=1e-5)

    def test_action_splitup(self):
        """
        Tests that if you linearly interpolate an action in N steps,
        in euler space; it's equivalent to doing it straight away
        ....
        Only test interpolation of the translation; all subsequent
        rotations would shift as the rotation is w.r.t. the original
        reference system
        """
        matrices = self._random_matrix()
        matrices[:, :3, :3] = torch.eye(3).repeat((matrices.shape[0], 1, 1))
        actions = Action(matrices)

        step_size = (
            torch.linalg.norm(actions.matrix[..., :3, -1], dim=-1) / 100
        )

        step = actions.get_step(step_size).matrix
        mat = torch.eye(4).unsqueeze(0).repeat(matrices.shape[0], 1, 1)
        mat = torch.bmm(mat, actions.matrix)

        mat_hat = torch.eye(4).unsqueeze(0).repeat(matrices.shape[0], 1, 1)
        for i in range(100):
            mat_hat = torch.bmm(mat_hat, step)

        assert_close(mat_hat, mat, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    # For testing - otherwise there are rounding errors
    torch.set_default_dtype(torch.float64)
    unittest.main()
