import torch

realsense_intrinsics = torch.tensor(
    [
        [615.132080078125, 0.0, 326.28509521484375],
        [0.0, 615.2212524414062, 231.7800750732422],
        [0.0, 0.0, 1.0],
    ]
)

opengl2ee = torch.tensor(
    [[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 0], [0, 0, 0, 1.0]]
)


class PinholeModel:
    def __init__(self, k_matrix, opengl=True):
        self.k_matrix = k_matrix
        self.opengl = opengl

    def xyz_to_uvd(self, x, y, z):
        """
        Predict the U, V, D for a coordinate assuming the observer camera is
        in the origin! works in EE Frame
        """
        px = self.k_matrix[0][2]
        py = self.k_matrix[1][2]
        fx = self.k_matrix[0][0]
        fy = self.k_matrix[1][1]

        # inhomogenous
        u = fx * x / (z + 1e-8) + px
        v = fy * y / (z + 1e-8) + py
        d = z

        return u, v, d

    def uvd_to_xyz(self, u, v, d):
        fx = self.k_matrix[0][0]
        fy = self.k_matrix[1][1]
        cx = self.k_matrix[0][2]
        cy = self.k_matrix[1][2]

        x_over_z = (u - cx) / fx
        y_over_z = (v - cy) / fy

        z = d
        x = x_over_z * z
        y = y_over_z * z

        return x, y, z

    def coords_to_frame(self, coords, frame):
        """
        convert coordinates to the camera frame, i.e. as if
        the coordinate system of the camera is eye
        """
        coords = torch.cat(
            [coords.unsqueeze(1), torch.ones(coords.shape[0], 1, 1)], dim=-1
        )
        frame = frame.unsqueeze(0).repeat(coords.shape[0], 1, 1)
        coords = torch.bmm(coords, frame.transpose(-2, -1)).squeeze(1)[..., :3]

        return coords

    def coords_to_global_frame(self, coords, camera_frame):
        if self.opengl:
            tf = opengl2ee[:3, :3].T.unsqueeze(0).repeat(len(coords), 1, 1)
            coords = torch.bmm(coords.unsqueeze(1), tf).squeeze(1)
        return self.coords_to_frame(coords, camera_frame)

    def coords_to_camera_frame(self, coords, camera_frame):
        coords = self.coords_to_frame(coords, torch.linalg.inv(camera_frame))
        if self.opengl:
            tf = opengl2ee[:3, :3].unsqueeze(0).repeat(len(coords), 1, 1)
            coords = torch.bmm(coords.unsqueeze(1), tf).squeeze(1)
        return coords

    def coords_in_view(self, coords, camera_pose, d_max=2):
        coords = self.coords_to_camera_frame(coords, camera_pose)
        u, v, d = self.xyz_to_uvd(*coords.T)
        zeros, ones = torch.zeros_like(u), torch.ones_like(u)
        filter_u = (u > zeros) * (u < ones * 640)
        filter_v = (v > zeros) * (v < ones * 480)
        filter_d = (d > zeros) * (d < ones * d_max)
        filter_ = filter_u * filter_d * filter_v
        return filter_
