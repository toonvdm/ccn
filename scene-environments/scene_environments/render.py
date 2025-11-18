import numpy as np
import copy

import pyrender
from pyrender.constants import RenderFlags as rf

import trimesh
from trimesh.resolvers import FilePathResolver


def load_mesh(mesh_path):
    mesh_kwargs = trimesh.exchange.dae.load_collada(
        mesh_path, resolver=FilePathResolver(mesh_path)
    )
    k = list(mesh_kwargs["geometry"].keys())[0]
    mesh_trimesh = trimesh.Trimesh(**mesh_kwargs["geometry"][k])
    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)
    mesh._name = mesh_path
    return mesh


ycb_classes = (
    "__background__",
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "040_large_marker",
    "051_large_clamp",
    "052_extra_large_clamp",
    "061_foam_brick",
    "holiday_cup1",
    "holiday_cup2",
    "sanning_mug",
)


class Renderer:
    def __init__(
        self,
        objects,
        intrinsics,
        resolution,
        meshes=None,
        crop_shape=None,
        plane=False,
        wall=False,
        background_color=(0, 0, 0),
        table_color=None,
        wall_color=None,
        object_centric=False,
    ):

        # whether to place the object center at the origin
        self.object_centric = object_centric
        self.cam_node = None

        self.object_nodes = {}
        if meshes is None:
            self.meshes = {}
            for mesh_path, _ in objects.items():
                self.meshes[mesh_path] = load_mesh(mesh_path)
        else:
            self.meshes = meshes

        if plane:
            trimesh_plane = trimesh.creation.box((1, 1, 0.1))
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=table_color
            )
            self.table_mesh = pyrender.Mesh.from_trimesh(
                trimesh_plane, material
            )
        else:
            self.table_mesh = None

        if wall:
            trimesh_wall = trimesh.creation.box((0.025, 1, 0.25))
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=wall_color
            )
            self.wall_mesh = pyrender.Mesh.from_trimesh(trimesh_wall, material)
        else:
            self.wall_mesh = None

        self.render_resolution = resolution

        self.scene = self.build_scene(
            objects, self.meshes, background_color, plane, wall
        )

        self.camera = self.build_camera(intrinsics, resolution, crop_shape)

        self.r = pyrender.OffscreenRenderer(*self.render_resolution)

    def set_table_color(self, color):
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=color)
        self.table_mesh.primitives[0].material = material

    def set_wall_color(self, color):
        if self.wall_mesh is not None:
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=color
            )
            self.wall_mesh.primitives[0].material = material

    def build_camera(
        self, intrinsics, render_resolution, output_resolution=None
    ):
        # Set up render parameters
        # Max look 2m far
        camera = pyrender.IntrinsicsCamera(
            fx=intrinsics[0][0],
            fy=intrinsics[1][1],
            cx=intrinsics[0][2],
            cy=intrinsics[1][2],
            zfar=3,
        )

        # vars used to square crop in the end
        self.w0, self.w1 = 0, render_resolution[0]
        self.h0, self.h1 = 0, render_resolution[1]
        if output_resolution is not None:
            self.w0 = (render_resolution[0] - output_resolution[0]) // 2
            self.w1 = -self.w0 if self.w0 != 0 else self.w1

            self.h0 = (render_resolution[1] - output_resolution[1]) // 2
            self.h1 = -self.h0 if self.h0 != 0 else self.h1

        return camera

    def build_scene(self, objects, meshes, background_color, plane, wall):

        self.object_nodes = {}

        # Build the scene
        scene = pyrender.Scene(
            ambient_light=[1.0, 1.0, 1.0], bg_color=background_color
        )

        # scene = pyrender.Scene(
        #     ambient_light=[0.02, 0.02, 0.02], bg_color=background_color
        # )
        # light_pose = np.eye(4)
        # light_pose[:3, -1] = [0, 0, 0.5]
        # scene.add(
        #     # pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=0.5), pose=light_pose
        #     pyrender.SpotLight(
        #         color=[1.0, 1.0, 1.0],
        #         intensity=2.0,
        #         innerConeAngle=0.05,
        #         outerConeAngle=0.5,
        #     )
        # )

        # Add objects to the scene
        for (mesh_path, mesh_pose) in objects.items():
            mesh = meshes[mesh_path]
            pose = mesh_pose.copy()
            # set pose w.r.t. the object center, but keep the
            # z-value, so that it is positioned on the table
            if self.object_centric:
                pose[:3, -1] -= mesh.centroid[:3]
            else:
                pose[:2, -1] -= mesh.centroid[:2]

            # Copy the mesh because pyrender makes changes to the
            # mesh, and then multiple renderers can not use the
            # same loaded mesh
            self.object_nodes[mesh_path] = scene.add(
                copy.deepcopy(mesh), pose=pose, name=mesh_path
            )

        # Render a ground plane (table) surface
        if plane:
            plane_pose = np.eye(4)
            plane_pose[2, -1] = -0.05
            self.object_nodes["table"] = scene.add(
                self.table_mesh, pose=plane_pose
            )

        if wall:
            wall_pose = np.eye(4)
            wall_pose[0, -1] = 0.0125
            wall_pose[2, -1] = 0.125
            self.object_nodes["wall"] = scene.add(
                self.wall_mesh, pose=wall_pose
            )

        return scene

    def update_object_positions(self, poses_dict, hide_others=False):

        if hide_others:
            for mn in self.scene.mesh_nodes:
                mn.mesh.is_visible = False

        for (pose_key, mesh_pose) in poses_dict.items():
            node = self.object_nodes[pose_key]
            mesh = self.meshes[pose_key]

            pose = mesh_pose.copy()
            # set pose w.r.t. the object center, but keep the
            # z-value, so that it is positioned on the table
            # only move the z-axis?
            if self.object_centric:
                pose[:3, -1] -= mesh.centroid[:3]
            else:
                pose[:2, -1] -= mesh.centroid[:2]
            node.mesh.is_visible = True
            self.scene.set_pose(node, pose)

    def remove_nodes(self, target_names):
        for name in target_names:
            node = self.object_nodes[name]
            self.scene.remove_node(node)

    def clear(self):
        for node in self.object_nodes.values():
            self.scene.remove_node(node)

    def render(self, camera_pose, render_instance_mask=False):
        """
        :param camera_pose: A 4x4 matrix describing the camera's extrinsics
        :param resolution: The resolution of the rendered image
        :return: RGB image, Depth image
        """
        flags = rf.FLAT | rf.OFFSCREEN
        # flags = rf.OFFSCREEN | rf.SHADOWS_DIRECTIONAL

        self.cam_node = self.scene.add(self.camera, pose=camera_pose)
        color, depth = self.r.render(self.scene, flags=flags)

        # Render the instance mask
        instance_mask = np.zeros_like(depth)
        if render_instance_mask:
            for mn in self.scene.mesh_nodes:
                mn.mesh.is_visible = False

            for i, node in enumerate(self.scene.mesh_nodes):
                name = node.name
                if name is not None:
                    node.mesh.is_visible = True
                    name = name.split("/")[-2].replace("_textured", "")

                    d_i = self.r.render(
                        self.scene, flags=flags | rf.DEPTH_ONLY
                    )
                    mask = depth
                    mask = np.logical_and(
                        (np.abs(d_i - depth) < 1e-6), np.abs(depth) > 0
                    )
                    try:
                        instance_mask[mask] = ycb_classes.index(name)
                    except ValueError:
                        instance_mask[mask] = -1
                    node.mesh.is_visible = False

            for mn in self.scene.mesh_nodes:
                mn.mesh.is_visible = True

        self.scene.remove_node(self.cam_node)

        return (
            color[self.h0 : self.h1, self.w0 : self.w1, :],
            depth[self.h0 : self.h1, self.w0 : self.w1],
            instance_mask[self.h0 : self.h1, self.w0 : self.w1],
        )
