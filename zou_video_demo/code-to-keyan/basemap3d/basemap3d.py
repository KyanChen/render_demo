import numpy as np

import pyrender
from pyrender import DirectionalLight

from . import transformation as ts
from .node_manager import NodeManager


class Render3D(object):

    def __init__(self, bg_color=0.0, ambient_light=50):

        # an empty scene
        self.scene = pyrender.Scene(bg_color=bg_color)

        # node manager and caption manager
        self.node_manager = NodeManager()
        # create lighting
        # self.scene.add(DirectionalLight(color=np.ones(3), intensity=0.0),
        #     pose=ts.create_pose_matrix(rx=-90))
        self.scene.ambient_light = ambient_light

        # nodes for vehicles
        self.vehicle_nodes = []

    def update_directional_light(self, new_pose=None, new_color=None,
                                 new_intensity=None):
        node = list(self.scene.directional_light_nodes)[0]
        if new_pose is not None:
            node.matrix = new_pose
        if new_color is not None:
            node.light.color = new_color
        if new_intensity is not None:
            node.light.intensity = new_intensity

    def update_point_light(self, new_pose=None, new_color=None,
                                 new_intensity=None):
        node = list(self.scene.point_light_nodes)[0]
        if new_pose is not None:
            node.matrix = new_pose
        if new_color is not None:
            node.light.color = new_color
        if new_intensity is not None:
            node.light.intensity = new_intensity

    def update_ambient_light(self, new_intensity=None):
        if new_intensity is not None:
            self.scene.ambient_light = new_intensity

    def update_camera_pose(self, new_pose=None):
        cam_node = list(self.scene.camera_nodes)[0]
        if new_pose is not None:
            self.scene.set_pose(cam_node, new_pose)

    def clear_vehicle_nodes(self):
        for (bn, v) in self.vehicle_nodes:
            self.scene.remove_node(bn)

    def create_vehicle_nodes(self, vehicle_list):
        self.vehicle_nodes = self.node_manager.create_vehicles(vehicle_list)
        for (bn, v) in self.vehicle_nodes:
            self.scene.add_node(bn)



class BasemapOffscreen3D(Render3D):

    def __init__(self, cam_pose=None,
                 yfov=np.pi / 3.0, viewport_h=768, viewport_w=768, ambient_light=50, bg_color=0.0):

        super(BasemapOffscreen3D, self).__init__(bg_color=bg_color, ambient_light=ambient_light)

        # off screen renderer
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=viewport_w/viewport_h)
        if cam_pose is None:
            cam_pose = ts.create_pose_matrix()
        self.scene.add(camera, pose=cam_pose)
        self.viewer = pyrender.OffscreenRenderer(viewport_w, viewport_h)

        self.scene.add(pyrender.PointLight(intensity=10., color=255.0),
                       pose=ts.create_pose_matrix(tz=50))


    def render(self, vehicle_list, cam_pose=None):

        self.clear_vehicle_nodes()
        self.create_vehicle_nodes(vehicle_list)
        self.update_camera_pose(cam_pose)

        flags = pyrender.RenderFlags.RGBA + pyrender.RenderFlags.SHADOWS_DIRECTIONAL
        rgba, depth = self.viewer.render(self.scene, flags=flags)
        color, alpha = np.array(rgba[:, :, 0:3]), np.array(rgba[:, :, -1])
        alpha = np.stack([alpha, alpha, alpha], axis=-1)

        return color, alpha, depth

