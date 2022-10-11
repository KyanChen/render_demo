from . import utils
from . import transformation as ts

import pyrender

import numpy as np
import os

from pathlib import Path


class NodeManager(object):

    def __init__(self):

        self.obj_meshes = utils.load_3d_models(
            meshdir=os.path.join(str(Path(__file__).resolve().parent), '3dmesh'))

    def create_vehicles(self, vehicle_list):

        vehicle_nodes = []

        for i in range(len(vehicle_list)):

            v = vehicle_list[i]

            m = len(self.obj_meshes.keys())
            idx = utils.hash_digits(v['id']) % m
            objmesh = self.obj_meshes[list(self.obj_meshes.keys())[idx]]

            meshes = objmesh['meshes']
            base_correction = objmesh['base_M']

            pose_matrix = ts.create_vehicle_pose(v=v)
            pose_matrix = np.array(np.mat(pose_matrix) * np.mat(base_correction))
            for x in meshes:
                node = pyrender.Node(mesh=x, matrix=pose_matrix)
                vehicle_nodes.append((node, v))

        return vehicle_nodes


