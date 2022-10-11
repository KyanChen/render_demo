import trimesh
from pyrender import Scene
from . import transformation as ts

import numpy as np
import json
import cv2
import os
import hashlib
import random



def hash_digits(s):
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10 ** 8


############ load 3D model ##############

def load_3d_models(meshdir, sample_m_objs=None):

    print('loading 3d car models...')

    obj_meshes = {}

    mesh_names = os.listdir(meshdir)

    if sample_m_objs is not None:
        # randomly sample m obj meshes from meshdir
        random.shuffle(mesh_names)
        mesh_names = mesh_names[0:sample_m_objs]

    for i in range(len(mesh_names)):
        try:
            obj_meshes[mesh_names[i]] = {}
            obj_path = os.path.join(
                meshdir, mesh_names[i], 'models/model_normalized.obj')
            data = trimesh.load(obj_path, force='scene')
            scene = Scene.from_trimesh_scene(data)
            M = ts.create_pose_normalization_matrix(scene, rx=0, rz=0, ry=0)
            obj_meshes[mesh_names[i]]['base_M'] = M
            obj_meshes[mesh_names[i]]['meshes'] = scene.meshes
        except:
            print('error in loading obj meshes: %s' % obj_path)

    return obj_meshes



def generate_random_vehicles(n=10, xmin=-15, xmax=15, zmin=-15, zmax=15, vtype='tank'):

    vehicle_list = []
    x_buff, z_buff = [], []

    for i in range(n):
        max_try_time = 20
        for _ in range(max_try_time):
            x, z = random.uniform(xmin, xmax), random.uniform(zmin, zmax)
            if _check_room(x, z, x_buff, z_buff):
                x_buff.append(x)
                z_buff.append(z)

                v = {}
                if vtype == 'tank':
                    v['id'] = str(1).zfill(5)
                elif vtype == 'PP':
                    v['id'] = str(0).zfill(5)
                else:
                    v['id'] = str(random.randint(0, 1)).zfill(5)
                v['h'], v['w'], v['l'] = 2.4, 2.8, 6.5
                v['x'], v['z'] = x, z
                v['y'] = v['h'] / 2.0
                v['heading'] = random.uniform(0, 360)
                vehicle_list.append(v)

                break

    return vehicle_list



def _check_room(x, z, x_buff, z_buff, thresh_dist=10):

    if len(x_buff) == 0:
        return True

    x_buff.append(0)
    z_buff.append(0)
    dx = np.array(x) - np.array(x_buff)
    dz = np.array(z) - np.array(z_buff)
    d = (dx**2 + dz**2)**0.5
    if np.min(d) > thresh_dist:
        return True
    else:
        return False



class VehicleManager(object):
    def __init__(self, n=10, xmin=-15, xmax=15, zmin=-15, zmax=15,
                 vtype='tank', moving_target=False):

        # initilize vehicle list
        self.vehicle_list = generate_random_vehicles(
            n=n, xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, vtype=vtype
        )

        self.moving_target = moving_target

        self.t = 0

    def step(self):

        self.t += 1
        if self.moving_target:

            v = {}
            v['id'] = str(1).zfill(5)  # moving tank
            v['h'], v['w'], v['l'] = 2.4, 2.8, 6.5
            v['y'] = v['h'] / 2.0

            # theta = 170
            # speed = 0.15
            # v['x'] = 0 + speed * self.t * np.sin((theta)/180.*np.pi)
            # v['z'] = 15 + speed * self.t * np.cos((theta)/180.*np.pi)

            # theta = -40
            # speed = 0.15
            # v['x'] = 20 + speed * self.t * np.sin((theta) / 180. * np.pi)
            # v['z'] = -15 + speed * self.t * np.cos((theta) / 180. * np.pi)

            theta = 0
            speed = 0.15
            v['x'] = 5 + speed * self.t * np.sin((theta) / 180. * np.pi)
            v['z'] = -20 + speed * self.t * np.cos((theta) / 180. * np.pi)

            v['heading'] = theta
            self.vehicle_list[0] = v

            return self.vehicle_list
        else:
            return self.vehicle_list

