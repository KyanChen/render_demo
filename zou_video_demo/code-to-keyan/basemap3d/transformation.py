import numpy as np


############ Pose matrix for basic 3D elements ##############

def create_vehicle_pose(v):
    pose_matrix = create_pose_matrix(
        sx=v['w'], sy=v['h'], sz=v['l'],
        tx=v['x'], ty=v['y'], tz=v['z'], ry=v['heading']
    )
    return pose_matrix


def create_pose_normalization_matrix(
        mesh_or_trimesh_or_scene,
        rx=0, ry=0, rz=0):

    dx, dy, dz = mesh_or_trimesh_or_scene.extents
    xc, yc, zc = mesh_or_trimesh_or_scene.centroid
    M = create_pose_matrix(
        tx=xc, ty=yc, tz=zc,
        sx=dx+1e-9, sy=dy+1e-9, sz=dz+1e-9)
    M_inv = np.linalg.inv(M)

    M_rot = create_pose_matrix(rx=rx, ry=ry, rz=rz)

    base_correction = np.array(np.mat(M_rot) * np.mat(M_inv))

    return base_correction




################ Let's do some math... ####################

def scale_matrix(sx=1.0, sy=1.0, sz=1.0):

    ScaleMatrix = np.eye(4)
    ScaleMatrix[0, 0] = sx  # scale on x
    ScaleMatrix[1, 1] = sy  # scale on y
    ScaleMatrix[2, 2] = sz  # scale on z

    return ScaleMatrix

def rotation_matrix(rx=0., ry=0., rz=0.):

    # input should be degree (e.g., 0, 90, 180)

    # degree to radians
    rx = rx * np.pi / 180.
    ry = ry * np.pi / 180.
    rz = rz * np.pi / 180.

    Rx = np.eye(4)
    Rx[1, 1] = np.cos(rx)
    Rx[1, 2] = -np.sin(rx)
    Rx[2, 1] = np.sin(rx)
    Rx[2, 2] = np.cos(rx)

    Ry = np.eye(4)
    Ry[0, 0] = np.cos(ry)
    Ry[0, 2] = np.sin(ry)
    Ry[2, 0] = -np.sin(ry)
    Ry[2, 2] = np.cos(ry)

    Rz = np.eye(4)
    Rz[0, 0] = np.cos(rz)
    Rz[0, 1] = -np.sin(rz)
    Rz[1, 0] = np.sin(rz)
    Rz[1, 1] = np.cos(rz)

    # RZ * RY * RX
    RotationMatrix = np.mat(Rz) * np.mat(Ry) * np.mat(Rx)

    return np.array(RotationMatrix)

def translation_matrix(tx=0., ty=0., tz=0.):

    TranslationMatrix = np.eye(4)
    TranslationMatrix[0, -1] = tx
    TranslationMatrix[1, -1] = ty
    TranslationMatrix[2, -1] = tz

    return TranslationMatrix

def create_pose_matrix(tx=0., ty=0., tz=0.,
                       rx=0., ry=0., rz=0.,
                       sx=1.0, sy=1.0, sz=1.0,
                       base_correction=np.eye(4)):

    # Scale matrix
    ScaleMatrix = scale_matrix(sx, sy, sz)

    # Rotation matrix
    RotationMatrix = rotation_matrix(rx, ry, rz)

    # Translation matrix
    TranslationMatrix = translation_matrix(tx, ty, tz)

    # TranslationMatrix * RotationMatrix * ScaleMatrix
    PoseMatrix = np.mat(TranslationMatrix) \
                 * np.mat(RotationMatrix) \
                 * np.mat(ScaleMatrix) \
                 * np.mat(base_correction)

    return np.array(PoseMatrix)

