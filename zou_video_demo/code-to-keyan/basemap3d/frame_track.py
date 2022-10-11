import numpy as np
import cv2

def estimate_transform(matched_keypoints):
    """Wrapper of cv2.estimateRigidTransform for convenience in vidstab process
    :param matched_keypoints: output of match_keypoints util function; tuple of (cur_matched_kp, prev_matched_kp)
    :return: transform as list of [dx, dy, da]
    """
    prev_matched_kp, cur_matched_kp = matched_keypoints

    transform = cv2.estimateAffine2D(np.array(prev_matched_kp),
                                            np.array(cur_matched_kp))[0]

    return transform


def frame_tracking(frame, frame_ref):

    if len(frame.shape) == 3:
        frame = frame[:, :, 0]
    if len(frame_ref.shape) == 3:
        frame_ref = frame_ref[:, :, 0]

    # ShiTomasi corner detection
    ref_pts = cv2.goodFeaturesToTrack(
        frame_ref, maxCorners=3000,
        qualityLevel=0.01, minDistance=7, blockSize=51)

    if ref_pts is None:
        print('no feature point detected')
        return None

    # Calculate optical flow (i.e. track feature points)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
        frame_ref, frame, ref_pts, None)
    # Filter only valid points
    idx = np.where(status == 1)[0]
    if idx.size == 0:
        print('no good point matched')
        return None

    if curr_pts.shape[0] < 5:
        print('no good point matched')
        return None

    transform = estimate_transform((np.array(ref_pts), np.array(curr_pts)))

    return transform


def update_transformation_matrix(M, m):

    # extend M and m to 3x3 by adding an [0,0,1] to their 3rd row
    M_ = np.concatenate([M, np.zeros([1,3])], axis=0)
    M_[-1, -1] = 1
    m_ = np.concatenate([m, np.zeros([1,3])], axis=0)
    m_[-1, -1] = 1

    M_new = np.matmul(m_, M_)
    return M_new[0:2, :]


def create_identy_transform():
    return np.array([[1., - 0.,  0.], [0.,  1.,  0.]], dtype=np.float)
