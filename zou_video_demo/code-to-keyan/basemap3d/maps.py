import cv2
import numpy as np

MAP_SIZE = 192
FIELD_SIZE = 80
GRID_SIZE = 5

def _wd2pxl(y_wd, x_wd):
    y_pxl = int(y_wd / FIELD_SIZE * MAP_SIZE + MAP_SIZE / 2.)
    x_pxl = int(x_wd / FIELD_SIZE * MAP_SIZE + MAP_SIZE / 2.)
    return y_pxl, x_pxl


def _wd2pxl_dist(dist_wd):
    dist_pxl = int(dist_wd / FIELD_SIZE * MAP_SIZE)
    return dist_pxl


def _draw_grids(map_layer):

    wd_y_min, wd_y_max = -FIELD_SIZE / 2., FIELD_SIZE/2.
    wd_x_min, wd_x_max = -FIELD_SIZE / 2., FIELD_SIZE / 2.

    # draw horizontal lines
    for y_wd in range(int(wd_y_min), int(wd_y_max), GRID_SIZE):
        y_pxl_1, x_pxl_1 = _wd2pxl(y_wd=y_wd, x_wd=wd_x_min)
        y_pxl_2, x_pxl_2 = _wd2pxl(y_wd=y_wd, x_wd=wd_x_max)
        map_layer = cv2.line(map_layer, pt1=(x_pxl_1, y_pxl_1), pt2=(x_pxl_2, y_pxl_2),
                             color=(200, 200, 200), thickness=1,
                             lineType=cv2.LINE_AA)

    # draw vertical lines
    for x_wd in range(int(wd_x_min), int(wd_x_max), GRID_SIZE):
        y_pxl_1, x_pxl_1 = _wd2pxl(y_wd=wd_y_min, x_wd=x_wd)
        y_pxl_2, x_pxl_2 = _wd2pxl(y_wd=wd_y_max, x_wd=x_wd)
        map_layer = cv2.line(map_layer, pt1=(x_pxl_1, y_pxl_1), pt2=(x_pxl_2, y_pxl_2),
                             color=(200, 200, 200), thickness=1,
                             lineType=cv2.LINE_AA)

    return map_layer


def _draw_center_target(map_layer):
    r_wd = 2
    r_pxl = _wd2pxl_dist(r_wd)
    yc_pxl, xc_pxl = _wd2pxl(y_wd=0, x_wd=0)
    map_layer = cv2.circle(map_layer, center=(xc_pxl, yc_pxl), radius=r_pxl,
                           color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
    return map_layer


def _rotate_pts(pts, a, xc, yc):
    M = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    pts_ = np.array(pts) - np.array([[xc, yc]])
    pts_rotate = np.matmul(M, pts_.T).T + np.array([[xc, yc]])
    pts_rotate = np.array(pts_rotate, dtype=np.int)
    return pts_rotate


def _draw_box(map_layer, yc_pxl, xc_pxl, w_pxl, l_pxl, heading):

    ext_y1, ext_x1 = int(l_pxl / 2.), int(w_pxl / 2.)
    ext_y2, ext_x2 = int(l_pxl / 2.), -int(w_pxl / 2.)
    ext_y3, ext_x3 = -int(l_pxl / 2.), -int(w_pxl / 2.)
    ext_y4, ext_x4 = -int(l_pxl / 2.), int(w_pxl / 2.)
    pts = [[xc_pxl + ext_x1, yc_pxl + ext_y1],
           [xc_pxl + ext_x2, yc_pxl + ext_y2],
           [xc_pxl + ext_x3, yc_pxl + ext_y3],
           [xc_pxl + ext_x4, yc_pxl + ext_y4]]
    pts_rotate = _rotate_pts(pts, a=heading/180.*np.pi, xc=xc_pxl, yc=yc_pxl)
    cv2.fillPoly(map_layer, [np.array(pts_rotate)], color=(255, 255, 255), lineType=cv2.LINE_AA)

    return map_layer


def _draw_text(map_layer, yc_pxl, xc_pxl, vid):
    text = 'v'+str(vid)
    tsize, _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                             fontScale=0.5, thickness=1)
    dx, dy = int(tsize[0]/2.0), int(tsize[1]/2.0)
    pt = (xc_pxl+dx, yc_pxl+dy)
    cv2.putText(map_layer, text=text, org=pt, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    return map_layer


def _draw_vehicles(map_layer, vehicle_list):

    for i in range(len(vehicle_list)):
        v = vehicle_list[i]
        xc, yc, w, l, heading = v['x'], v['z'], v['w'], v['l'], v['heading']
        yc_pxl, xc_pxl = _wd2pxl(y_wd=yc, x_wd=xc)
        w_pxl, l_pxl = _wd2pxl_dist(w), _wd2pxl_dist(l)
        map_layer = _draw_box(map_layer, yc_pxl=yc_pxl, xc_pxl=xc_pxl,
                              w_pxl=w_pxl, l_pxl=l_pxl, heading=heading)
        map_layer = _draw_text(map_layer, yc_pxl=yc_pxl, xc_pxl=xc_pxl, vid=i)

    return map_layer


def render_maps(vehicle_list):

    map_layer = np.ones([MAP_SIZE, MAP_SIZE, 3]) * 128.
    map_layer = map_layer.astype(np.uint8)

    map_layer = _draw_grids(map_layer=map_layer)
    # map_layer = _draw_center_target(map_layer=map_layer)
    map_layer = _draw_vehicles(map_layer=map_layer, vehicle_list=vehicle_list)

    return map_layer


def blend_maps(img, map_layer, loc='tl'):

    img = img.astype(np.float32) / 255.
    map_layer = map_layer.astype(np.float32) / 255.

    if map_layer.shape[0] != MAP_SIZE or map_layer.shape[1] != MAP_SIZE:
        map_layer = cv2.resize(map_layer, [MAP_SIZE, MAP_SIZE])

    img_h, img_w, c = img.shape

    if loc == 'tl':
        y1, y2 = 0, MAP_SIZE
        x1, x2 = 0, MAP_SIZE
    elif loc == 'tr':
        y1, y2 = 0, MAP_SIZE
        x1, x2 = img_w - MAP_SIZE, img_w
    else:
        raise NotImplementedError(
            'Wrong loc %s (choose one from [tr, tl])'
            % loc)

    base_layer = img[y1:y2, x1:x2, :]

    alpha = 0.75
    mixed_layer = alpha*map_layer + (1-alpha)*base_layer
    img[y1:y2, x1:x2, :] = mixed_layer

    img = (img*255.).astype(np.uint8)

    return img



