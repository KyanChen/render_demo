from basemap3d import BasemapOffscreen3D, blend_images, utils, harmonization, maps, transformation
from basemap3d.frame_track import frame_tracking, update_transformation_matrix, create_identy_transform
import detection.detector as det
from skyar.synfilesmoke import FireSmoke
from skyar.synrain import Rain
import numpy as np
import cv2, os, glob

if __name__ == '__main__':

    ################ configs ################
    bg_img_folder = r'../../tmp/01'
    remove_color = True
    weather_control = False
    img_height, img_width = 512, 640
    dist, height = 100, 120
    yfov = 16.0 / 180 * np.pi
    ambient_light = 0.4
    vtype = 'tank'
    vehicle_manager = utils.VehicleManager(n=5, zmin=-20, zmax=0, xmin=-10, xmax=20, vtype=vtype, moving_target=True)

    # background images
    img_dirs = glob.glob(os.path.join(bg_img_folder, "*.bmp"))

    # initialize the renderer
    cam_pose = transformation.create_pose_matrix(
        tz=dist, ty=height, tx=0, rz=0, ry=0, rx= - np.arctan(height / dist) * 180 / np.pi)
    target_renderer = BasemapOffscreen3D(
        cam_pose=None, yfov=yfov, viewport_h=img_height, viewport_w=img_width, ambient_light=ambient_light)
    target_renderer.update_point_light(new_intensity=0)
    target_renderer.update_camera_pose(new_pose=cam_pose)

    # initialize frame-by-frame transformation matrix
    transform = create_identy_transform()

    # initialize object detector
    detector = det.ObjDet()

    # initialize firesmoke renderer
    firesmoke_renderer = FireSmoke(img_h=img_height, img_w=img_width)

    # intitialize weather controller
    weather_renderer = Rain(haze_intensity=6.0, rain_intensity=8.0, with_windhield_layer=False)

    # initialize other variables
    img_bg = cv2.resize(cv2.imread(img_dirs[0], cv2.IMREAD_COLOR), (img_width, img_height))
    img_bg_prev = img_bg.copy()
    img_buff_input, img_buff_output = [], []

    for idx in range(len(img_dirs)):

        # read a background image and do frame-by-frame tracking
        img_bg = cv2.resize(cv2.imread(img_dirs[idx], cv2.IMREAD_COLOR), (img_width, img_height))
        d_transform = frame_tracking(frame=img_bg, frame_ref=img_bg_prev)
        transform = update_transformation_matrix(M=transform, m=d_transform)
        print('processing frame id: %d' % idx)

        if d_transform is not None:

            # update vehicles
            vehicle_list = vehicle_manager.step()

            ######################## target render ########################
            # render a frame
            target_fg, target_mask, _ = target_renderer.render(vehicle_list)
            # warp the rendered images based on frame-by-frame tracking
            target_fg_warp = cv2.warpAffine(target_fg, transform, (img_width, img_height))
            target_mask_warp = cv2.warpAffine(target_mask, transform, (img_width, img_height))
            target_fg_warp, target_alpha_warp = harmonization(img_fg=target_fg_warp, mask=target_mask_warp, img_bg=img_bg)
            # blend rendering with background
            img_blend = blend_images(img_bg=img_bg, img_fg=target_fg_warp, alpha=target_alpha_warp)

            ######################## other effect render ########################
            # render a frame
            _, location_mask, _ = target_renderer.render([vehicle_list[-1]]) # the last vehicle is on fire, render its mask
            color, alpha = firesmoke_renderer.reder(location_mask)
            # warp the rendered images based on frame-by-frame tracking
            color_warp = cv2.warpAffine(color, transform, (img_width, img_height), borderMode=cv2.BORDER_REPLICATE)
            alpha_warp = cv2.warpAffine(alpha, transform, (img_width, img_height), borderMode=cv2.BORDER_REPLICATE)
            # blend rendering with background
            # img_blend = blend_images(img_bg=img_blend, img_fg=color_warp, alpha=alpha_warp)

            ######################### weather control ###########################
            if weather_control:
                img_blend = weather_renderer.forward(img_blend)

            if remove_color:
                grey = cv2.cvtColor(img_blend, cv2.COLOR_BGR2GRAY)
                img_blend = np.stack([grey, grey, grey], axis=-1)

            # target detection and visualize
            img_blend, bbs = detector.detect(img_blend, mask=target_mask_warp)

            # add maps
            vis_map = maps.render_maps(vehicle_list)
            img_blend = maps.blend_maps(img=img_blend, map_layer=vis_map)

            img_bg_prev = img_bg.copy()
            img_buff_output.append(img_blend)
            img_buff_input.append(img_bg)

            cv2.imshow('render', img_blend)
            cv2.waitKey(100)

    # save videos
    video_writer_1 = cv2.VideoWriter('./demo_input.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15.0, (img_width, img_height))
    video_writer_2 = cv2.VideoWriter('./demo_output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15.0, (img_width, img_height))
    for i in range(len(img_buff_output)):
        video_writer_1.write(img_buff_input[i])
        video_writer_2.write(img_buff_output[i])
        print('writing videos, frame id %d / %d' % (i, len(img_buff_output)))
