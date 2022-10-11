import os.path
import glob
import cv2
import numpy as np
from skyar import synrain


if __name__ == '__main__':

    bg_img_folder = r'../../tmp/02'
    img_dirs = glob.glob(os.path.join(bg_img_folder, "*.bmp"))

    wc = synrain.Rain(haze_intensity=4.0, rain_intensity=2.0)
    haze_intensity = np.linspace(2.0, 1.0, len(img_dirs))
    rain_intensity = np.linspace(2.0, 1.0, len(img_dirs))

    img_buff_input = []
    img_buff_output = []
    for idx in range(len(img_dirs)):
        img_bg = cv2.imread(img_dirs[idx], cv2.IMREAD_GRAYSCALE)

        wc.haze_intensity = haze_intensity[idx]
        wc.rain_intensity = rain_intensity[idx]
        img_blend = wc.forward(img_bg)

        img_buff_input.append(np.stack([img_bg, img_bg, img_bg], axis=-1))
        img_buff_output.append(img_blend)

        cv2.namedWindow('render', cv2.WINDOW_NORMAL)
        cv2.imshow('render', img_blend)
        cv2.waitKey(1)

    # save video
    video_writer_1 = cv2.VideoWriter('./demo_input.mp4', cv2.VideoWriter_fourcc(*'MP4V'),
                                     15.0, (720, 480))
    video_writer_2 = cv2.VideoWriter('./demo_output.mp4', cv2.VideoWriter_fourcc(*'MP4V'),
                                   15.0, (720, 480))
    for i in range(len(img_buff_input)):
        img_in = cv2.resize(img_buff_input[i], (720, 480))
        img_out = cv2.resize(img_buff_output[i], (720, 480))
        video_writer_1.write(img_in)
        video_writer_2.write(img_out)
        print('writing video, frame id %d / %d' % (i, len(img_buff_input)))

