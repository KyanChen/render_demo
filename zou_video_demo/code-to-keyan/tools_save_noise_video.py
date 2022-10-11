import cv2
import numpy as np
from basemap3d import maps

img_height, img_width = 512, 640

video_writer = cv2.VideoWriter('./demo_output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15.0, (img_width, img_height))

for i in range(200):
    img = np.random.rand(int(img_height/2), int(img_width/2))
    img = (img * 255.0).astype(np.uint8)
    img = cv2.resize(img, [img_width, img_height])
    img = np.stack([img, img, img], axis=-1)

    # add maps
    vis_map = maps.render_maps([])
    img = maps.blend_maps(img=img, map_layer=vis_map)

    video_writer.write(img)
    print(i)
