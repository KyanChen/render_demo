import json
import os

import cv2
import matplotlib
import pandas as pd
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.animation import FuncAnimation

def update(num):
    idx = num // 30
    lon_start = lons[idx-1, 0]
    lat_start = lats[idx-1, 1]
    lon_end = lons[idx, 0]
    lat_end = lats[idx, 1]
    idx_sub = num - 30*(num // 30)
    if idx_sub == 0:
        idx_sub = 1
    lon_tmp = lon_start + idx_sub * (lon_end - lon_start) / 30
    lat_tmp = lat_start + idx_sub * (lat_end - lat_start) / 30
    m.plot([lon_tmp-(lon_end - lon_start) / 30, lon_tmp], [lat_tmp-(lat_end - lat_start) / 30, lat_tmp], marker=None, color='m', latlon=True)
    canvas = FigureCanvasTkAgg(plt.gcf())
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    rgb_image = image[:, :, :3]
    r, g, b = cv2.split(rgb_image)
    img_bgr = cv2.merge([b, g, r])
    videoWriter.write(img_bgr)
        # plt.clf()
        # plt.close()

        lon_before = lon_tmp
        lat_before = lat_tmp


lon_leftup = 90.380707
lon_rightdown = 150.560404
lat_leftup = 40.20447
lat_rightdown = -5  # 建立地图投影
# lon_leftup = 90.380707
# lon_rightdown = 100.560404
# lat_leftup = 30
# lat_rightdown = 0  # 建立地图投影

fig, ax = plt.subplots(figsize=(11, 8), dpi=100)  # 建立绘图平台
m = Basemap(
    projection='merc',
    llcrnrlat=lat_rightdown,
    urcrnrlat=lat_leftup,
    llcrnrlon=lon_leftup,
    urcrnrlon=lon_rightdown,
    resolution='i'
)

m.bluemarble()
m.drawcoastlines(linewidth=0.5, color='gray')
m.drawcountries(linewidth=0.5, color='gray')
line = np.loadtxt('data1.txt', delimiter=',')
draw_line(line, m, videoWriter)

animation = FuncAnimation(fig, update, frames=len(line)*30, interval=10)
animation.save('rain.gif', fps=60, writer='imagemagick')
plt.show()
