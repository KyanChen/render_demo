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


# def draw_line(csv_file, m):
#     os.makedirs('lines', exist_ok=True)
#     # folder = 'data/train_dataset/train'
#     folder = ''
#     data = pd.read_csv(folder+csv_file)
#     lats = data['lat']
#     lons = data['lon']
#     times = data['time']
#     x, y = m(lons, lats)
#     for idx in range(1, len(x), 10):
#         x_tmp = x[:idx]
#         y_tmp = y[:idx]
#         m.plot(x_tmp, y_tmp, marker=None, color='m')
#         plt.savefig(f'lines/line_{idx}.png')

def draw_line(data, m, videoWriter):
    lons = data[:, 0]
    lats = data[:, 1]
    max_time = 30
    for i in range(1, len(lons)):
        lon_start = lons[i-1]
        lat_start = lats[i - 1]
        lon_end = lons[i]
        lat_end = lats[i]
        lon_before = lon_start
        lat_before = lat_start
        for j in range(max_time):
            lon_tmp = lon_start + j * (lon_end - lon_start) / max_time
            lat_tmp = lat_start + j * (lat_end - lat_start) / max_time
            m.plot([lon_before, lon_tmp], [lat_before, lat_tmp], marker=None, linewidth=2, color='r', latlon=True)
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
fps = 30  # 视频帧率
fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
video_path = 'test_video1.mp4'
videoWriter = cv2.VideoWriter(video_path, fourcc, fps, (2200, 1600))

fig, ax = plt.subplots(figsize=(11, 8), dpi=200)  # 建立绘图平台
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
# lines = json.load(open('keep_files.json'))
draw_line(line, m, videoWriter)
videoWriter.release()
# plt.show()

# map = Basemap(llcrnrlon=3.75, llcrnrlat=39.75, urcrnrlon=4.35, urcrnrlat=40.15, epsg=5520)
# map.arcgisimage(
#     # server='http://server.arcgisonline.com/arcgis/rest/services',
#     service='ESRI_Imagery_World_2D',
#     xpixels=1500, verbose=True
# )
# # http://server.arcgisonline.com/arcgis/rest/services
# plt.show()

# m.drawcoastlines(linewidth=0.2, color='gray', zorder=3)
# m.drawstates(linewidth=0.25)
# m.drawcountries(linewidth=0.25)
# # plt.show()
# cmap_new = truncate_colormap(plt.cm.terrain, 0.23, 1.0)  # 截取colormap，要绿色以上的（>=0.23）
# cmap_new.set_under([198/255, 234/255, 250/255])  # 低于0的填色为海蓝
# lev = np.arange(0, 6000, 200)
# norm3 = mpl.colors.BoundaryNorm(lev, cmap_new.N) # 标准化level，映射色标
# cf = m.contourf(x,y,topo,levels=lev,norm=norm3
#               ,cmap=cmap_new, extend='both')

# cb=plt.colorbar(cf, ax=ax,shrink=0.7,aspect=30,pad=0.05,orientation='horizontal') #色标
# cb.ax.tick_params(labelsize=10,pad=2,direction='in') #色标tick
#
# x,y=m(lon_list,lat_list)
# plt.scatter(x,y,marker='o',color='red',edgecolors='k')
# plt.show()
# plt.savefig('test.png')
