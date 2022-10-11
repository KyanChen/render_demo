import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import matplotlib.patches as mpathes
import pickle

def draw_rectangle(m,xmin,ymin,xmax,ymax,ax,name=None):
    # fig, ax = plt.subplots()
    xmin,ymin=m(xmin,ymin)
    xmax,ymax=m(xmax,ymax)
    height=ymax-ymin
    width=xmax-xmin
    rect = mpathes.Rectangle((xmin,ymin),width,height,fill=False,color='k',linewidth=2)
    ax.add_patch(rect)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=128):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name,a=minval, b=maxval),cmap(np.linspace(minval, maxval, n)))
    return new_cmap

topo_file='I:\dataset\deep-downscale\DEM\elevation/ETOPO2v2g_f4.nc'
data=nc.Dataset(topo_file)
topo=data.variables['z'][:,:]
lat=data.variables['y'][:]
lon=data.variables['x'][:]

lon_leftup=70;lat_leftup=52
lon_rightdown=140;lat_rightdown=0 #建立地图投影

lon_leftup=103.16662;lat_leftup=37.34194
lon_rightdown=120.83338;lat_rightdown=22.072418 #建立地图投影

llcrnrlon=103.16662
urcrnrlon=120.83338
llcrnrlat=22.072418
urcrnrlat=37.34194

fig, ax = plt.subplots() #建立绘图平台



m = Basemap(projection='merc', llcrnrlat=lat_rightdown, urcrnrlat=lat_leftup, llcrnrlon=lon_leftup,
            urcrnrlon=lon_rightdown, resolution='i')
m.drawcoastlines(linewidth=0.2, color='gray',zorder=3)

parallels = np.arange(np.ceil(lat_rightdown),int(lat_leftup)+3,3) #纬线
m.drawparallels(parallels,labels=[True,False,False,False],linewidth=0.5,dashes=[3,6])
plt.yticks(parallels,len(parallels)*[''])
meridians = np.arange(np.ceil(lon_leftup),int(lon_rightdown)+3,3) #经线
m.drawmeridians(meridians,labels=[False,False,False,True],linewidth=0.5,dashes=[3,6])
plt.xticks(meridians,len(meridians)*[''])
m.drawcoastlines(linewidth=0.25)
m.drawstates(linewidth=0.25)
m.drawcountries(linewidth=0.25)
m.readshapefile(r'I:\dataset\deep-downscale\auxi_data/gadm36_CHN_1','states',drawbounds=True)



# draw_rectangle(m,llcrnrlon,llcrnrlat,urcrnrlon,urcrnrlat,ax)

station_info_file=r'I:\dataset\deep-downscale\IDD\obs_station_infos.pickle'
with open(station_info_file,'rb') as fp:
    station_infos=pickle.load(fp)
lon_list=[station_infos[key][1] for key in station_infos.keys()]
lat_list=[station_infos[key][2] for key in station_infos.keys()]

# plt.show()

lons, lats = np.meshgrid(lon,lat) #经纬度2维化
x, y = m(lons, lats) #投影映射

cmap_new = truncate_colormap(plt.cm.terrain, 0.23, 1.0) #截取colormap，要绿色以上的（>=0.23）
cmap_new.set_under([198/255,234/255,250/255]) #低于0的填色为海蓝
lev=np.arange(0,6000,200)
norm3 = mpl.colors.BoundaryNorm(lev, cmap_new.N) #标准化level，映射色标
cf=m.contourf(x,y,topo,levels=lev,norm=norm3
              ,cmap=cmap_new,extend='both')

cb=plt.colorbar(cf, ax=ax,shrink=0.7,aspect=30,pad=0.05,orientation='horizontal') #色标
cb.ax.tick_params(labelsize=10,pad=2,direction='in') #色标tick

x,y=m(lon_list,lat_list)
plt.scatter(x,y,marker='o',color='red',edgecolors='k')
plt.show()
# plt.savefig('global_etopo.png',dpi=300,bbox_inches='tight') #存图