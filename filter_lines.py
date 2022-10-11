import json

import numpy as np
import pandas as pd
import glob

lon_leftup = 90.380707
lon_rightdown = 170.560404
lat_leftup = 60.20447
lat_rightdown = 0

keep_file_name = []
folder = 'data/train_dataset/train'
files = glob.glob(folder+'/*.csv')
for file in files:
    data = pd.read_csv(file)
    lats = data['lat']
    lons = data['lon']
    if np.any(lats < lat_leftup) and np.any(lats > lat_rightdown) \
    and np.any(lons < lon_rightdown) and np.any(lons > lon_leftup):
        if np.max(lons) - np.min(lons) > 10 or np.max(lons) - np.min(lons) > 10:
            keep_file_name.append(file)
json.dump(keep_file_name, open('keep_files.json', 'w'), indent=4)