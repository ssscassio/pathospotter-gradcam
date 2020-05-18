# -*- coding: utf-8 -*-
import cv2
import os
import glob
import numpy as np
from scipy import stats


path = '../cam_glomerulo/dataset'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
            files.append(os.path.join(r, file))
            
files = np.sort(files)

representacoes={}
height = np.array([])
width = np.array([])
channels = np.array([])


# Interate over images
#os.chdir(path)
for image_name in files:
    img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
    height = np.append(height, img.shape[0])
    width = np.append(width, img.shape[1])
    channels = np.append(channels, img.shape[2])
    

height_avg = np.average(height)
width_avg = np.average(width)

height_std = np.std(height)
width_std = np.std(width)

height_med = np.median(height)
width_med= np.median(width)

#height_var = np.var(height)
#width_var = np.var(width)

channels_avg = np.average(channels)

print("Height Avg: %f"%(height_avg))
print("Width Avg: %f"%(width_avg))
print("Height std: %f"%(height_std))
print("Width std: %f"%(width_std))
print("Height median: %f"%(height_med))
print("Width median: %f"%(width_med))
print("channel Avg: %f"%(channels_avg))