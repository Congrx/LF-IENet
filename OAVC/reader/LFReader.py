import file_io
import h5py
import numpy as np
import os
import glob
import cv2
import pdb
import imageio


def read_UrbanLF_Syn(scene_index):
    image_root_path = './data/UrbanLF/UrbanLF_Syn/'
    img_list = image_root_path + 'Image' + str(scene_index) + '/'
    LF = np.zeros((9, 9, 480, 640, 3), dtype=np.uint8)
    for i in range(9):
        for j in range(9):
            img_name = str(i+1) + '_' + str(j+1)
            LF[i, j] = cv2.imread(img_list+img_name+'.png')

    return LF
    
def read_UrbanLF_Real(scene_index):
    image_root_path = './data/UrbanLF/UrbanLF_Real/'
    img_list = image_root_path + 'Image' + str(scene_index) + '/'
    LF = np.zeros((9, 9, 432, 623, 3), dtype=np.uint8)
    for i in range(9):
        for j in range(9):
            img_name = str(i+1) + '_' + str(j+1)
            LF[i, j] = cv2.imread(img_list+img_name+'.png')

    return LF