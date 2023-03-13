import cv2
import numpy as np
import os
from joblib import Parallel, delayed
import multiprocessing

import pdb
import time
import sys
sys.path.insert(0, 'reader')
from reader import LFReader


# This function computes vote cost of all light field views for a possible refocused disparity
# LF: full light field image
# central : central view
# refo_dis : refocused disparity
def vote_cost(LF, central, refo_dis):
    #print(refo_dis)    
    V, U, H, W, C = LF.shape
    cost = np.zeros((H, W), dtype=dtype)
    for v in range(V):
        for u in range(U):
            target = LF[v, u]      
            translation_matrix = np.array([[1, 0, refo_dis * (u - U // 2)], [0, 1, refo_dis * (v - U // 2)]], dtype=dtype)  # 变化矩阵
            target_translation = cv2.warpAffine(target, translation_matrix, (W, H), flags=cv2.INTER_LINEAR)
            vote = np.abs(central - target_translation).mean(axis=2)
            vote[vote >= t] = 1.0
            cost += vote
    return cost

if __name__ == '__main__':
    dtype = np.float32
    
    dataset_name = 'UrbanLF_Real'  
    scene_number = 824
    #dataset_name = 'UrbanLF_Syn' 
    #scene_number = 250
    for index in range(1,scene_number+1):
        print("Image:",index," start")
        # load light field image
        if dataset_name == 'UrbanLF_Real' : 
            LF = LFReader.read_UrbanLF_Real(index)
        else:
            LF = LFReader.read_UrbanLF_Syn(index)
        V, U, H, W, C = LF.shape   
        central_uint8 = LF[V // 2, U // 2]  
        LF = LF.astype(dtype) / 255.0
        central = LF[V // 2, U // 2]
        
        # set dis range
        d_res = 256 # number of possible disparities, set this parameter to 201 for light field image with large disparity
        d_max = 1.5
        d_min = -0.5
        d_step = (d_max - d_min) / (d_res - 1)
        d_all =  d_min + np.arange(d_res, dtype=dtype) * d_step

        # Step 1  Computing adaptive threshold
        delta_e = 0.1 # sampling interval
        deviation = np.zeros((V, U, H, W), dtype=dtype)
        for v in range(V):
            for u in range(U):
                translation_matrix = np.array([[1, 0, delta_e * (u - U // 2)], [0, 1, delta_e * (v - U // 2)]], dtype=dtype)
                target_translation = cv2.warpAffine(central, translation_matrix, (W, H), flags=cv2.INTER_LINEAR) # Equation 12
                deviation[v, u] = np.abs(central - target_translation).mean(axis=2)

        t = deviation.mean(axis=(0, 1))
        t_max = 0.005 # set this parameter to 0.01 for light field image with large disparity
        t_min = 0.002
        t[t > t_max] = t_max # Equation 13 adaptive threshold truncation
        t[t < t_min] = t_min # Equation 13 adaptive threshold truncation

        # Step 2   Building vote cost volume
        start_time = time.time()
        num_cores = multiprocessing.cpu_count()
        cv = Parallel(n_jobs=num_cores)(delayed(vote_cost)(LF, central, refo_dis) for refo_dis in d_all)
        cv = np.array(cv)
        print('Building vote cost volume in {}s'.format(round(time.time() - start_time, 2)))
        cv = cv.transpose((1, 2, 0))
        pred = np.argmin(cv, axis=2)
        pred = pred.astype(dtype) / (d_res - 1)
        cv2.imwrite('depth_before_cost_volume_fiter.png', 255 - pred*255)

        # Step 3   Cost volume filtering by bilateral filter
        # This implementation could be improved as it re-computes filter weights for each cost slice
        start_time_bf = time.time()
        for i in range(d_res):
            cv[:, :, i] = cv2.ximgproc.jointBilateralFilter(central, cv[:, :, i], d=7, sigmaColor=0.1, sigmaSpace=3)
        print('Cost volume filter in ', time.time() - start_time_bf)

        pred = np.argmin(cv, axis=2)
        pred = pred.astype(dtype) / (d_res - 1)
        cv2.imwrite('depth_after_cost_volume_fiter.png', 255 - pred*255)

        # Step 4    Depth refinement by fast weighted median filter
        # Different wmf_r and wmf_sigma may result in slightly different depth accuracy
        start_time_wmf = time.time()
        wmf_r = 5
        wmf_sigma = 25.5
        pred = cv2.ximgproc.weightedMedianFilter(central_uint8, pred, r=wmf_r, sigma=wmf_sigma)
        # Second-time filtering is optional. It can slightly improve depth accuracy
        pred = cv2.ximgproc.weightedMedianFilter(central_uint8, pred, r=3, sigma=25.5)
        print('Weighted median filter in {}s'.format(round(time.time() - start_time_wmf, 2)))

        pred = (d_max-d_min) * pred + d_min
        print('Total time {}s'.format(round(time.time() - start_time, 2)))

        # 保存最终的视差
        if dataset_name == 'UrbanLF_Real' : 
            np.save('./data/UrbanLF/UrbanLF_Real/Image'+str(index)+'/disparity_OAVC.npy',pred)
        else:
            np.save('./data/UrbanLF/UrbanLF_Syn/Image'+str(index)+'/5_5_disparity_OAVC.npy',pred)
