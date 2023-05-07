#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:43:21 2023

@author: ravi
"""
import numpy as np
import pylab

runfile('/home/ravi/Desktop/fac-ppg/inference_saliency_predictor_EG_preTrained.py', wdir='/home/ravi/Desktop/fac-ppg')
idx = np.where(saliency_diff>0)[0]
sd_a = (rate_array[idx, index] - pred_array[idx, index]) / pred_array[idx, index]
nui_h = (rate_array[idx, 2] - pred_array[idx, 2]) / pred_array[idx, 2]
nui_s = (rate_array[idx, 3] - pred_array[idx, 3]) / pred_array[idx, 3]
nui_f = (rate_array[:, 4] - pred_array[:, 4]) / pred_array[:, 4]
pylab.boxplot([sd_a, nui_h, nui_s, nui_f], labels=["Angry", "Happy", "Sad", "Fear"], sym="")


#%%

runfile('/home/ravi/Desktop/fac-ppg/inference_saliency_predictor_EG_preTrained.py', wdir='/home/ravi/Desktop/fac-ppg')
idx = np.where(saliency_diff>0)[0]
sd_h = (rate_array[idx, index] - pred_array[idx, index]) / pred_array[idx, index]
nui_a = (rate_array[idx, 1] - pred_array[idx, 1]) / pred_array[idx, 1]
nui_s = (rate_array[idx, 3] - pred_array[idx, 3]) / pred_array[idx, 3]
nui_f = (rate_array[:, 4] - pred_array[:, 4]) / pred_array[:, 4]
pylab.boxplot([nui_a, sd_h, nui_s, nui_f], labels=["Angry", "Happy", "Sad", "Fear"], sym="")


#%%

runfile('/home/ravi/Desktop/fac-ppg/inference_saliency_predictor_EG_preTrained.py', wdir='/home/ravi/Desktop/fac-ppg')
idx = np.where(saliency_diff>0)[0]
sd_s = (rate_array[idx, index] - pred_array[idx, index]) / pred_array[idx, index]
nui_a = (rate_array[idx, 1] - pred_array[idx, 1]) / pred_array[idx, 1]
nui_h = (rate_array[idx, 2] - pred_array[idx, 2]) / pred_array[idx, 2]
nui_f = (rate_array[:, 4] - pred_array[:, 4]) / pred_array[:, 4]
pylab.boxplot([nui_a, nui_h, sd_s, nui_f], labels=["Angry", "Happy", "Sad", "Fear"], sym="")


#%%

runfile('/home/ravi/Desktop/fac-ppg/inference_saliency_predictor_EG_preTrained.py', wdir='/home/ravi/Desktop/fac-ppg')
idx = np.where(saliency_diff>0)[0]
nui_a = (rate_array[idx, 1] - pred_array[idx, 1]) / pred_array[idx, 1]
nui_h = (rate_array[idx, 2] - pred_array[idx, 2]) / pred_array[idx, 2]
nui_s = (rate_array[idx, 3] - pred_array[idx, 3]) / pred_array[idx, 3]
sd_f = (rate_array[idx, index] - pred_array[idx, index]) / pred_array[idx, index]
pylab.boxplot([nui_a, nui_h, nui_h, sd_f], labels=["Angry", "Happy", "Sad", "Fear"], sym="")