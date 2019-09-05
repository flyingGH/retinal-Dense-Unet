# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:48:13 2019

@author: kasy
"""

import glob
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from configs.utils.config_utils import process_config
import os

config = process_config('configs/segmention_config.json')

DATASET_NAME = config.dataset
PRE_NAME = config.preprocess
MODEL_NAME = config.exp_name

#DATASET_NAME = 'DRIVE'
#PRE_NAME = 'DUNet'

#MODEL_NAME = 'VesselNet'

print('Process ', PRE_NAME, ';Dataset ', DATASET_NAME)
#train_dir = './data/{0}/{1}/train/'.format(PRE_NAME, DATASET_NAME)

test_gt_path = './experiments/{0}/test/groundtruth/'.format(MODEL_NAME)
test_prob_path = './experiments/{0}/test/result/{0}/{1}/{2}/result/'.format(MODEL_NAME, PRE_NAME, DATASET_NAME)
# 0_255, gray image
test_gt_list = []
test_prob_list = []


for i in os.listdir(test_gt_path):
	test_gt_list.append(os.path.join(test_gt_path, i))

for i in os.listdir(test_prob_path):
    #print('---',i)
    test_prob_list.append(os.path.join(test_prob_path, i))

#print(test_gt_list)
#print('**', test_prob_list)

#assert 0
gt_img_list = []
prob_img_list = []

for index in range(len(test_gt_list)):
	gt_img = cv2.imread(test_gt_list[index])[..., 0]
	prob_img = cv2.imread(test_prob_list[index])[..., 0]

	gt_img_list.append(gt_img)
	prob_img_list.append(prob_img)

TP = []
FP = []
TN = []
FN = []

for threshold in tqdm(range(0, 255)):
	temp_TP = 0.0
	temp_FP = 0.0
	temp_TN = 0.0
	temp_FN = 0.0
	for index in range(len(test_gt_list)):
		prob_img = prob_img_list[index]
		gt_img = gt_img_list[index]
		#print(prob_img.shape)
		gt_img=(gt_img>0)*1
		prob_img=(prob_img>=threshold)*1
    
		temp_TP = temp_TP + (np.sum(prob_img * gt_img))
		temp_FP = temp_FP + np.sum(prob_img * ((1 - gt_img)))
		temp_FN = temp_FN + np.sum(((1 - prob_img)) * ((gt_img)))
		temp_TN = temp_TN + np.sum(((1 - prob_img)) * (1 - gt_img))
	TP.append(temp_TP)
	FP.append(temp_FP)
	TN.append(temp_TN)
	FN.append(temp_FN)
    
TP = np.asarray(TP).astype('float32')
FP = np.asarray(FP).astype('float32')
FN = np.asarray(FN).astype('float32')
TN = np.asarray(TN).astype('float32')

FPR = (FP) / (FP + TN)
TPR = (TP) / (TP + FN)
AUC = np.round(np.sum((TPR[1:] + TPR[:-1]) * (FPR[:-1] - FPR[1:])) / 2., 4)

Precision = (TP) / (TP + FP)
Recall = TP / (TP + FN)
MAP = np.round(np.sum((Precision[1:] + Precision[:-1]) * (Recall[:-1] - Recall[1:])) / 2.,4)

#IOU
intersection=0.0
union=0.0
threshold = 128
accuracy_list = []
for index in range(len(test_gt_list)):
    gt_img = gt_img_list[index]
    prob_img = prob_img_list[index]
    gt_img = (gt_img > 0) * 1
    prob_img = (prob_img >= threshold) * 1
    w, h = prob_img.shape[0], prob_img.shape[1]
    accuracy = (np.sum(prob_img * gt_img) + np.sum((1-prob_img)*(1-gt_img))) /(w*h)
    accuracy_list.append(accuracy)
    intersection=intersection+np.sum(gt_img*prob_img)
    union=union+np.sum(gt_img)+np.sum(prob_img)-np.sum(gt_img*prob_img)
iou=np.round(intersection/union,4)

pre_median = Precision[128]
recal_median = Recall[128]
beta_square = 0.3

F_beta = ((1+beta_square)*pre_median*recal_median)/(beta_square*pre_median+recal_median)

print('AUC {0:.4f}, Acc {2:.4f}, F-beta {1:.4f}'.format(AUC, F_beta, np.mean(accuracy_list)))
