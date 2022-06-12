#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 09:33:04 2022

@author: klee
"""
import xml.etree.ElementTree as ET
import numpy as np
import os, cv2


iou_threshold=0.5
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    if xB - xA < 0 or yB - yA < 0:
        return 0
    interArea = (xB - xA + 1) * (yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def load_gt(annotation_path):  
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    bboxes = []
    for object in root.findall('object'):
        object_name = object.find('name').text
        bbox =object.find('bndbox')
        ymin = int(bbox.find('ymin').text)
        xmin = int(bbox.find('xmin').text)
        ymax = int(bbox.find('ymax').text)
        xmax = int(bbox.find('xmax').text)
        bboxes.append([ymin,xmin,ymax,xmax])
        print([ymin,xmin,ymax,xmax])
        labelindex=object.find('name').text
    if bboxes == []:
        return bboxes
        #assert 1<0
    return np.asarray(bboxes)

def match_gt_pred(arr_gt,arr_pred):
    TP_pred_bool = np.zeros(len(arr_pred))
    GT_bool = np.zeros(len(arr_gt))
    for ind_pred, pred in enumerate(arr_pred):        
        for ind_gt, gt in enumerate(arr_gt):
            iou =  bb_intersection_over_union(pred, gt)
            if iou > iou_threshold and GT_bool[ind_gt]==0:
                TP_pred_bool[ind_pred] = 1
                GT_bool[ind_gt] = 1                        
    return TP_pred_bool


groundtruth_xml='../golden.xml'
prediction_xml='../annotation.xml'

print ('===== GT =====')
Boxes_gt=load_gt(groundtruth_xml)
print ('===== PD =====')
Boxes_predict=load_gt(prediction_xml)

TP=match_gt_pred(Boxes_gt,Boxes_predict)
precision = np.sum(TP)/len(TP)
recall = np.sum(TP)/len(Boxes_gt)
print('precision', precision)
print('recall', recall)



