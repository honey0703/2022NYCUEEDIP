#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import os, cv2
# from tqdm import tqdm

annota_dir = '/Users/zhaoyuhan/Documents/OneDrive - 清華大學/碩一下/Image Processing/Final Project/PCB_DATASET/p_Annotations/Spurious_copper/'
origin_dir = '/Users/zhaoyuhan/Documents/OneDrive - 清華大學/碩一下/Image Processing/Final Project/PCB_DATASET/p_images/Spurious_copper/'
target_dir1= '/Users/zhaoyuhan/Documents/OneDrive - 清華大學/碩一下/Image Processing/Final Project/PCB_DATASET/p_images_anno/Spurious_copper/'

def divide_img(oriname):
    img_file = os.path.join(origin_dir, oriname + '.jpg')
    im = cv2.imread(img_file)

    xml_file = os.path.join(annota_dir, oriname + '.xml') 
    tree = ET.parse(xml_file)
    root = tree.getroot()
#im = cv2.imread(imgfile)
    for object in root.findall('object'):
        object_name = object.find('name').text
        Xmin = int(object.find('bndbox').find('xmin').text)
        Ymin = int(object.find('bndbox').find('ymin').text)
        Xmax = int(object.find('bndbox').find('xmax').text)
        Ymax = int(object.find('bndbox').find('ymax').text)
        color = (0, 0, 255)
        cv2.rectangle(im, (Xmin, Ymin), (Xmax, Ymax), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, object_name, (Xmin, Ymin - 7), font, 0.5, (0, 0, 255), 2)
        cv2.imshow('01', im)
        
    img_name = oriname + '.jpg'
    to_name = os.path.join(target_dir1, img_name)
    cv2.imwrite(to_name, im)

img_list = os.listdir(origin_dir)
for name in img_list:
    divide_img(name.rstrip('.jpg'))

