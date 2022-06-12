#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import os, cv2
from tqdm import tqdm




im = cv2.imread('test.jpg')
tree = ET.parse('test.xml')
root = tree.getroot()
#im = cv2.imread(imgfile)
for object in root.findall('object'):
    object_name = object.find('name').text
    Xmin = int(object.find('bndbox').find('xmin').text)
    Ymin = int(object.find('bndbox').find('ymin').text)
    Xmax = int(object.find('bndbox').find('xmax').text)
    Ymax = int(object.find('bndbox').find('ymax').text)
    color = (0, 250, 0)
    cv2.rectangle(im, (Xmin, Ymin), (Xmax, Ymax), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX   
    cv2.putText(im, object_name, (Xmin, Ymin - 7), font, 0.5, (6, 230, 230), 2)
    cv2.imshow('01', im)

tree = ET.parse('test_p.xml')
root = tree.getroot()
#im = cv2.imread(imgfile)
for object in root.findall('object'):
    object_name = object.find('name').text
    Xmin = int(object.find('bndbox').find('xmin').text)
    Ymin = int(object.find('bndbox').find('ymin').text)
    Xmax = int(object.find('bndbox').find('xmax').text)
    Ymax = int(object.find('bndbox').find('ymax').text)
    color = (0, 0, 250)
    cv2.rectangle(im, (Xmin, Ymin), (Xmax, Ymax), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX   
    cv2.putText(im, object_name, (Xmin, Ymin - 7), font, 0.5, (6, 230, 230), 2)
    cv2.imshow('01', im)

    

cv2.imwrite('test_boxes.jpg',im)