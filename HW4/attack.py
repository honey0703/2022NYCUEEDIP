#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 10:05:00 2022

@author: kuei6
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
# from skimage.metrics import structural_similarity as ssim

def show_img(img):
    plt.figure(figsize=(15,15))
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.show()


img = cv2.imread('output/embedded_lena.bmp')
#img_ = cv2.imread('128.png')
show_img(img)  #original image

# level 1 attack: noise

def gaussian_noise(img, mean=0, sigma=0.1):

    # int -> float (標準化)
    img = img / 255
    # 隨機生成高斯 noise (float + float)
    noise = np.random.normal(mean, sigma, img.shape)
    # noise + 原圖
    gaussian_out = img + noise
    # 所有值必須介於 0~1 之間，超過1 = 1，小於0 = 0
    gaussian_out = np.clip(gaussian_out, 0, 1)

    # 原圖: float -> int (0~1 -> 0~255)
    gaussian_out = np.uint8(gaussian_out*255)
    # noise: float -> int (0~1 -> 0~255)
    noise = np.uint8(noise*255)
    return gaussian_out


img_noise = gaussian_noise(img)
show_img(img_noise)  #noised image
cv2.imwrite('attack/noise.png', img_noise)

# level 2 attack: blurring
img_blur = cv2.GaussianBlur(img, (7, 7), 2)
show_img(img_blur)  #blurred image
cv2.imwrite('attack/blur.png', img_blur)


# level 3 attack: compression
img_compress = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
show_img(img_compress) #compressed image
restore = cv2.resize(img_compress, (1024, 1024), interpolation=cv2.INTER_AREA)
cv2.imwrite('attack/compress.png', img_compress)
cv2.imwrite('attack/decompress.png', restore)


# img = cv2.imread('watermark.png')
# print (img)
# cv2.imwrite('watermark.bmp', img)