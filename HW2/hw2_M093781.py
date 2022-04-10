#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 22:20:31 2022

@author: zhaoyuhan
"""

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

# Draw the grayscale histgram to observe what kind of method need to be used.
def hist(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()


# Median Filter
# Refernce: https://www.twblogs.net/a/5bb031222b7177781a0fd3e1)
def blur(img, factor): 
    image_blur = cv2.medianBlur(img, factor)
    return image_blur


# Use Laplancian filter + original image to implement sharpness
def sharp(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    return image_sharp


# Normalize the grayscale distribution for each r, g, b layer
def norm(img):
    b, g, r = cv2.split(img)
    b_out = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    cv2.normalize(b, b_out, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    g_out = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    cv2.normalize(g, g_out, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    r_out = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    cv2.normalize(r, r_out, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    img_out = cv2.merge([b_out, g_out, r_out])
    return img_out


# Modified their contrast 
# Refernce: https://www.wongwonggoods.com/python/python_opencv/opencv-modify-contrast/)
def contrast(img, brightness=20 , contrast=-25):
    B = brightness / 255.0
    c = contrast / 255.0 
    k = math.tan((45 + 44 * c) / 180 * math.pi)
    img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# Homomorphic Filter
# Reference: https://stackoverflow.com/questions/64284739/homomorphic-filtering-on-the-frequency-domain-python-opencv
def homomorphic(img, radius=5):
    # read input and convert to grayscale
    hh, ww = img.shape[:2]
    
    # take ln of image
    img_log = np.log1p(np.float64(img), dtype=np.float64) 
    
    # do dft saving as complex output
    dft = np.fft.fft2(img_log, axes=(0,1))
    
    # apply shift of origin to center of image
    dft_shift = np.fft.fftshift(dft)
    
    # create black circle on white background for high pass filter
    mask = np.zeros_like(img, dtype=np.float64)
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    cv2.circle(mask, (cx,cy), radius, 1, -1)
    mask = 1 - mask
    
    # antialias mask via blurring
    mask = cv2.GaussianBlur(mask, (47,47), 0)
    
    # apply mask to dft_shift
    dft_shift_filtered = np.multiply(dft_shift,mask)
    
    # shift origin from center to upper left corner
    back_ishift = np.fft.ifftshift(dft_shift_filtered)
    
    # do idft saving as complex
    img_back = np.fft.ifft2(back_ishift, axes=(0,1))
    
    # combine complex real and imaginary components to form (the magnitude for) the original image again
    img_back = np.abs(img_back)
    
    # apply exp to reverse the earlier log
    # img_homomorphic = np.exp(img_back, dtype=np.float64)
    
    # scale result
    img_homomorphic = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img_homomorphic


if __name__ == '__main__':
    # ----- input 1 ------------------------------------------------------------------
    img1 = cv2.imread('input1.bmp')
    hist_img = hist(img1)                             # Draw input grayscale histgram
    # --- Homomorphic                                 # Homomorphic
    b, g, r = cv2.split(img1)
    tmp1 = np.hstack((b, g, r))
    plt.imshow(tmp1, cmap='gray')
    plt.show()
    img_homo_b = homomorphic(b)
    img_homo_g = homomorphic(g)
    img_homo_r = homomorphic(r)
    img_homo = cv2.merge((img_homo_b, img_homo_g, img_homo_r))
    # --- End of Homomorphic
    image_contrast = contrast(img_homo, -10, 20)      # Contrast
    hist_o = hist(image_contrast)                     # Draw output grayscale histgram
    tmp = np.hstack((img1, image_contrast))           # Plot 2 images together
    plt.imshow(tmp)  
    plt.show()
    cv2.imwrite('output1.bmp', image_contrast)        # Save output image
    
    # ----- input 2 ------------------------------------------------------------------
    img2 = cv2.imread('input2.bmp')
    hist_img = hist(img2)                             # Draw inpput grayscale histgram
    image_blur = blur(img2, 7)                        # Blur
    image_sharp = sharp(image_blur)                   # Sharpness
    image_contrast = contrast(image_sharp, -60, 20)   # Contrast
    image_norm = norm(image_contrast)
    hist_img_o = hist(image_norm)                     # Draw output grayscale histgram
    tmp = np.hstack((img2, image_norm))               # Plot 2 images together
    plt.imshow(tmp)
    plt.show()
    cv2.imwrite('output2.bmp', image_norm)            # Save output image
    
    # ----- input 3 ------------------------------------------------------------------
    img3 = cv2.imread('input3.bmp')
    hist_img = hist(img3)                             # Draw input grayscale histgram
    # --- Homomorphic                                 # Homomorphic
    b, g, r = cv2.split(img3)
    tmp1 = np.hstack((b, g, r))
    img_homo_b = homomorphic(b)
    img_homo_g = homomorphic(g)
    img_homo_r = homomorphic(r)
    img_homo = cv2.merge((img_homo_b, img_homo_g, img_homo_r))
    # --- End of Homomorphic
    image_contrast = contrast(img_homo, 0, 15)        # Contrast
    hist_o = hist(image_contrast)                     # Draw output grayscale histgram
    tmp = np.hstack((img3, image_contrast))           # Plot 2 images together
    plt.imshow(tmp)  
    plt.show()
    cv2.imwrite('output3.bmp', image_contrast)        # Save output image
    
    # ----- input 4 --------------------------------------------------------------------
    img4 = cv2.imread('input4.bmp')
    hist_img = hist(img4)
    image_contrast = contrast(img4, 25, 60)            # Contrast
    image_sharp = sharp(image_contrast)                # Sharpness
    hist_o = hist(image_sharp)                         # Draw output grayscale histgram
    tmp = np.hstack((img4, image_sharp))               # Plot 2 images together
    plt.imshow(tmp)  
    plt.show()
    cv2.imwrite('output4.bmp', image_sharp)            # Save output image





    