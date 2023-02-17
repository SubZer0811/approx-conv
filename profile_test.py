import cv2
import numpy as np

import corr_fft_v2
import corr_helper

F_SIZE = 128
IMG_SIZE = 64
K_SIZE = 7

img, kern = corr_helper.gen_img_kern((IMG_SIZE,IMG_SIZE),(K_SIZE,K_SIZE))

img = cv2.imread('lena.png', 0)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

img_p, kern_p = corr_helper.pad_img_kern(img, kern, F_SIZE)

out = corr_fft_v2.corr(img_p, kern_p, (IMG_SIZE, K_SIZE))