import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy

import corr_brute
import corr_fft_v1
import corr_fft_v2
import corr_fft_v3
import corr_helper

import helper

F_SIZE = 32
IMG_SIZE = 14
K_SIZE = 7

img, kern = corr_helper.gen_img_kern((IMG_SIZE,IMG_SIZE),(K_SIZE,K_SIZE))

img = cv2.imread('lena.png', 0)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

img_p, kern_p = corr_helper.pad_img_kern(img, kern, F_SIZE)

[W, W_I] = corr_fft_v3.compute_w(F_SIZE)

out_b = corr_brute.corr(img_p, kern, img.shape[0]+kern.shape[0]-1)
out_v3 = corr_fft_v3.corr(img_p, kern_p, W, W_I, (IMG_SIZE, K_SIZE))
out_v2 = corr_fft_v2.corr(img_p, kern_p, (IMG_SIZE, K_SIZE))
out_v1 = corr_fft_v1.corr(img_p, kern_p, (IMG_SIZE, K_SIZE))

# print(out_v3.shape, out_b.shape)
# helper.view_image("Original", img_p)
# helper.view_image("Kernel", kern_p)
# helper.view_image("brute", out_b)
# helper.view_image("v3.real", out_v3.real)
# # helper.view_image("v1", out_v1.real)
# # helper.view_image("v3.imag", out_v3.imag)
# helper.view_image("diff", out_v3.real - out_b)

print(cv2.PSNR(np.float32(out_v3.real), np.float32(out_b)))
print(cv2.PSNR(np.float32(out_v2.real), np.float32(out_b)))
print(cv2.PSNR(np.float32(out_v1.real), np.float32(out_b)))
print(np.sqrt(np.mean((out_v3.real-out_b)**2)))