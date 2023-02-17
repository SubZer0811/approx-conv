import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy

import corr_brute
import corr_fft_v1
import corr_fft_v2
import corr_fft_v3
import corr_fft_v3_1
import corr_helper

import helper

F_SIZE = 128
IMG_SIZE = 64
K_SIZE = 7

img, kern = corr_helper.gen_img_kern((IMG_SIZE,IMG_SIZE),(K_SIZE,K_SIZE))

img = cv2.imread('lena.png', 0)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))


[W, W_I] = corr_fft_v3.compute_w(F_SIZE)
[W_1, W_I_1] = corr_fft_v3_1.compute_w(F_SIZE)
print(cv2.PSNR(np.asarray(W).real, np.asarray(W_1).real))
print(cv2.PSNR(np.asarray(W).imag, np.asarray(W_1).imag))
W = np.asarray(W)
W_I = np.asarray(W_I)
print(np.allclose(W, W_I.conj()))
# exit(0)

img_p, kern_p = corr_helper.pad_img_kern(img, kern, F_SIZE)

out_b = corr_brute.corr(img_p, kern, img.shape[0]+kern.shape[0]-1)
out_v3_1 = corr_fft_v3_1.corr(img_p, kern_p, W_1, W_I_1, (IMG_SIZE, K_SIZE))

helper.view_image("brute", out_b)
helper.view_image("v3_1", out_v3_1.real)

print(cv2.PSNR(np.float32(out_v3_1.real), np.float32(out_b)))
print(np.sqrt(np.mean((out_v3_1.real-out_b)**2)))