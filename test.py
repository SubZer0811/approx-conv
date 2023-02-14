import numpy as np
import cv2

import corr_brute
import corr_fft_v1
import corr_fft_v2
import corr_fft_v3
import corr_helper
import helper

F_SIZE = 128
K_SIZE = 5
IMG_SIZE = [8, 16, 32, 64]

np.random.seed(0)

img_orig = cv2.imread("lena.png", 0)
kern = np.random.rand(K_SIZE, K_SIZE)

W, W_I = corr_fft_v3.compute_w(F_SIZE)

timer = helper.Timer()

for img_size in IMG_SIZE:
	print(img_size, end=", ")

	img = cv2.resize(img_orig, (img_size, img_size))
	img_p, kern_p = corr_helper.pad_img_kern(img, kern, F_SIZE)

	timer.start()
	out_b = corr_brute.corr(img_p, kern, (img_size+K_SIZE-1))
	timer.stop()
	print(timer, end=", ")

	timer.start()
	out_v1 = corr_fft_v1.corr(img_p, kern_p, (img_size, K_SIZE))
	timer.stop()
	print(timer, end=", ")

	timer.start()
	out_v2 = corr_fft_v2.corr(img_p, kern_p, (img_size, K_SIZE))
	timer.stop()
	print(timer, end=", ")

	timer.start()
	out_v3 = corr_fft_v3.corr(img_p, kern_p, W, W_I, (img_size, K_SIZE))
	timer.stop()
	print(timer)

	# print(out_b.shape, out_v1.shape, out_v2.shape, out_v3.shape)

	# print(cv2.PSNR(out_b, out_v1.real))
	# print(cv2.PSNR(out_b, out_v2.real))
	# print(cv2.PSNR(out_b, out_v3.real))