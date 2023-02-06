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
K_SIZE = 5

img, kern = corr_helper.gen_img_kern((9,9),(K_SIZE,K_SIZE))
# print(kern)
# kern = np.round(kern, 1)
# print(kern.dtype)
# kern = np.asarray(
# 		[[0.1, 0.1, 0.1],
# 		[0.2, 0.2, 0.2],
# 		[0.1, 0.1, 0.1]], dtype=np.float64)

kern = np.asarray([
 [0.7, 0.1, 0.4, 1.0, 0.3],
 [0.6, 0.6, 0.1, 0.4, 0.1],
 [0.4, 0.4, 0.6, 0.7, 0.3],
 [0.5, 1.0, 0.5, 0.2, 0.6],
 [0.4, 0.0, 0.3, 0.6, 0.2]])
# print(kern)


img = cv2.imread('lena.png', 0)
img = cv2.resize(img, (8, 8))
# img = np.ones((4,4))

p = F_SIZE//2 + 1

img_p, kern_p = corr_helper.pad_img_kern(img, kern, F_SIZE)

[W, W_I] = corr_fft_v3.compute_w(F_SIZE,np.complex128)


# FFT of kernel
v3_kern = corr_fft_v3.FFT_2D(kern_p, W)
sc_kern = scipy.fft.fft2(kern_p)
print("PSNR of FFT(kern).real", cv2.PSNR(v3_kern.real, sc_kern.real))
print("PSNR of FFT(kern).imag", cv2.PSNR(v3_kern.imag, sc_kern.imag))
print(f"Range of FFT(kern): [{np.min(v3_kern)}, {np.max(v3_kern)}]")
print()

# FFT of image
v3_img = corr_fft_v3.FFT_2D(img_p, W)
sc_img = scipy.fft.fft2(img_p)
print("PSNR of FFT(img).real", cv2.PSNR(v3_img.real, sc_img.real))
print("PSNR of FFT(img).imag", cv2.PSNR(v3_img.imag, sc_img.imag))
print(f"Range of FFT(img): [{np.min(v3_img)}, {np.max(v3_img)}]")
print()

# Element-wise multiplication
v3_mult = v3_img * v3_kern
sc_mult = sc_img * sc_kern
print("PSNR of mult.real", cv2.PSNR(v3_mult.real, sc_mult.real))
print("PSNR of mult.imag", cv2.PSNR(v3_mult.imag, sc_mult.imag))
print(f"Range of mult: [{np.min(v3_mult)}, {np.max(v3_mult)}]")
print()

# IFFT of mult
v3_out = corr_fft_v3.IFFT_2D(v3_mult, W_I)
sc_out = scipy.fft.ifft2(sc_mult)
print("PSNR of IFFT(mult).real", cv2.PSNR(v3_out.real, sc_out.real))
print("PSNR of IFFT(mult).imag", cv2.PSNR(v3_out.imag, sc_out.imag))
print(f"Range of IFFT(mult): [{np.min(v3_mult)}, {np.max(v3_mult)}]")
print()



'''
norm_img, mu, sigma = corr_helper.normalize(img)
img_ = corr_helper.denormalize(norm_img, mu, sigma)

# helper.view_image("v3", v3_out.real)
# helper.view_image("sc", sc_out.real)
'''
exit(0)



'''
img_n, mu, sigma = corr_helper.normalize(img_p)
out_ = corr_brute.corr(img_n, kern)
out = corr_helper.denormalize(out_, mu, sigma)

print(cv2.PSNR(out, corr_brute.corr(img_p, kern)))

img_n, max = corr_helper.normalize(img_p)
out_ = corr_brute.corr(img_n, kern)
out = corr_helper.denormalize(out_, max)

print(cv2.PSNR(out, corr_brute.corr(img_p, kern)))

out_v3_ = corr_fft_v3.corr(img_n, kern_p, W, W_I)
out_v3 = corr_helper.denormalize(out_v3_.real, max)

print(cv2.PSNR(out_v3, corr_brute.corr(img_p, kern)))
'''


# out_b = corr_brute.corr(img_p, kern)
# out_v3 = corr_fft_v3.corr(img_p, kern_p, W, W_I)
# out_v3 = np.roll(out_v3, p, [0,1])

# o_s = img.shape[0] + kern.shape[0] - 1
# p_ = (F_SIZE - o_s) // 2

# out_b = out_b[p_:p_+o_s,p_:p_+o_s]
# out_v3 = out_v3[p_:p_+o_s,p_:p_+o_s]

# print(out_v3.shape, out_b.shape)
# helper.view_image("Original", img_p)
# helper.view_image("Kernel", kern_p)
# helper.view_image("brute", out_b)
# # helper.view_image("v1", out_v1.real)
# helper.view_image("v3.real", out_v3.real)
# # helper.view_image("v3.imag", out_v3.imag)
# helper.view_image("diff", out_v3.real - out_b)

# print(np.mean((abs(out_v3 - out_b))))
# print(cv2.PSNR(np.float32(out_v3.real), np.float32(out_b)))
# print(np.sqrt(np.mean((out_v3.real-out_b)**2)))