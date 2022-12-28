import numpy as np
from corr_func import *
import helper
import cv2

# Size of image and kernel after padding
F_SIZE = 16

# Get image and kernel
np.random.seed = 0
img = np.random.randint(0, 256, (8, 8))
kern = np.random.random((3, 3))

# Size of both image and kernel needs to be atleast of size (N+M-1) and a power of 2
# where N is size of image and M is size of kernel
# pad both image and kernel accordingly
img_h = img.shape[0]
kern_h = kern.shape[0]

final_h = pow(2,int(np.ceil(np.log2(img_h + kern_h - 1))))
assert final_h <= 128, "Image and kernel size needs to be smaller!"

img_hp1 = (F_SIZE - img_h) // 2
img_hp2 = F_SIZE - img_h - img_hp1

kern_hp1 = (F_SIZE - kern_h) // 2
kern_hp2 = F_SIZE - kern_h - kern_hp1

img_p = np.pad(img, ((img_hp1,img_hp2),(img_hp1,img_hp2)))
kern_p = np.pad(kern, ((kern_hp1,kern_hp2),(kern_hp1,kern_hp2)))

# Apply fourier transform on both image and kernel
# Multiply the image and kernel in fourier domain
f = FFT_2D_iter(img_p) * FFT_2D_iter(kern_p).conj()

# Apply inverse fourier transform on the multiplied signals
out = IFFT_2D_iter(f)

# Shift the output so as to bring the output to the center
out = np.roll(out, shift=F_SIZE//2, axis=[0,1])

# helper.view_image(img_p)
# helper.view_image(kern_p)
# helper.view_image(out.real)
# helper.view_image(cv2.filter2D(img_p, -1, kernel=kern_p))

# helper.plt.show()
cv2_out = cv2.filter2D(img_p, -1, kern_p, borderType=cv2.BORDER_CONSTANT)

error = np.mean(np.square(out-cv2_out))
print(error)

print(np.allclose(out.real, cv2.filter2D(img_p, -1, kernel=kern_p, borderType=cv2.BORDER_CONSTANT).real))