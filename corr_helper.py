import numpy as np

def gen_img_kern(img_size, kern_size, F_SIZE=None):
	img = np.random.random(img_size)
	kern = np.random.random(kern_size)
	
	if F_SIZE:
		return pad_img_kern(img, kern, F_SIZE)
	return (img, kern)

def pad_img_kern(img, kern, F_SIZE):
	img_h = img.shape[0]
	kern_h = kern.shape[0]
	
	final_h = pow(2,int(np.ceil(np.log2(img_h + kern_h - 1))))
	assert final_h <= F_SIZE, "Image and kernel size needs to be smaller!"

	img_hp1 = (F_SIZE - img_h) // 2
	img_hp2 = F_SIZE - img_h - img_hp1

	kern_hp1 = (F_SIZE - kern_h) // 2
	kern_hp2 = F_SIZE - kern_h - kern_hp1

	img_p = np.pad(img, ((img_hp1,img_hp2),(img_hp1,img_hp2)))
	kern_p = np.pad(kern, ((kern_hp1,kern_hp2),(kern_hp1,kern_hp2)))

	return (img_p, kern_p)

def corr(img, kern, fft_func, ifft_func):
	return ifft_func(fft_func(img) * fft_func(kern))