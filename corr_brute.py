import numpy as np

def corr(img, kern):
	K_HEIGHT = kern.shape[0]
	K_WIDTH = kern.shape[1]

	IMG_HEIGHT = img.shape[0]
	IMG_WIDTH = img.shape[1]

	padd_ver = int(K_HEIGHT/2)
	padd_hor = int(K_WIDTH/2)

	img_pad = np.pad(img, ((padd_ver,padd_ver),(padd_hor,padd_hor)))

	out = np.zeros(img.shape)

	for i in range(IMG_HEIGHT):
		for j in range(IMG_WIDTH):
			sum = 0
			for k in range(kern.shape[0]):
				for l in range(kern.shape[1]):
					sum += img_pad[i+k,j+l] * kern[k,l]
			
			out[i,j] = sum

	return out