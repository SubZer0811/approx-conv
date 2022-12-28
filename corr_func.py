import numpy as np
import scipy

def FFT_1D_iter(signal):

	signal_f = [signal[int(format(str(bin(i))[:1:-1], f'0{int(np.log2(len(signal)))}s'), base=2)] for i in range(len(signal))]

	n = len(signal)
	for s in range(1, int(np.log2(n))+1):
		m = 1 << s
		m2 = m >> 1

		w = 1+0j

		wm = np.power(np.e, -1j * np.pi / m2)

		for j in range(m2):
			for k in range(j, n, m):

				t = w * signal_f[k + m2]
				u = signal_f[k]

				signal_f[k] = u + t
				signal_f[k + m2] = u - t

			w *= wm

	return signal_f

def IFFT_1D_iter(signal_f):
	n = len(signal_f)
	signal = [i/n for i in signal_f]


	for s in range(int(np.log2(n)), 0, -1):
		m = pow(2,s)	# interval between butterflies
		m2 = m >> 1		# size of butterfly

		w = 1+0j

		wm = np.power(np.e, -1j * np.pi / m2)

		for j in range(m2):
			for k in range(j, n, m):

				u = signal[k]
				v = signal[k + m2]

				signal[k] = u + v
				signal[k + m2] = (u - v) * w

			w /= wm

	signal = [signal[int(format(str(bin(i))[:1:-1], f'0{int(np.log2(len(signal)))}s'), base=2)] for i in range(len(signal))]

	return signal

def FFT_2D_iter(signal):

	rows_FFT = []
	for i in signal:
		rows_FFT.append(FFT_1D_iter(i))
	
	cols_FFT = np.asarray(rows_FFT, dtype=complex).T
	signal_f = []
	for i in cols_FFT:
		signal_f.append(FFT_1D_iter(i))
	return np.asarray(signal_f).T

def IFFT_2D_iter(signal_f):

	rows_FFT = []
	for i in signal_f:
		rows_FFT.append(IFFT_1D_iter(i))
	
	cols_FFT = np.asarray(rows_FFT, dtype=complex).T
	signal = []
	for i in cols_FFT:
		signal.append(IFFT_1D_iter(i))
	return np.asarray(signal).T

def corr_brute(img, kern):
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

def FFT_1D(signal):

	n = len(signal)
	if n == 1:
		return signal

	w = np.power(np.e, -2 * np.pi * (1j) / n)
	signal_e = [signal[i] for i in range(len(signal)) if i%2 == 0]
	signal_o = [signal[i] for i in range(len(signal)) if i%2 == 1]

	SIGNAL = [0] * n
	SIGNAL_E = FFT_1D(signal_e)        # like how 'f' is signal in time domain and 'F' is signal in frequency domain,
	SIGNAL_O = FFT_1D(signal_o)        # 'signal' is in time domain and 'SIGNAL' is in frequency domain

	n2 = round(n/2)
	for i in range(n2):
		SIGNAL[i] = SIGNAL_E[i] + np.power(w, i) * SIGNAL_O[i]
		SIGNAL[i+n2] = SIGNAL_E[i] - np.power(w, i) * SIGNAL_O[i]

	return SIGNAL

def IFFT_1D(signal):

	def worker(signal):
		n = len(signal)
		if n == 1:
			return signal

		w = np.power(np.e, 2 * np.pi * (1j) / n)
		signal_e = [signal[i] for i in range(len(signal)) if i%2 == 0]
		signal_o = [signal[i] for i in range(len(signal)) if i%2 == 1]

		SIGNAL = [0] * n
		SIGNAL_E = worker(signal_e)
		SIGNAL_O = worker(signal_o)

		n2 = round(n/2)
		for i in range(n2):
			SIGNAL[i] = SIGNAL_E[i] + np.power(w, i) * SIGNAL_O[i]
			SIGNAL[i+n2] = SIGNAL_E[i] - np.power(w, i) * SIGNAL_O[i]

		return SIGNAL
	return np.asarray(worker(signal)) / len(signal)

def FFT_2D(img):
	height, width = img.shape[0], img.shape[1]
	h_pad = 0
	w_pad = 0
	
	if np.ceil(np.log2(height)) != np.floor(np.log2(height)):
		h_pad = np.power(2, int(np.ceil(np.log2(height)))) - height

	if np.ceil(np.log2(width)) != np.floor(np.log2(width)):
		w_pad = np.power(2, int(np.ceil(np.log2(height)))) - width

	img = np.pad(img, ((0,h_pad),(0,w_pad)))
	
	rows_FFT = []
	for i in img:
		rows_FFT.append(FFT_1D(i))
	
	cols_FFT = np.asarray(rows_FFT, dtype=complex).T
	IMG = []
	for i in cols_FFT:
		IMG.append(FFT_1D(i))
	return np.asarray(IMG).T

def IFFT_2D(img):
	rows_FFT = []
	for i in img:
		rows_FFT.append(IFFT_1D(i))
	
	cols_FFT = np.asarray(rows_FFT, dtype=complex).T
	IMG = []
	for i in cols_FFT:
		IMG.append(IFFT_1D(i))
	return np.asarray(IMG).T

def pad2power2(sig):
	height, width = sig.shape[0], sig.shape[1]
	h_pad = 0
	w_pad = 0

	if np.ceil(np.log2(height)) != np.floor(np.log2(height)):
		h_pad = np.power(2, int(np.ceil(np.log2(height)))) - height

	if np.ceil(np.log2(width)) != np.floor(np.log2(width)):
		w_pad = np.power(2, int(np.ceil(np.log2(height)))) - width

	sig = np.pad(sig, ((0,h_pad),(0,w_pad)))

	return sig

def pad_img_kern(img, kern):

	# padd input image with at least (K-1) zeros to avoid aliasing
	# K is size of kernel
	K = kern.shape[0]
	img_pad = np.pad(img, (K-1, K-1))

	# pad filter to size of image by padding with zeros
	kern_pad = np.pad(kern, (int((img_pad.shape[0]-K)/2),int((img_pad.shape[0]-K)/2)))
	
	img_pad = pad2power2(img_pad)
	kern_pad = pad2power2(kern_pad)

	return img_pad, kern_pad

def corr_fft(img, kern):

	# img_pad, kern_pad = pad_img_kern(img, kern)
	img_pad = img
	kern_pad = kern
	# Transform image and filter using DFT

	# img_f_s = scipy.fft.fft2(img_pad)
	img_f = FFT_2D(img_pad)
	# kern_f_s = scipy.fft.fft2(kern_pad).conj()
	kern_f = FFT_2D(kern_pad).conj()

	# print(np.allclose(img_f, img_f_s))
	# print(np.allclose(kern_f, kern_f_s))

	# Element-wise multiply image and filter in frequency domain
	out_f = img_f * kern_f
	# out_f_s = img_f_s * kern_f_s

	# print(np.allclose(out_f, out_f_s))

	# Convert output back to spatial domain
	# out_s = scipy.fft.ifft2(out_f_s)
	out = IFFT_2D(out_f)

	# print(np.allclose(out, out_s))

	# error = np.mean(np.square(out.real-out_s.real))
	# print(error)

	# out = np.roll(out, shift=8, axis=[0,1])
	# error = np.mean(np.square(out.real[5:5+9,5:5+9]-cv2.filter2D(img_pad, -1, kern_pad)[5:5+9,5:5+9]))
	# print(error)
	
	# view_image(cv2.filter2D(img_pad, -1, kern_pad))
	# Reconstruct Linear convolution from Circular convolution
	out = out.real
	# view_image(out.real)
	
	return out