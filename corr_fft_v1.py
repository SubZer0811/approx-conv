import numpy as np

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