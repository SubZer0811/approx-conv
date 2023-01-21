import numpy as np

def FFT_1D(signal):

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

def IFFT_1D(signal_f):
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

def FFT_2D(signal):

	rows_FFT = []
	for i in signal:
		rows_FFT.append(FFT_1D(i))
	
	cols_FFT = np.asarray(rows_FFT, dtype=complex).T
	signal_f = []
	for i in cols_FFT:
		signal_f.append(FFT_1D(i))
	return np.asarray(signal_f).T

def IFFT_2D(signal_f):

	rows_FFT = []
	for i in signal_f:
		rows_FFT.append(IFFT_1D(i))
	
	cols_FFT = np.asarray(rows_FFT, dtype=complex).T
	signal = []
	for i in cols_FFT:
		signal.append(IFFT_1D(i))
	return np.asarray(signal).T

def corr(img, kern):
	return IFFT_2D(FFT_2D(img) * FFT_2D(kern))