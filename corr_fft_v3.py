import numpy as np
import corr_helper
import helper

def compute_w(sig_len,dtype):
	W = [[0]*sig_len for i in range(sig_len)]
	W_I = [[0]*sig_len for i in range(sig_len)]

	for i in range(0, sig_len):
		for j in range(1, sig_len):
			W[i][j] = np.power(np.power(np.e, -1j * np.pi / j), i)
			W_I[i][j] = np.power(np.power(np.e, 1j * np.pi / j), i)
			# W[i][j] = (i,j)

	return (W, W_I)

def FFT_1D(signal, W):

	signal_f = [signal[int(format(str(bin(i))[:1:-1], f'0{int(np.log2(len(signal)))}s'), base=2)] for i in range(len(signal))]
	signal_f = np.asarray(signal_f, dtype=np.complex128)
	
	n = len(signal)
	for s in range(1, int(np.log2(n))+1):
		m = 1 << s
		m2 = m >> 1
		
		for j in range(m2):
			for k in range(j, n, m):

				t = W[j][m2] * signal_f[k + m2]
				u = signal_f[k]

				signal_f[k] = u + t
				signal_f[k + m2] = u - t

	return signal_f

def IFFT_1D(signal_f, W_I):
	n = len(signal_f)
	signal = [i/n for i in signal_f]


	for s in range(int(np.log2(n)), 0, -1):
		m = pow(2,s)	# interval between butterflies
		m2 = m >> 1		# size of butterfly

		for j in range(m2):
			for k in range(j, n, m):

				u = signal[k]
				v = signal[k + m2]

				signal[k] = u + v
				signal[k + m2] = (u - v) * W_I[j][m2]

	signal = [signal[int(format(str(bin(i))[:1:-1], f'0{int(np.log2(len(signal)))}s'), base=2)] for i in range(len(signal))]

	return signal

def FFT_2D(signal, W):

	rows_FFT = []
	for i in signal:
		rows_FFT.append(FFT_1D(i, W))
	
	cols_FFT = np.asarray(rows_FFT, dtype=np.complex128).T
	signal_f = []
	for i in cols_FFT:
		signal_f.append(FFT_1D(i, W))
	return np.asarray(signal_f).T

def IFFT_2D(signal_f, W_I):

	rows_FFT = []
	for i in signal_f:
		rows_FFT.append(IFFT_1D(i, W_I))
	
	cols_FFT = np.asarray(rows_FFT, dtype=np.complex128).T
	signal = []
	for i in cols_FFT:
		signal.append(IFFT_1D(i, W_I))
	return np.asarray(signal).T

def corr(img, kern, W, W_I, unpad=None):
	res = IFFT_2D(FFT_2D(img, W) * FFT_2D(kern, W).conj(), W_I)

	if unpad:
		res = corr_helper.fft_unpad(res, unpad)

	return res

if __name__ == "__main__":

	img, kern = corr_helper.gen_img_kern((100, 100), (5,5), F_SIZE=128)

	W, W_I = compute_w(128)
	print(np.allclose(img, IFFT_2D(FFT_2D(img, W), W_I)))

