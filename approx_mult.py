import numpy as np
import struct
import ctypes

f = 0.3
f_b = '{:032b}'.format(np.asarray(f, dtype=np.float32).view(np.int32).item())
s = f_b[0]
e = f_b[1:9]
m = f_b[9:]

def get_bin(num):
	f_b = '{:032b}'.format(np.asarray(num, dtype=np.float32).view(np.int32).item())
	s = f_b[0]
	e = f_b[1:9]
	m = f_b[9:]
	return (s, e, m)

def mult(A, B):

	A_bin = ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', A))
	B_bin = ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', B))
	A_s, A_e, A_m = A_bin[0], A_bin[1:9] , A_bin[9:]
	B_s, B_e, B_m = B_bin[0], B_bin[1:9] , B_bin[9:]
	
	print(A_bin)
	print(B_bin)

	out_s = bin(int(A_s) ^ int(B_s))[2:]
	out_e = bin(int(A_e, 2) + int(B_e, 2) - 127)[2:]
	out_m_ = bin(int('1'+A_m, 2) * int('1'+B_m, 2))[2:]

	if len(out_m_[:-46]) > 1:
		out_e += 1
		out_m__ = out_m_[-47:]
	else:
		out_m__ = out_m_[-46:]

	out_m = out_m__[:23]
	return (out_s, out_e, out_m)

if __name__ == "__main__":
	s, e, m = mult(10, 0.5)
	# out = (256 - e) * m
	print(s+e+m)
	print(type(s))
	print(type(e))
	print(type(m))
	# print(out)

	b = ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', 0.5))
	print(b)
	i = ctypes.c_float.from_buffer(ctypes.c_uint32(int(s+e+m, 2))).value
	print(i)