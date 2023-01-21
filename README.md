# Approximate Computation of Correlation/Convolution

## corr_brute
This is the brute force method of computing correlation by moving the kernel over the input image and find the sum of products of the overlapping pixels

## corr_fft_v1
This method performs correlation by transforming the image and kernel using Fourier Transform and performing element-wise multiplication and finally performing inverse fourier transform. This method performs fourier and inverse fourier transform using a recursive approach.

## corr_fft_v2
In this method, an iterative apprpoach of corr_fft_v1 is implemented. This is so that it becomes easy to implement in hardware.

## corr_fft_v3
In this method, the iterative approach of corr_fft_v2 is optimized by precomputing the w (omega) terms and storing them in a hash table.