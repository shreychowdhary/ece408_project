#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
   	namespace op
	{

		#define ONE_IN_TILE_WIDTH 32
		#define ONE_OUT_TILE_WIDTH 28
		#define MUL_TILE_WIDTH 32

		#define MAX_M_SIZE 24
		#define MAX_C_SIZE 12
		#define MAX_K_SIZE 5

		// Constant memory variable for convolutional kernel. 
		__constant__ float convo_kernel[MAX_M_SIZE * MAX_C_SIZE * MAX_K_SIZE * MAX_K_SIZE];

		/**
		 * Performs forward convolutional kenel.
		 * Params:
		 * 	float *y	- Output array
		 *  float *x 	- Input array
		 *  float *k	- Convolutional kernel
		 *  int M		- Number of output feature maps
		 *  int C		- Number of input feature maps
		 * 	int H		- Width of Image
		 *  int W		- Height of Image
		 * 	int K		- Filter Size
		 *  H_grid  - Number of vertical tiles per output map
		 *  W_grid 	- Number of horizontal tiles per output map
		 *  H_out		- Output Height
		 *  W_out		- Output Width
		 */
		// use when C = 1
		__global__ void one_forward_kernel(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__ k, 
									const int M, const int C, const int H, const int W, const int K,
									const int H_grid, const int W_grid, const int H_out, const int W_out)
		{

			#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
			#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
			#define k4d(i3, i2, i1, i0) convo_kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
			const int tx = threadIdx.x;
			const int ty = threadIdx.y;
			const int b = blockIdx.x;
			const int m = blockIdx.y;
			const int h = (blockIdx.z / W_grid) * ONE_OUT_TILE_WIDTH + ty;
			const int w = (blockIdx.z % W_grid) * ONE_OUT_TILE_WIDTH + tx;
	
			__shared__ float xs[ONE_IN_TILE_WIDTH][ONE_IN_TILE_WIDTH];

			if (h >= 0 && h < H && w >= 0 && w < W) {
				xs[ty][tx] = x4d(b,0,h,w);
			} else {
				xs[ty][tx] = 0.0;
			}
			
			__syncthreads();

			if (ty >= ONE_OUT_TILE_WIDTH || tx >= ONE_OUT_TILE_WIDTH || h >= H_out || w >= W_out) {
				return;
			}

			float acc = 0.0;
			#pragma unroll 5
			for (int p = 0; p < K; ++p) {	// Loop over filter
				#pragma unroll 5
				for (int q = 0; q < K; ++q) {
					acc += xs[ty+p][tx+q] * k4d(m, 0, p, q);
				}
			}
			y4d(b, m, h, w) = acc;
			#undef y4d
			#undef x4d
			#undef k4d
		}
		/**
		 * Performs forward convolutional kenel.
		 * Params:
		 * 	float *y	- Output array
		 *  float *x 	- Input array
		 *  float *k	- Convolutional kernel
		 *  int M		- Number of output feature maps
		 *  int C		- Number of input feature maps
		 * 	int H		- Width of Image
		 *  int W		- Height of Image
		 * 	int K		- Filter Size
		 *  H_grid  - Number of vertical tiles per output map
		 *  W_grid 	- Number of horizontal tiles per output map
		 *  H_out		- Output Height
		 *  W_out		- Output Width
		 */
		// use when C > 1
		__global__ void mul_forward_kernel(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__ k, 
									const int M, const int C, const int H, const int W, const int K,
									const int H_grid, const int W_grid, const int H_out, const int W_out)
		{
			#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
			#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
			#define k4d(i3, i2, i1, i0) convo_kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
			/*const int tx = threadIdx.x;
			const int ty = threadIdx.y;
			const int tz = threadIdx.z;
			const int b = blockIdx.x;
			const int m = blockIdx.y;
			const int h = (blockIdx.z / W_grid) * MUL_OUT_TILE_WIDTH + ty;
			const int w = (blockIdx.z % W_grid) * MUL_OUT_TILE_WIDTH + tx;
	
			__shared__ float xs[MAX_C_SIZE][MUL_IN_TILE_WIDTH][MUL_IN_TILE_WIDTH];

			for (int c_grid = 0; c_grid < C/MUL_C_TILE_WIDTH; ++c_grid) {
				int c = c_grid*MUL_C_TILE_WIDTH+tz;
				if (h >= 0 && h < H && w >= 0 && w < W) {
					xs[c][ty][tx] = x4d(b,c,h,w);
				} else {
					xs[c][ty][tx] = 0.0;
				}
			}
			__syncthreads();

			if (ty >= MUL_OUT_TILE_WIDTH || tx >= MUL_OUT_TILE_WIDTH || h >= H_out || w >= W_out|| tz > 0) {
				return;
			}

			float acc = 0.0;
      #pragma unroll
			for (int c = 0; c < C; ++c) { // Sum over all input channels
        #pragma unroll
				for (int p = 0; p < K; ++p) {	// Loop over filter
          #pragma unroll
					for (int q = 0; q < K; ++q) {
						acc += xs[c][ty+p][tx+q] * k4d(m, c, p, q);
					}
				}
			}
			y4d(b, m, h, w) = acc;*/
			const int b = blockIdx.x;
			const int m = blockIdx.y;
			const int h = (blockIdx.z/W_grid)*MUL_TILE_WIDTH+threadIdx.y;
			const int w = (blockIdx.z%W_grid)*MUL_TILE_WIDTH+threadIdx.x;
			if (h >= H_out || w >= W_out) {
				return;
			}
			float acc = 0;
			#pragma unroll 12
			for (int c = 0; c < C; ++c) { // Sum over all input channels
				#pragma unroll 5
				for (int p = 0; p < K; ++p) {	// Loop over filter
					#pragma unroll 5
					for (int q = 0; q < K; ++q) {
						acc += x4d(b,c,h+p,w+q)*k4d(m,c,p,q);
					}
				}
			}
			y4d(b,m,h,w) = acc;
			#undef y4d
			#undef x4d
			#undef k4d
		}

		/* 
		This function is called by new-inl.h
		Any code you write should be executed by this function.
		For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
		*/
		template <>
		void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
		{

			// Extract the tensor dimensions into B,M,C,H,W,K
			const int B = x.shape_[0]; // Batch Size
			const int M = y.shape_[1]; // Output Feature Map Size
			const int C = x.shape_[1]; // Input Feature Map Size
			const int H = x.shape_[2]; // Height of Image
			const int W = x.shape_[3]; // Width of Image
			const int K = w.shape_[3]; // Filter Size
			const int H_out = H - K + 1; // Output Height
			const int W_out = W - K + 1; // Output Width

			// Initialize constant memory allocations
			int kernelSize = M * C * K * K * sizeof(float);
  		int offset = 0;
			cudaMemcpyToSymbol(convo_kernel, w.dptr_, kernelSize, offset, cudaMemcpyHostToDevice);

			
			if (C > 1) {
				const int W_grid = ceil(W_out / (float)MUL_TILE_WIDTH); // Number of horizontal tiles per output map
				const int H_grid = ceil(H_out / (float)MUL_TILE_WIDTH); // Number of vertical tiles per output map
				const int Z = H_grid * W_grid;
				dim3 gridDim(B, M, Z);
				dim3 blockDim(MUL_TILE_WIDTH,MUL_TILE_WIDTH, 1);

				mul_forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, M, C, H, W, K, H_grid, W_grid, H_out, W_out);
			} else {
				const int W_grid = ceil(W_out / (float)ONE_OUT_TILE_WIDTH); // Number of horizontal tiles per output map
				const int H_grid = ceil(H_out / (float)ONE_OUT_TILE_WIDTH); // Number of vertical tiles per output map
				const int Z = H_grid * W_grid;
				dim3 gridDim(B, M, Z);
				dim3 blockDim(ONE_IN_TILE_WIDTH,ONE_IN_TILE_WIDTH, 1);

				one_forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, M, C, H, W, K, H_grid, W_grid, H_out, W_out);
			}
			// Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
			MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

		}

		/* 
			This tells mxnet how to do an op when it's not a float.
			This is not used in the ECE408 project
		*/
		template <typename gpu, typename DType>
		void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
		{
			CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
		}

	}
}

#endif
