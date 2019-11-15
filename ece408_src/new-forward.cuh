#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

#define TILE_WIDTH 32

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
 *  W_grid 		- Number of horizontal tiles per output map
 *  H_out		- Output Height
 *  W_out		- Output Width
 */
__global__ void forward_kernel(float *y, const float *x, const float *k, 
							   const int M, const int C, const int H, const int W, const int K, 
							   const int W_grid, const int H_out, const int W_out)
{

	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

	const int b = blockIdx.x;
	const int m = blockIdx.y;
	const int h = (blockIdx.z/W_grid)*TILE_WIDTH+threadIdx.y;
	const int w = (blockIdx.z%W_grid)*TILE_WIDTH+threadIdx.x;
	if (h >= H_out || w >= W_out) {
		return;
	}
	float acc = 0;
	for (int c = 0; c < C; ++c) { // Sum over all input channels
		for (int p = 0; p < K; ++p) {	// Loop over filter
			for (int q = 0; q < K; ++q) {
				acc += x4d(b,c,h+p,w+q)*k4d(m,c,p,q);
			}
		}
	}
	y4d(b,m,h,w) = acc;

	// An example use of these macros:
	// float a = y4d(0,0,0,0)
	// y4d(0,0,0,0) = a
		

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
	const int H_out = H-K+1; // Output Height
	const int W_out = W-K+1; // Output Width
	const int W_grid = ceil(W_out / (float)TILE_WIDTH); // Number of horizontal tiles per output map
	const int H_grid = ceil(H_out / (float)TILE_WIDTH); // Number of vertical tiles per output map
	const int Z = H_grid*W_grid;
    // Set the kernel dimensions
    dim3 gridDim(B,M,Z);
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_,M,C,H,W,K,W_grid,H_out,W_out);

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
