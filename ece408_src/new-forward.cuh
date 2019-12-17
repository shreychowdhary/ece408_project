#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
   	namespace op
	{

		#define ONE_IN_TILE_WIDTH 32
		#define ONE_OUT_TILE_WIDTH 28
		#define MUL_IN_TILE_WIDTH1 16
		#define MUL_IN_TILE_HEIGHT1 12
		#define MUL_IN_TILE_WIDTH2 16
		#define MUL_IN_TILE_HEIGHT2 24
        

		#define MAX_M_SIZE 24
		#define MAX_C_SIZE 12
		#define MAX_K_SIZE 5

		// use when C = 1


        __global__ void mul_forward_kernel_one(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__ k, 
                                               const int M, const int C, const int H, const int W, const int K,
                                               const int H_out, const int W_out)
		{
            #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
            #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
            #define k4d(i3, i2, i1, i0) convo_kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

            __shared__ float tileK[MUL_IN_TILE_HEIGHT1][MUL_IN_TILE_WIDTH1];
            __shared__ float tileX[MUL_IN_TILE_WIDTH1][MUL_IN_TILE_WIDTH1];

            const int row = blockIdx.y * blockDim.y + threadIdx.y;
            const int col = blockIdx.x * blockDim.x + threadIdx.x;

            const int h_in = col / H_out;
            const int w_in = col % H_out;

            const int b = blockIdx.z;

            float acc = 0;

            const int tx = threadIdx.x;
            const int ty = threadIdx.y;
            const int filter_area = K * K;
            const int out_area = H_out * W_out;
            int T = ceil(1.0 * C * K * K / MUL_IN_TILE_WIDTH1);
            
            for (int t = 0; t < T; t++)
            {
                if (tx + t*MUL_IN_TILE_WIDTH1 < filter_area * C) {
                    tileK[ty][tx] = k[row*filter_area*C + (tx + t*MUL_IN_TILE_WIDTH1)];
                } else {
                    tileK[ty][tx] = 0;
                }

                // unrolled x tile
                #pragma unroll
                for (int i = 0; i < ceil(1.0 * MUL_IN_TILE_WIDTH1 / MUL_IN_TILE_HEIGHT1); i++)
                {
                    if (ty + i * MUL_IN_TILE_HEIGHT1 < MUL_IN_TILE_WIDTH1) {
                        int rx = t * MUL_IN_TILE_WIDTH1 + i * MUL_IN_TILE_HEIGHT1 + ty;
                        int k_lin = rx % filter_area;
                        if (rx < filter_area*C && col < out_area) {
                            tileX[ty + i * MUL_IN_TILE_HEIGHT1][tx] = x4d(b, rx / filter_area, h_in + k_lin/K, w_in + k_lin%K);
                        } else {
                            tileX[ty + i * MUL_IN_TILE_HEIGHT1][tx] = 0;
                        }
                    }
                }
                
                __syncthreads();
                #pragma unroll
                for (int ti = 0; ti < MUL_IN_TILE_WIDTH1; ti++)
                {
                    acc += tileK[ty][ti] * tileX[ti][tx];
                }
                __syncthreads();
            }

            if (col < out_area) {
                y4d(b, row, 0, col) = acc;
            }
           
            #undef y4d
            #undef x4d
            #undef k4d
		}
        __global__ void mul_forward_kernel_two(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__ k, 
                                               const int M, const int C, const int H, const int W, const int K,
                                               const int H_out, const int W_out)
		{
            #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
            #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
            #define k4d(i3, i2, i1, i0) convo_kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

            __shared__ float tileK[MUL_IN_TILE_HEIGHT2][MUL_IN_TILE_WIDTH2];
            __shared__ float tileX[MUL_IN_TILE_WIDTH2][MUL_IN_TILE_WIDTH2];

            const int row = blockIdx.y * blockDim.y + threadIdx.y;
            const int col = blockIdx.x * blockDim.x + threadIdx.x;

            const int h_in = col / H_out;
            const int w_in = col % H_out;

            const int b = blockIdx.z;

            float acc = 0.0;
            
            const int tx = threadIdx.x;
            const int ty = threadIdx.y;
            const int filter_area = K * K;
            const int out_area = H_out * W_out;
            int T = ceil(1.0 * C * K * K / MUL_IN_TILE_WIDTH2);

            for (int t = 0; t < T; t++)
            {
                if (tx + t*MUL_IN_TILE_WIDTH2 < filter_area * C) {
                    tileK[ty][tx] = k[row*filter_area*C + (tx + t*MUL_IN_TILE_WIDTH2)];
                } else {
                    tileK[ty][tx] = 0;
                }

                if (ty < MUL_IN_TILE_WIDTH2) {
                    int rx = t * MUL_IN_TILE_WIDTH2 + ty;
                    int k_lin = rx % filter_area;
                    if (rx < filter_area*C && col < out_area) {
                        tileX[ty][tx] = x4d(b, rx / filter_area, h_in + k_lin/K, w_in + k_lin%K);
                    } else {
                        tileX[ty][tx] = 0;
                    }
                }
                __syncthreads();
                #pragma unroll
                for (int ti = 0; ti < MUL_IN_TILE_WIDTH2; ti++)
                {
                    acc += tileK[ty][ti] * tileX[ti][tx];
                }
                __syncthreads();
            }

            if (col < out_area) {
                y4d(b, row, 0, col) = acc;
            }
            
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
            // printf("  B = %d\n", B);
            // printf("  M = %d\n", M);
            // printf("  C = %d\n", C);
            // printf("  H = %d\n", H);
            // printf("  W = %d\n", W);
            // printf("  K = %d\n", K);
            // printf("  H_out = %d\n", H_out);
            // printf("  W_out = %d\n", W_out);
			
			if (C > 1) {
                dim3 gridDim(ceil(1.0 * H_out * W_out / MUL_IN_TILE_WIDTH2), ceil(1.0 * M / MUL_IN_TILE_HEIGHT2), B);
                dim3 blockDim(MUL_IN_TILE_WIDTH2, MUL_IN_TILE_HEIGHT2);
				mul_forward_kernel_two<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, M, C, H, W, K, H_out, W_out);
				
			} else {
                dim3 gridDim(ceil(1.0 * H_out * W_out / MUL_IN_TILE_WIDTH1), ceil(1.0 * M / MUL_IN_TILE_HEIGHT1), B);
                dim3 blockDim(MUL_IN_TILE_WIDTH1, MUL_IN_TILE_HEIGHT1);
				mul_forward_kernel_one<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, M, C, H, W, K, H_out, W_out);
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
