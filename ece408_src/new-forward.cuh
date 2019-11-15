#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{
const int TILE_WIDTH = 8;

__constant__ float MASK[7200];

__global__ void forward_kernel(float * __restrict__ y, const float __restrict__ *x, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
    
    int b1 = blockIdx.x;
    int b2 = blockIdx.y;
    int b3 = blockIdx.z;
    int t1 = threadIdx.x;
    int t2 = threadIdx.y;
    int t3 = threadIdx.z;
    int m = b1 * TILE_WIDTH + t1;
    int h = b2 * TILE_WIDTH + t2;
    int w = b3 * TILE_WIDTH + t3;

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define K4d(i3, i2, i1, i0) MASK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    __shared__ float subTile[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];

    int currM = blockIdx.x * blockDim.x;
    int currH = blockIdx.y * blockDim.y;
    int currW = blockIdx.z * blockDim.z;
    int nextM = (blockIdx.x + 1) * blockDim.x;
    int nextH = (blockIdx.y + 1) * blockDim.y;
    int nextW = (blockIdx.z + 1) * blockDim.z;

    int mhw = m * (H_out * W_out) + h * (W_out) + w;
    for (int b = 0; b < B; b++)
    {
        int bmhw = b * (M * H_out * W_out) + mhw;
        if (m < M && h < H && w < W)
            subTile[t1][t2][t3] = x4d(b, m, h, w);
        else
            subTile[t1][t2][t3] = 0;
        __syncthreads();
        if (m < M && h < H_out && w < W_out)
            y[bmhw] = 0;
        for (int c = 0; c < C; c++)
        {
            // for (int p = 0; p < K; p++)
            // {
            //     for (int q = 0; q < K; q++)
            //     {
            //         if (m < M && h < H_out && w < W_out)
            //         {
            //             if (c >= currM && c < nextM && (h + p) >= currH && (h + p) < nextH && (w + q) >= currW && (w + q) < nextW)
            //                 y[bmhw] += subTile[c - currM][t2 + p][t3 + q] * K4d(m, c, p, q);
            //             else
            //                 y[bmhw] += x4d(b, c, (h + p), (w + q)) * K4d(m, c, p, q);
            //         }
            //     }
            // }
            if (m < M && h < H_out && w < W_out)
            {
                if (c >= currM && c < nextM && (h + 0) >= currH && (h + 0) < nextH && (w + 0) >= currW && (w + 0) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 0][t3 + 0] * K4d(m, c, 0, 0);
                else
                    y[bmhw] += x4d(b, c, (h + 0), (w + 0)) * K4d(m, c, 0, 0);

                if (c >= currM && c < nextM && (h + 0) >= currH && (h + 0) < nextH && (w + 1) >= currW && (w + 1) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 0][t3 + 1] * K4d(m, c, 0, 1);
                else
                    y[bmhw] += x4d(b, c, (h + 0), (w + 1)) * K4d(m, c, 0, 1);

                if (c >= currM && c < nextM && (h + 0) >= currH && (h + 0) < nextH && (w + 2) >= currW && (w + 2) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 0][t3 + 2] * K4d(m, c, 0, 2);
                else
                    y[bmhw] += x4d(b, c, (h + 0), (w + 2)) * K4d(m, c, 0, 2);

                if (c >= currM && c < nextM && (h + 0) >= currH && (h + 0) < nextH && (w + 3) >= currW && (w + 3) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 0][t3 + 3] * K4d(m, c, 0, 3);
                else
                    y[bmhw] += x4d(b, c, (h + 0), (w + 3)) * K4d(m, c, 0, 3);

                if (c >= currM && c < nextM && (h + 0) >= currH && (h + 0) < nextH && (w + 4) >= currW && (w + 4) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 0][t3 + 4] * K4d(m, c, 0, 4);
                else
                    y[bmhw] += x4d(b, c, (h + 0), (w + 4)) * K4d(m, c, 0, 4);

                if (c >= currM && c < nextM && (h + 1) >= currH && (h + 1) < nextH && (w + 0) >= currW && (w + 0) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 1][t3 + 0] * K4d(m, c, 1, 0);
                else
                    y[bmhw] += x4d(b, c, (h + 1), (w + 0)) * K4d(m, c, 1, 0);

                if (c >= currM && c < nextM && (h + 1) >= currH && (h + 1) < nextH && (w + 1) >= currW && (w + 1) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 1][t3 + 1] * K4d(m, c, 1, 1);
                else
                    y[bmhw] += x4d(b, c, (h + 1), (w + 1)) * K4d(m, c, 1, 1);

                if (c >= currM && c < nextM && (h + 1) >= currH && (h + 1) < nextH && (w + 2) >= currW && (w + 2) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 1][t3 + 2] * K4d(m, c, 1, 2);
                else
                    y[bmhw] += x4d(b, c, (h + 1), (w + 2)) * K4d(m, c, 1, 2);

                if (c >= currM && c < nextM && (h + 1) >= currH && (h + 1) < nextH && (w + 3) >= currW && (w + 3) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 1][t3 + 3] * K4d(m, c, 1, 3);
                else
                    y[bmhw] += x4d(b, c, (h + 1), (w + 3)) * K4d(m, c, 1, 3);

                if (c >= currM && c < nextM && (h + 1) >= currH && (h + 1) < nextH && (w + 4) >= currW && (w + 4) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 1][t3 + 4] * K4d(m, c, 1, 4);
                else
                    y[bmhw] += x4d(b, c, (h + 1), (w + 4)) * K4d(m, c, 1, 4);

                if (c >= currM && c < nextM && (h + 2) >= currH && (h + 2) < nextH && (w + 0) >= currW && (w + 0) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 2][t3 + 0] * K4d(m, c, 2, 0);
                else
                    y[bmhw] += x4d(b, c, (h + 2), (w + 0)) * K4d(m, c, 2, 0);

                if (c >= currM && c < nextM && (h + 2) >= currH && (h + 2) < nextH && (w + 1) >= currW && (w + 1) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 2][t3 + 1] * K4d(m, c, 2, 1);
                else
                    y[bmhw] += x4d(b, c, (h + 2), (w + 1)) * K4d(m, c, 2, 1);

                if (c >= currM && c < nextM && (h + 2) >= currH && (h + 2) < nextH && (w + 2) >= currW && (w + 2) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 2][t3 + 2] * K4d(m, c, 2, 2);
                else
                    y[bmhw] += x4d(b, c, (h + 2), (w + 2)) * K4d(m, c, 2, 2);

                if (c >= currM && c < nextM && (h + 2) >= currH && (h + 2) < nextH && (w + 3) >= currW && (w + 3) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 2][t3 + 3] * K4d(m, c, 2, 3);
                else
                    y[bmhw] += x4d(b, c, (h + 2), (w + 3)) * K4d(m, c, 2, 3);

                if (c >= currM && c < nextM && (h + 2) >= currH && (h + 2) < nextH && (w + 4) >= currW && (w + 4) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 2][t3 + 4] * K4d(m, c, 2, 4);
                else
                    y[bmhw] += x4d(b, c, (h + 2), (w + 4)) * K4d(m, c, 2, 4);

                if (c >= currM && c < nextM && (h + 3) >= currH && (h + 3) < nextH && (w + 0) >= currW && (w + 0) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 3][t3 + 0] * K4d(m, c, 3, 0);
                else
                    y[bmhw] += x4d(b, c, (h + 3), (w + 0)) * K4d(m, c, 3, 0);

                if (c >= currM && c < nextM && (h + 3) >= currH && (h + 3) < nextH && (w + 1) >= currW && (w + 1) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 3][t3 + 1] * K4d(m, c, 3, 1);
                else
                    y[bmhw] += x4d(b, c, (h + 3), (w + 1)) * K4d(m, c, 3, 1);

                if (c >= currM && c < nextM && (h + 3) >= currH && (h + 3) < nextH && (w + 2) >= currW && (w + 2) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 3][t3 + 2] * K4d(m, c, 3, 2);
                else
                    y[bmhw] += x4d(b, c, (h + 3), (w + 2)) * K4d(m, c, 3, 2);

                if (c >= currM && c < nextM && (h + 3) >= currH && (h + 3) < nextH && (w + 3) >= currW && (w + 3) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 3][t3 + 3] * K4d(m, c, 3, 3);
                else
                    y[bmhw] += x4d(b, c, (h + 3), (w + 3)) * K4d(m, c, 3, 3);

                if (c >= currM && c < nextM && (h + 3) >= currH && (h + 3) < nextH && (w + 4) >= currW && (w + 4) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 3][t3 + 4] * K4d(m, c, 3, 4);
                else
                    y[bmhw] += x4d(b, c, (h + 3), (w + 4)) * K4d(m, c, 3, 4);

                if (c >= currM && c < nextM && (h + 4) >= currH && (h + 4) < nextH && (w + 0) >= currW && (w + 0) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 4][t3 + 0] * K4d(m, c, 4, 0);
                else
                    y[bmhw] += x4d(b, c, (h + 4), (w + 0)) * K4d(m, c, 4, 0);

                if (c >= currM && c < nextM && (h + 4) >= currH && (h + 4) < nextH && (w + 1) >= currW && (w + 1) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 4][t3 + 1] * K4d(m, c, 4, 1);
                else
                    y[bmhw] += x4d(b, c, (h + 4), (w + 1)) * K4d(m, c, 4, 1);

                if (c >= currM && c < nextM && (h + 4) >= currH && (h + 4) < nextH && (w + 2) >= currW && (w + 2) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 4][t3 + 2] * K4d(m, c, 4, 2);
                else
                    y[bmhw] += x4d(b, c, (h + 4), (w + 2)) * K4d(m, c, 4, 2);

                if (c >= currM && c < nextM && (h + 4) >= currH && (h + 4) < nextH && (w + 3) >= currW && (w + 3) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 4][t3 + 3] * K4d(m, c, 4, 3);
                else
                    y[bmhw] += x4d(b, c, (h + 4), (w + 3)) * K4d(m, c, 4, 3);

                if (c >= currM && c < nextM && (h + 4) >= currH && (h + 4) < nextH && (w + 4) >= currW && (w + 4) < nextW)
                    y[bmhw] += subTile[c - currM][t2 + 4][t3 + 4] * K4d(m, c, 4, 4);
                else
                    y[bmhw] += x4d(b, c, (h + 4), (w + 4)) * K4d(m, c, 4, 4);
            }
        }
        __syncthreads();
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

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    printf("B: %d\n", B);
    printf("M: %d\n", M);
    printf("C: %d\n", C);
    printf("H: %d\n", H);
    printf("W: %d\n", W);
    printf("K: %d\n", K);
    // Set the kernel dimensions
    dim3 gridDim(ceil(float(M)/float(TILE_WIDTH)),
                 ceil(float(H)/float(TILE_WIDTH)),
                 ceil(float(W)/float(TILE_WIDTH)));
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,TILE_WIDTH);

    cudaMemcpyToSymbol(MASK, w.dptr_, M * C * K * K * sizeof(float));

    // Call the kernel
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,B,M,C,H,W,K);

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