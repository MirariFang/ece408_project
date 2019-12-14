#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{
const int TILE_WIDTH = 16;

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int column = bx * TILE_WIDTH + tx;
    int row = by * TILE_WIDTH + ty;
    int weightLength = C * K * K;

    float acc = 0;

    int numIter = ceil(weightLength / (1.0 * TILE_WIDTH));

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    __shared__ float tileMatWUnroll[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileMatXUnroll[TILE_WIDTH][TILE_WIDTH];

    for (int i = 0; i < numIter; i++)
    {
        int tempCol = i * TILE_WIDTH + tx;
        int tempRow = i * TILE_WIDTH + ty;
        tileMatWUnroll[ty][tx] = 0;
        tileMatXUnroll[ty][tx] = 0;

        int W_m = row;
        int W_c = tempCol / (K * K);
        int W_h = (tempCol % (K * K)) / K;
        int W_w = (tempCol % (K * K)) % K;

        if (tempCol < weightLength && row < M)
            tileMatWUnroll[ty][tx] = k4d(W_m, W_c, W_h, W_w);
        else
            tileMatWUnroll[ty][tx] = 0;

        int X_b = bz;
        int X_c = tempRow / (K * K);
        int X_p = (tempRow % (K * K)) / K;
        int X_q = (tempRow % (K * K)) % K;
        int X_h = column / W_out;
        int X_w = column % W_out;

        if (tempRow < weightLength && column < H_out * W_out)
        {
            tileMatXUnroll[ty][tx] = x4d(X_b, X_c, (X_h + X_p), (X_w + X_q));
        }
        else
        {
            tileMatXUnroll[ty][tx] = 0;
        }

        __syncthreads();

        for (int q = 0; q < TILE_WIDTH; q++)
        {
            acc += tileMatWUnroll[ty][q] * tileMatXUnroll[q][tx];
        }
        __syncthreads();
        int Y_b = bz;
        int Y_m = row;
        int Y_h = column / W_out;
        int Y_w = column % W_out;
        if (row < M && column < W_out * H_out)
        {
            y4d(Y_b, Y_m, Y_h, Y_w) = acc;
        }
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
    /*
    printf("B: %d\n", B);
    printf("M: %d\n", M);
    printf("C: %d\n", C);
    printf("H: %d\n", H);
    printf("W: %d\n", W);
    printf("K: %d\n", K);
    */
    // Set the kernel dimensions
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    dim3 gridDim(ceil(H_out * W_out / (1.0 * TILE_WIDTH)),
                 ceil(M / (1.0 * TILE_WIDTH)),
                 B);
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_,B,M,C,H,W,K);

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