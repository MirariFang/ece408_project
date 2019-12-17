#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{
const int TILE_WIDTH_N = 4;
const int TILE_WIDTH_M = 8;
const int STEPS = 2;
const int NUM_REG = TILE_WIDTH_M/TILE_WIDTH_N;

/*
Size of the unroll matrix would be C*K*K*H_out*W_out
Size of the x matrix is M*(C*K*K)
*/

__global__ void forward_kernel(float *__restrict__ y, const float *__restrict__ x, const float *__restrict__ k, 
                                const int B, const int M, const int C, const int H, const int W, const int K){

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
int bx = blockIdx.x;
int by = blockIdx.y;
int bz = blockIdx.z;
int tx = threadIdx.x;
int ty = threadIdx.y;
int tz = threadIdx.z;
int gridSize = gridDim;
__shared__ float tileMatXUnroll[STEPS][TILE_WIDTH_N];
int Reg[STEPS];
int numIter = ceil((C*K*K)/STEPS);
int w_row_start;
int w_col_start;
int x_row_start;
int x_col_start;
int y_row_start;
int y_col_start;
for(int i=0;i<numIter;i++){
    /*Load*/
    for(int j=0;j<STEPS;j++)
        Reg[j] = w_row_num+i*(bl);
        tileMatXUnroll[j] = x4d(_, _, x_col_start, x_rol_start);
        
    for(int j=0;j<TILE_WIDTH_N;j++){
        for(int k)
            y4d(Y_b, Y_m, Y_h, Y_w) += Reg[]*tileMatXUnroll[k][];
    }
}

#undef y4d
#undef x4d
#undef k4d
}

/*
const int TILE_WIDTH = 16;
__global__ void forward_kernel(float *__restrict__ y, const float *__restrict__ x, const float *__restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)
{

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
            __syncthreads();
        }

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

*/

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
    // Set the kernel dimensions
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    dim3 gridDim(ceil(H_out * W_out / (1.0 * TILE_WIDTH)),
                 M,
                 B);
    dim3 blockDim(TILE_WIDTH_M,1,1);

    // Call the kernel Assume x is unrolled
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