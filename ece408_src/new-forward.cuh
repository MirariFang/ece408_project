#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

const int TILE_WIDTH_M = 32;
const int TILE_WIDTH_N = 16;
const int STEPS = TILE_WIDTH_M / TILE_WIDTH_N;
__global__ void forward_kernel(float *__restrict__ y, const float *__restrict__ x, const float *__restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int tx = threadIdx.x;
    int row = by * TILE_WIDTH_M + tx;
    const int filterSize = 25; // K * K = 25 (constant)
    int weightLength = C * filterSize;
    int numIter = ceil(weightLength / (1.0 * STEPS));
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int outCol = H_out * W_out;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * outCol) + (i2) * (outCol) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * filterSize) + (i2) * (filterSize) + (i1) * (K) + i0]

    float tileMatWUnroll[STEPS] = {};
    float acc[TILE_WIDTH_N] = {};
    __shared__ float tileMatXUnroll[STEPS][TILE_WIDTH_N];

    for (int i = 0; i < numIter; i++)
    {
        for (int j = 0; j < STEPS; j++)
        {
            int tempCol = i * STEPS + j;
            int W_m = row;
            int W_c = tempCol / filterSize;
            int W_h = (tempCol % filterSize) / K;
            int W_w = (tempCol % filterSize) % K;
            if (tempCol < weightLength && row < M)
                tileMatWUnroll[j] = k4d(W_m, W_c, W_h, W_w);
            else
                tileMatWUnroll[j] = 0;
        }

        int load_ty = tx / TILE_WIDTH_N;
        int load_tx = tx % TILE_WIDTH_N;
        int col = bx * TILE_WIDTH_N + load_tx;
        int tempRow = i * STEPS + load_ty;
        int X_b = bz;
        int X_c = tempRow / filterSize;
        int X_p = (tempRow % filterSize) / K;
        int X_q = (tempRow % filterSize) % K;
        int X_h = col / W_out;
        int X_w = col % W_out;
        if (tempRow < weightLength && col < outCol)
            tileMatXUnroll[load_ty][load_tx] = x4d(X_b, X_c, (X_h + X_p), (X_w + X_q));
        else
            tileMatXUnroll[load_ty][load_tx] = 0;

        __syncthreads();

        for (int j = 0; j < TILE_WIDTH_N; j++)
        {
            /*Number of steps */
            acc[j] += tileMatWUnroll[0] * tileMatXUnroll[0][j];
            acc[j] += tileMatWUnroll[1] * tileMatXUnroll[1][j];
        }
        __syncthreads();
    }

    for (int j = 0; j < TILE_WIDTH_N; j++)
    {
        int col = bx * TILE_WIDTH_N + j;
        int Y_b = bz;
        int Y_m = row;
        int Y_h = col / W_out;
        int Y_w = col % W_out;
        if (row < M && col < outCol)
        {
            y4d(Y_b, Y_m, Y_h, Y_w) = acc[j];
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
    dim3 gridDim(ceil(H_out * W_out / (1.0 * TILE_WIDTH_N)),
                 ceil(M / (1.0 * TILE_WIDTH_M)),
                 B);
    dim3 blockDim(TILE_WIDTH_M, 1, 1);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);

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
    CHECK_EQ(0, 1) << "Remove this line and replace it with your implementation.";
}
} // namespace op
} // namespace mxnet

#endif