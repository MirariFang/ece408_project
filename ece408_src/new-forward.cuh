#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{
const int TILE_WIDTH = 64;
const int BLOCK_WIDTH = 32;

__global__ void forward_kernel(float *__restrict__ y, const float *__restrict__ x, const float *__restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)
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
    int col = bx * TILE_WIDTH + tx;
    int row = by * TILE_WIDTH + ty;

    const int filterSize = 25; // K * K = 25 (constant)
    int weightLength = C * filterSize;

    float acc[4];
    acc[0] = 0;
    acc[1] = 0;
    acc[2] = 0;
    acc[3] = 0;

    int numIter = ceil(weightLength / (1.0 * TILE_WIDTH));

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int outCol = H_out * W_out;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * outCol) + (i2) * (outCol) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * filterSize) + (i2) * (filterSize) + (i1) * (K) + i0]

    __shared__ float tileMatWUnroll[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileMatXUnroll[TILE_WIDTH][TILE_WIDTH];
    const int incre_col[4] = {0, 32, 0, 32};
    const int incre_row[4] = {0, 0, 32, 32};
    for (int i = 0; i < numIter; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            int tempCol = i * TILE_WIDTH + tx + incre_col[j];
            int tempRow = i * TILE_WIDTH + ty + incre_row[j];
            int W_m = row + incre_row[j];
            int W_c = tempCol / filterSize;
            int W_h = (tempCol % filterSize) / K;
            int W_w = (tempCol % filterSize) % K;
            int X_b = bz;
            int X_c = tempRow / filterSize;
            int X_p = (tempRow % filterSize) / K;
            int X_q = (tempRow % filterSize) % K;
            int X_h = (col+incre_col[j]) / W_out;
            int X_w = (col+incre_col[j]) % W_out;
            if (tempCol < weightLength && (row + incre_row[j]) < M)
                tileMatWUnroll[ty + incre_row[j]][tx + incre_col[j]] = k4d(W_m, W_c, W_h, W_w);
            else
                tileMatWUnroll[ty + incre_row[j]][tx + incre_col[j]] = 0;
            if (tempRow < weightLength && (col+incre_col[j]) < outCol)
                tileMatXUnroll[ty + incre_row[j]][tx + incre_col[j]] = x4d(X_b, X_c, (X_h + X_p), (X_w + X_q));
            else
                tileMatXUnroll[ty + incre_row[j]][tx + incre_col[j]] = 0;
        }
        __syncthreads();

        for (int j = 0; j < 4; j++)
        {
            acc[j] += tileMatWUnroll[ty + incre_row[j]][0] * tileMatXUnroll[0][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][1] * tileMatXUnroll[1][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][2] * tileMatXUnroll[2][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][3] * tileMatXUnroll[3][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][4] * tileMatXUnroll[4][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][5] * tileMatXUnroll[5][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][6] * tileMatXUnroll[6][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][7] * tileMatXUnroll[7][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][8] * tileMatXUnroll[8][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][9] * tileMatXUnroll[9][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][10] * tileMatXUnroll[10][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][11] * tileMatXUnroll[11][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][12] * tileMatXUnroll[12][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][13] * tileMatXUnroll[13][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][14] * tileMatXUnroll[14][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][15] * tileMatXUnroll[15][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][16] * tileMatXUnroll[16][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][17] * tileMatXUnroll[17][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][18] * tileMatXUnroll[18][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][19] * tileMatXUnroll[19][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][20] * tileMatXUnroll[20][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][21] * tileMatXUnroll[21][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][22] * tileMatXUnroll[22][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][23] * tileMatXUnroll[23][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][24] * tileMatXUnroll[24][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][25] * tileMatXUnroll[25][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][26] * tileMatXUnroll[26][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][27] * tileMatXUnroll[27][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][28] * tileMatXUnroll[28][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][29] * tileMatXUnroll[29][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][30] * tileMatXUnroll[30][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][31] * tileMatXUnroll[31][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][32] * tileMatXUnroll[32][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][33] * tileMatXUnroll[33][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][34] * tileMatXUnroll[34][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][35] * tileMatXUnroll[35][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][36] * tileMatXUnroll[36][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][37] * tileMatXUnroll[37][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][38] * tileMatXUnroll[38][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][39] * tileMatXUnroll[39][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][40] * tileMatXUnroll[40][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][41] * tileMatXUnroll[41][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][42] * tileMatXUnroll[42][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][43] * tileMatXUnroll[43][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][44] * tileMatXUnroll[44][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][45] * tileMatXUnroll[45][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][46] * tileMatXUnroll[46][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][47] * tileMatXUnroll[47][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][48] * tileMatXUnroll[48][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][49] * tileMatXUnroll[49][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][50] * tileMatXUnroll[50][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][51] * tileMatXUnroll[51][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][52] * tileMatXUnroll[52][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][53] * tileMatXUnroll[53][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][54] * tileMatXUnroll[54][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][55] * tileMatXUnroll[55][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][56] * tileMatXUnroll[56][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][57] * tileMatXUnroll[57][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][58] * tileMatXUnroll[58][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][59] * tileMatXUnroll[59][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][60] * tileMatXUnroll[60][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][61] * tileMatXUnroll[61][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][62] * tileMatXUnroll[62][tx + incre_col[j]];
            acc[j] += tileMatWUnroll[ty + incre_row[j]][63] * tileMatXUnroll[63][tx + incre_col[j]];
        }
        __syncthreads();

        for (int j = 0; j < 4; j++)
        {
            int Y_b = bz;
            int Y_m = (row+incre_col[j]);
            int Y_h = (col+incre_row[j]) / W_out;
            int Y_w = (col+incre_row[j]) % W_out;
            if ((row+incre_col[j]) < M && (col+incre_row[j]) < outCol)
            {
                y4d(Y_b, Y_m, Y_h, Y_w) = acc[j];
            }
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

    // printf("B: %d\n", B);
    // printf("M: %d\n", M);
    // printf("C: %d\n", C);
    // printf("H: %d\n", H);
    // printf("W: %d\n", W);
    // printf("K: %d\n", K);

    // Set the kernel dimensions
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    dim3 gridDim(ceil(H_out * W_out / (1.0 * TILE_WIDTH)),
                 ceil(M / (1.0 * TILE_WIDTH)),
                 B);
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

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