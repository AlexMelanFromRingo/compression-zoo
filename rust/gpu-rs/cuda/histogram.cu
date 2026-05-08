// Byte-histogram kernel: count occurrences of each of the 256 byte
// values in a buffer of length n. Used as a building block for SA-IS
// bucket counting and BWT bucket layout.
//
// Strategy: each thread block builds a private histogram in shared
// memory (one u32 per bin = 1 KB per block, no bank conflicts), then
// atomically reduces into the global output. This is the standard
// CUDA pattern.

#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>

extern "C" int gpu_rs_available(void) {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess) return 0;
    return n_dev > 0 ? 1 : 0;
}

#define BINS 256
#define BLOCK_SIZE 256

__global__ void histogram_kernel(const uint8_t *__restrict__ data,
                                 uint64_t n,
                                 uint32_t *__restrict__ out)
{
    __shared__ uint32_t local[BINS];

    // Initialise shared bins.
    int tid = threadIdx.x;
    if (tid < BINS) local[tid] = 0;
    __syncthreads();

    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + tid;
    while (i < n) {
        atomicAdd(&local[data[i]], 1u);
        i += stride;
    }
    __syncthreads();

    // Reduce into global output.
    if (tid < BINS) {
        atomicAdd(&out[tid], local[tid]);
    }
}

// Returns 0 on success, negative cudaError_t on failure.
extern "C" int gpu_rs_histogram_u8(const uint8_t *data,
                                   uint64_t n,
                                   uint32_t out[BINS])
{
    if (n == 0) {
        for (int i = 0; i < BINS; ++i) out[i] = 0;
        return 0;
    }

    uint8_t  *d_data = nullptr;
    uint32_t *d_out  = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_data, n);              if (err) goto fail;
    err = cudaMalloc(&d_out, BINS * sizeof(uint32_t)); if (err) goto fail;
    err = cudaMemcpy(d_data, data, n, cudaMemcpyHostToDevice); if (err) goto fail;
    err = cudaMemset(d_out, 0, BINS * sizeof(uint32_t)); if (err) goto fail;

    {
        // Grid sized so each thread handles ~64 bytes — gives enough
        // work per block to hide global-load latency without
        // saturating the atomic-add traffic at the global reduce.
        uint64_t target_threads = (n + 63) / 64;
        if (target_threads < BLOCK_SIZE) target_threads = BLOCK_SIZE;
        uint64_t grid = (target_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid > 65535) grid = 65535;
        histogram_kernel<<<(int)grid, BLOCK_SIZE>>>(d_data, n, d_out);
        err = cudaGetLastError();
        if (err) goto fail;
        err = cudaDeviceSynchronize();
        if (err) goto fail;
    }

    err = cudaMemcpy(out, d_out, BINS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err) goto fail;

    cudaFree(d_data);
    cudaFree(d_out);
    return 0;

fail:
    if (d_data) cudaFree(d_data);
    if (d_out)  cudaFree(d_out);
    return -(int)err;
}
