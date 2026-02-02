#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h> 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define TILE_WIDTH 16 

// CPU 
void matrixMultiplyCPU(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}


//  Naive CUDA 

__global__ void matrixMultiplyNaive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Optimized CUDA

__global__ void matrixMultiplyOptimized(float* A, float* B, float* C, int N) {
    
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0;

    
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
       
        if (Row < N && (m * TILE_WIDTH + tx) < N)
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;

        if (Col < N && (m * TILE_WIDTH + ty) < N)
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();

       
        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];

        __syncthreads(); 
    }

    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}

void runTest(int N) {
    printf("\n---\n");
    printf("Testing Matrix Size N = %d\n", N);
    size_t size = N * N * sizeof(float);

   
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C_Ref = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 100) / 100.0f;
        h_B[i] = (float)(rand() % 100) / 100.0f;
    }

    float* d_A = 0;
    float* d_B = 0;
    float* d_C = 0;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  
    float cpu_ms = 0, naive_ms = 0, opt_ms = 0, cublas_ms = 0;
    cudaEvent_t start = 0, stop = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

   
    
        printf("1 Running CPU... ");
        clock_t t1 = clock();
        matrixMultiplyCPU(h_A, h_B, h_C_Ref, N);
        clock_t t2 = clock();
        cpu_ms = (float)(t2 - t1) / CLOCKS_PER_SEC * 1000.0f;
        printf("Done. Time: %.2f ms\n", cpu_ms);
    
    


    dim3 threadsNaive(16, 16);
    dim3 blocksNaive((N + 15) / 16, (N + 15) / 16);

    printf("2 Running Naive CUDA... ");
    cudaEventRecord(start);
    matrixMultiplyNaive << <blocksNaive, threadsNaive >> > (d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&naive_ms, start, stop);
    printf("Done. Time: %.2f ms\n", naive_ms);

    dim3 threadsOpt(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksOpt((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    printf("3 Running Optimized CUDA... ");
    cudaEventRecord(start);


    matrixMultiplyOptimized << <blocksOpt, threadsOpt >> > (d_A, d_B, d_C, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("\n OPTIMIZED KERNEL FAILED: %s \n", cudaGetErrorString(err));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&opt_ms, start, stop);
    printf("Done. Time: %.2f ms\n", opt_ms);

   
    printf("4 Running cuBLAS... ");

  
    cublasHandle_t handle = 0;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    cudaEventRecord(start);

    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_B, N,  
        d_A, N,  
        &beta,
        d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cublas_ms, start, stop);

    cublasDestroy(handle);
    printf("Done. Time: %.2f ms\n", cublas_ms);

    printf("\n--- Performance Summary (N=%d) ---\n", N);
    if (cpu_ms > 0) printf("CPU           : %.2f ms\n", cpu_ms);
    printf("Naive CUDA    : %.2f ms\n", naive_ms);
    printf("Optimized CUDA: %.2f ms\n", opt_ms);
    printf("cuBLAS        : %.2f ms\n", cublas_ms);

    if (cublas_ms > 0 && opt_ms > 0) {
        printf("\n>> Speedup (Optimized vs Naive): %.2fx\n", naive_ms / opt_ms);
        printf(">> Speedup (cuBLAS vs Optimized): %.2fx\n", opt_ms / cublas_ms);
    }


    free(h_A); free(h_B); free(h_C_Ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    srand(2023); 

   
    int sizes[] = { 512, 1024, 2048 ,4096};

    for (int i = 0; i < 4; i++) {
        runTest(sizes[i]);
    }

   
    printf("\nPress Enter to exit...");
    getchar();
    return 0;
}