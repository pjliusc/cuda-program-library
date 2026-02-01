#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void convolutionGPU(float *image_input, float *kernel, float *output, int input_size, int kernel_size, int stride) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int output_size = (input_size - kernel_size) / stride + 1;
  if (r < output_size && c < output_size) {
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
      for (int j = 0; j < kernel_size; j++)  {
        sum += image_input[(r * stride + i) * input_size + (c * stride + j)] * kernel[i * kernel_size + j];
      }
    }
    output[r * output_size + c] = sum;
  }
}

int main(int argc, char **argv) {
  // commandline inputs: M, N, stride
  int M = (argc > 3) ? atoi(argv[1]) : 7;
  int N = (argc > 3) ? atoi(argv[2]) : 3;
  int stride = (argc > 3) ? atoi(argv[3]) : 1;
  size_t image_size = M * M * sizeof(float);
  size_t kernel_size = N * N * sizeof(float);
  int R = (M - N) / stride + 1;
  size_t output_size = R * R * sizeof(float);

  // host memory allocation
  float *h_image_input = (float *)malloc(image_size);
  float *h_kernel = (float *)malloc(kernel_size);
  float *h_output = (float *)malloc(output_size);

  // initialize matrices with random values
  for (int i = 0; i < M * M; i++) {
    h_image_input[i] = rand() % 100 / 100.0f;
  }
  for (int i = 0; i < N * N; i++) {
    h_kernel[i] = rand() % 100 / 100.0f;
  }

  // device memory
  float *d_image_input, *d_kernel, *d_output;
  cudaMalloc(&d_image_input, image_size);
  cudaMalloc(&d_kernel, kernel_size);
  cudaMalloc(&d_output, output_size);

  // copy inputs to device
  cudaMemcpy(d_image_input, h_image_input, image_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice);

  // gpu launch configuration
  dim3 block(16, 16);
  dim3 grid(
      (R + block.x - 1) / block.x,
      (R + block.y - 1) / block.y
  );

  // cuda timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // launch function then synchronize timing with CPU
  convolutionGPU<<<grid, block>>>(d_image_input, d_kernel, d_output, M, N, stride);
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms;
  cudaEventElapsedTime(&elapsed_ms, start, stop);

  // copy results back to cpu
  cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);


  printf("CUDA execution time (M=%d, N=%d, stride=%d): %f seconds\n", M, N, stride, elapsed_ms/1000.0f);

  // cleanup
  free(h_image_input); free(h_kernel); free(h_output);
  cudaFree(d_image_input); cudaFree(d_kernel); cudaFree(d_output);

  return 0;
}
