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

int main() {
    const int image_sizes[]  = {28, 56, 112};
    const int kernel_sizes[] = {3, 5, 7};
    const int stride = 1;
    const int runs = 10;   // 10 images

    dim3 block(16, 16);

    printf("Runtime per configuration (10 images each)\n");
    printf("Image\tKernel\tTime (ms)\n");
    printf("---------------------------\n");

    float final_ms = 0.0f;

    for (int im = 0; im < 3; im++) {
        for (int k = 0; k < 3; k++) {

            int M = image_sizes[im];
            int N = kernel_sizes[k];
            int R = (M - N) / stride + 1;

            size_t image_bytes  = M * M * sizeof(float);
            size_t kernel_bytes = N * N * sizeof(float);
            size_t output_bytes = R * R * sizeof(float);

            float *h_image  = (float *)malloc(image_bytes);
            float *h_kernel = (float *)malloc(kernel_bytes);
            float *h_output = (float *)malloc(output_bytes);

            for (int i = 0; i < M * M; i++)
                h_image[i] = rand() / (float)RAND_MAX;

            for (int i = 0; i < N * N; i++)
                h_kernel[i] = rand() / (float)RAND_MAX;

            dim3 grid((R + block.x - 1) / block.x,
                      (R + block.y - 1) / block.y);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            float total_ms = 0.0f;

            for (int r = 0; r < runs; r++) {

                float *d_image, *d_kernel, *d_output;

                cudaEventRecord(start);

                CHECK(cudaMalloc(&d_image, image_bytes));
                CHECK(cudaMalloc(&d_kernel, kernel_bytes));
                CHECK(cudaMalloc(&d_output, output_bytes));

                CHECK(cudaMemcpy(d_image, h_image, image_bytes, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice));

                convolutionGPU<<<grid, block>>>(d_image, d_kernel, d_output, M, N, stride);
                CHECK(cudaDeviceSynchronize());

                CHECK(cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost));

                CHECK(cudaFree(d_image));
                CHECK(cudaFree(d_kernel));
                CHECK(cudaFree(d_output));

                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                float ms;
                cudaEventElapsedTime(&ms, start, stop);
                total_ms += ms;
            }

            printf("%d\t%d\t%.4f\n", M, N, total_ms);
            final_ms += total_ms;

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            free(h_image);
            free(h_kernel);
            free(h_output);
        }
    }

    printf("Total runtime (90 images total): %.4f ms\n", final_ms);
    return 0;
}
