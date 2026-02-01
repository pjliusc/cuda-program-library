#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// multiply two NxN matrices elementwise and sum up the products
float elmWise(float *A, float *B, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * B[i * N + j];
        }
    }
    return sum;
}

// get a square-shaped slice of matrix 
float* getMatSquare(float *A, int start_y, int start_x, int M, int N) {
    size_t size = N * N * sizeof(float);
    float *S = (float *)malloc(size);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            S[i * N + j] = A[(i + start_y) * M + (j + start_x)];
        }
    }
    return S;
}

// run convolution
extern float* conv(float *A, float *B, int M, int N, int stride) {

    int R = (int)(floor((((float)M-N)/stride))) + 1;   
    if (stride == 1) {
        int R = (int)(M - ceil((float)N/2));
    }

    size_t sizeR = pow(R, 2) * sizeof(float);
    float *C = (float *)malloc(sizeR);
    for (int i = 0; i < R; i += stride) {
        for (int j = 0; j < R; j += stride) {
            float *slice = getMatSquare(A, i*stride, j*stride, M, N);
            C[i*R + j] = elmWise(slice, B, N);                  
        }
    }
    return C;
}

// 3x3 sobel filter function
float sobel_3x3(float *A) {
    float G_x[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float G_y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

    float res = sqrt(pow(elmWise(G_x, A, 3), 2) + pow(elmWise(G_y, A, 3), 2))/4.0;
    return res;
}

// full 3x3 sobel filter on a matrix
extern float* sobel3(float *A, int M, int stride) {
    int R = M - 2;
    size_t sizeR = pow(R, 2) * sizeof(float);

    float *C = (float *)malloc(sizeR);

    int max_result_size = M - 2;
    for (int i = 0; i < max_result_size; i += stride) {
        for (int j = 0; j < max_result_size; j += stride) {
            float *slice = getMatSquare(A, i, j, M, 3);
                C[i*max_result_size + j] = sobel_3x3(slice);                    
        }
    }

    return C;
}