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

// 
void conv(float *A, float *B, float *C, int M, int N, int stride) {
    //int conv_center = N/2;
    int max_result_size = M - ceil((float)N/2);
    for (int i = 0; i < max_result_size; i += stride) {
        for (int j = 0; j < max_result_size; j += stride) {
            float *slice = getMatSquare(A, i, j, M, N);
            C[i*max_result_size + j] = elmWise(slice, B, N);
        }
    }
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi (argv[1]) : 7; // use matrix size as input
    size_t sizeM = M * M * sizeof(float);
    size_t sizeN = 3 * 3 * sizeof(float);
    int R = (int)(M - ceil((float)3/2));
    size_t sizeR = pow(R, 2) * sizeof(float);

    float *A = (float *)malloc(sizeM);
    float *B = (float *)malloc(sizeN);
    float *C = (float *)malloc(sizeR);

    for (int i = 0; i < M * M; i++) {
        //A[i] = rand() % 100 / 100.0f;
        A[i] = 1.0;
    }

    A[1] = 2.0;
    A[M*7-1] = 3.0;

    for (int i = 0; i < 3 * 3; i++) {
        //B[i] = rand() % 100 / 100.0f;
        B[i] = 0.0;
    }

    B[1] = 1.0;
    B[3] = 1.0;
    B[4] = 1.0;
    B[5] = 1.0;
    B[6] = 1.0;
    B[7] = 1.0;
    B[8] = 1.0;

    clock_t start = clock();
    conv(A, B, C, M, 3, 1);
    clock_t end = clock();

    double elapsed = (double) (end - start) / CLOCKS_PER_SEC;
    printf("CPU execution time (N=%d): %f seconds\n", M, elapsed);

    int valcount = 0;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < R; j++) {
            printf("%f ",C[i*R + j]);
            valcount += 1;
        }
        printf("\n");
    }
    printf("\n valcount: %d", valcount);
    printf("\n R: %d", R);

    free(A); free(B); free(C);
    return 0;
}