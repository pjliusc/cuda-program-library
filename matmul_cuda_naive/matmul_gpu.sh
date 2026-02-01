nvcc matmul_gpu.cu -o matrix_gpu -O2
./matrix_gpu 512
./matrix_gpu 1024
./matrix_gpu 2048
./matrix_gpu 4096
./matrix_gpu 8192
./matrix_gpu 16384


