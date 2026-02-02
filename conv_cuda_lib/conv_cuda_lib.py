import ctypes
import numpy as np
import time
from datasets import load_dataset
import matplotlib.pyplot as plt
from math import ceil
from PIL import Image

# Load shared library
lib = ctypes.cdll.LoadLibrary("./conv_lib.so")

lib.conv.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"), # input
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"), # kernel
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"), # output
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

# Load fashion_mnist dataset
data = load_dataset("zalando-datasets/fashion_mnist")

# Get a small subset of images for demo purposes
train_dataset = data["train"][:10]
sample_set = []
for i in range(0, len(train_dataset["image"])):
  img = train_dataset["image"][i]
  sample_set.append(np.array(img).astype('float32'))
  img.save(f'data/input/{i}.png')

edge_kernels = {
    3: np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32),

    5: np.array([
        [ 0, -1, -1, -1,  0],
        [-1, -2, -2, -2, -1],
        [-1, -2, 16, -2, -1],
        [-1, -2, -2, -2, -1],
        [ 0, -1, -1, -1,  0]
    ], dtype=np.float32),

    7: np.array([
        [ 0,  0, -1, -1, -1,  0,  0],
        [ 0, -1, -3, -3, -3, -1,  0],
        [-1, -3,  0,  7,  0, -3, -1],
        [-1, -3,  7, 24,  7, -3, -1],
        [-1, -3,  0,  7,  0, -3, -1],
        [ 0, -1, -3, -3, -3, -1,  0],
        [ 0,  0, -1, -1, -1,  0,  0]
    ], dtype=np.float32)
}

vertical_edge_kernels = {
    3: np.array([
        [-1,  0,  1],
        [-1,  0,  1],
        [-1,  0,  1]
    ], dtype=np.float32),

    5: np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-3,  0,  3],
        [-2,  0,  2],
        [-1,  0,  1]
    ], dtype=np.float32),

    7: np.array([
        [-1,  0,  1],
        [-4,  0,  4],
        [-6,  0,  6],
        [-8,  0,  8],
        [-6,  0,  6],
        [-4,  0,  4],
        [-1,  0,  1]
    ], dtype=np.float32)
}

M_list = [28, 56, 112]
N_list = [3, 5, 7]
stride = 1

# main execution loop
start = time.time()
for M in M_list:
  for N in N_list:
    kernel = edge_kernels[N]
    kernel = np.ascontiguousarray(kernel, dtype=np.float32)

    result_size = (M - N) // stride + 1
    output = np.zeros(result_size * result_size, dtype=np.float32)
    lib.conv.restype = None

    results = []
    for i in range(0, len(sample_set)):
        pil_img = Image.fromarray(sample_set[i].astype(np.uint8))
        pil_img = pil_img.resize((M, M), resample=Image.BILINEAR)
        img_array = np.asarray(pil_img, dtype=np.float32)

        output.fill(0)
        lib.conv(img_array.ravel(), kernel.ravel(), output, M, N, stride)
        result = output.reshape(result_size, result_size)

        results.append(result)
        vis = np.abs(result)
        vis = vis / vis.max() * 255
        img = Image.fromarray(vis.astype(np.uint8))
        img.save(f"data/output/{i}_{M}_{N}.png")

# exec time
print(f"CUDA-based library exec time: {(time.time() - start)*1000:.4f}ms")
