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

kernel = np.array(
  [[-1, 0, 1],
  [-1, 0, 1],
    [-1, 0, 1]]
).astype('float32')
kernel = np.ascontiguousarray(kernel, dtype=np.float32)

M_list = [28]
N_list = [3]
stride = 1

# main execution loop
start = time.time()
for M in M_list:
  for N in N_list:
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
        img.save(f"data/output/{i}.png")

  print(f"CUDA-based library exec time: {(time.time() - start)*1000}ms")
