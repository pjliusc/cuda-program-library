import ctypes
import numpy as np
import time
from datasets import load_dataset
import matplotlib.pyplot as plt
from math import ceil, floor
from PIL import Image

def main():
    # Load shared library
    lib = ctypes.cdll.LoadLibrary("./conv_c.so")

    # define function argument datatypes 
    lib.conv.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"), 
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]

    lib.sobel3.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]

    M = 28 # an input fashion mnist image is 28x28
    N = 3
    result_size = M - ceil(N*1.0/2)
    lib.conv.restype = ctypes.POINTER(ctypes.c_float*(result_size**2))
    lib.sobel3.restype = ctypes.POINTER(ctypes.c_float*(result_size**2))

    # Load fashion_mnist dataset
    data = load_dataset("zalando-datasets/fashion_mnist")

    # Get a small subset of images for demo purposes
    train_dataset = data["train"][:10]
    sample_set = []
    for i in range(0, len(train_dataset["image"])):
        img = train_dataset["image"][i]
        plt.imshow(img)
        sample_set.append(np.array(img).astype('float32'))
        img.save(f'data/input/{i}.png')

    print(sample_set[0])

    ident = np.array(
        [[0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]]
    ).astype('float32')

    # run identity test with stride 1
    print("Running identity test with stride 1..")
    for i in range(0, len(sample_set)):
        img_array = sample_set[i]
        temp = lib.conv(img_array.ravel(), ident.ravel(), M, N, 1).contents
        result = np.array([i for i in temp]).reshape(result_size, result_size)

        img = Image.fromarray(result.astype('uint8'))
        img.show()
        img.save(f"data/output/ident/{i}.png")

    # run sobel test
    print("Running sobel filter..")
    for i in range(0, len(sample_set)):
        img_array = sample_set[i]
        temp = lib.sobel3(img_array.ravel(), M, N, 1).contents
        result = np.array([i for i in temp]).reshape(result_size, result_size)

        img = Image.fromarray(result.astype('uint8'))
        img.show()
        img.save(f"data/output/sobel/{i}.png")
        
    # run identity test with stride 2
    print("Running identity test with stride 2..")

    # adjust result size for stride 2
    result_size = floor((M-N)/2.0) + 1
    lib.conv.restype = ctypes.POINTER(ctypes.c_float*(result_size**2))
    
    for i in range(0, len(sample_set)):
        img_array = sample_set[i]
        temp = lib.conv(img_array.ravel(), ident.ravel(), M, N, 2).contents
        result = np.array([i for i in temp]).reshape(result_size, result_size)
        print(result)
        img = Image.fromarray(result.astype('uint8'))
        img.show()
        img.save(f"data/output/ident_2/{i}.png")

if __name__ == "__main__":
    main()