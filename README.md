# kmp-mpi-cuda
kmp algorithm mpi and cuda version

To compile mpi app use cmake

In order to compile cuda app type:
nvcc -o kmp_cuda src/kmp_cuda.cu

Both programs take 3 arguments: path to data file, pattern and is_kmp bool flag
