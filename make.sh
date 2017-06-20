#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd utils
python build.py build_ext --inplace
cd ../

cd layers/reorg/src
echo "Compiling reorg layer kernels by nvcc..."
#nvcc -c -o reorg_cuda_kernel.cu.o reorg_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
nvcc -I/usr/local/cuda/include  -c -o reorg_cuda_kernel.cu.o reorg_cuda_kernel.cu \
-x cu -Xcompiler -fPIC -arch=sm_52
cd ../
python build.py
cd ../

cd roi_pooling/src/cuda
echo "Compiling roi_pooling kernels by nvcc..."
nvcc -I/usr/local/cuda/include -c -o roi_pooling_kernel.cu.o roi_pooling_kernel.cu \
-x cu -Xcompiler -fPIC -arch=sm_52
cd ../../
python build.py
cd ../
