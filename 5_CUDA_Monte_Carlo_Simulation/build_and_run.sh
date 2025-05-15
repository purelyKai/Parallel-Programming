#!/bin/bash

for t in 1024 4096 16384 65536 262144 1048576 2097152
do
    for b in 8 32 64 128 256
    do
        /usr/local/apps/cuda/cuda-10.1/bin/nvcc -DNUMTRIALS=$t -DBLOCKSIZE=$b -o main main.cu
        ./main
    done
done
