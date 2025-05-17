#!/bin/bash

# Output CSV header
echo "ArraySize, WorkElements, MegaPointsProcessedPerSecond" > performance_data.csv

# Define the different array sizes to test
ARRAY_SIZES=(4096 16384 65536 262144 1048576 4194304)

# Define the number of work elements to test
WORK_ELEMENTS=(8 16 32 64 128 256)

# Run tests for all combinations
for arrs in "${ARRAY_SIZES[@]}"; do
    for numworkelems in "${WORK_ELEMENTS[@]}"; do
        echo "Running with array size of $arrs and $numworkelems number of work elements..."

        # Compile with specific array size and number of work elements
        g++ -DDATASIZE=$arrs -DLOCALSIZE=$numworkelems -o main main.cpp /usr/local/apps/cuda/10.1/lib64/libOpenCL.so.1 -lm -fopenmp

        # Run and append the performance data to the CSV file
        ./main >> performance_data.csv 2>&1
    done
done

echo "All tests completed! Results saved in performance_data.csv"
