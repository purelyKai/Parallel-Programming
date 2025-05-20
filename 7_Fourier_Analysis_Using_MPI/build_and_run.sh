#!/bin/bash

# Output CSV header
echo "Processors, Elements, MegaSumsPerSecond" > performance_data.csv

# Compile with MPI and link math library
mpicxx -o main main.cpp -lm

# Number of processors to test
PROCESSORS=(1 2 4 8 16 32)

# Test each processor count
for p in "${PROCESSORS[@]}"; do
    # Run MPI program and append the performance data to the CSV file
    mpirun -np $p ./main >> performance_data.csv 2>&1
done

echo "All tests completed! Results saved in performance_data.csv"
