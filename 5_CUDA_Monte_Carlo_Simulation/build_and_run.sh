#!/bin/bash

# Output CSV header
echo "Trials, Threads, MegaTrialsPerSecond, Probability" > performance_data.csv

# Define the number of threads per block to test
THREAD_COUNTS=(8 32 64 128 256)

# Define the number of trials to test
TRIAL_COUNTS=(1024 4096 16384 65536 262144 1048576 2097152)

for numt in "${THREAD_COUNTS[@]}"; do
    for numtrials in "${TRIAL_COUNTS[@]}"; do
        echo "Running with $numt threads and $numtrials trials..."

        # Compile with specific thread and trial counts
        nvcc -DNUMTRIALS=$numtrials -DBLOCKSIZE=$numt -o main main.cu

        # Run and append the performance data to the CSV file
        ./main >> performance_data.csv 2>&1
    done
done

echo "All tests completed! Results saved in performance_data.csv"
