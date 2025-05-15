#!/bin/bash

# Output CSV header
echo "Trials, Threads, MegaTrialsPerSecond, Probability" > performance_data.csv

# Define the number of threads per block to test
THREAD_COUNTS=(8 16 32 64 128 256)

# Define the number of trials to test
TRIAL_COUNTS=(1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304)

# Run tests for all combinations
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
