#!/bin/bash

# Output CSV header
echo "Threads,Trials,Probability,MegaTrialsPerSecond" > performance_data.csv

# Define the number of threads to test
THREAD_COUNTS=(1 2 4 6 8)

# Define the number of trials to test
TRIAL_COUNTS=(100000 500000 1000000 5000000 10000000)

# Run tests for all combinations
for numt in "${THREAD_COUNTS[@]}"; do
    for numtrials in "${TRIAL_COUNTS[@]}"; do
        echo "Running with $numt threads and $numtrials trials..."
        
        # Compile with specific thread and trial counts
        g++ -fopenmp -DNUMT=$numt -DNUMTRIALS=$numtrials -o main main.cpp
        
        # Run and append the performance data to the CSV file
        ./main >> performance_data.csv 2>&1
    done
done

echo "All tests completed! Results saved in performance_data.csv"
