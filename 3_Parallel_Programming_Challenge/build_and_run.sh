#!/bin/bash

# Create output files with headers
echo "NUMT,NUMCITIES,NUMCAPITALS,MegaCityCapitalsPerSecond" > results.csv
echo "NUMT,NUMCITIES,NUMCAPITALS,MegaCityCapitalsPerSecond" > extra_results.csv

# Run tests for different combinations of threads and capitals
for t in 1 2 4 6 8
do
    for n in 2 3 4 5 10 15 20 30 40 50
        do
        echo "Running with NUMT=$t, NUMCAPITALS=$n"
        g++ main.cpp -DNUMT=$t -DNUMCAPITALS=$n -o main -fopenmp -lm
        ./main >> extra_results.csv 2>> results.csv
    done
done

echo "Testing complete. Results saved to results.csv and extra_results.csv"
