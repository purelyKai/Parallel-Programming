#!/bin/bash

#SBATCH -J Fourier
#SBATCH -A cs475-575
#SBATCH -p classmpifinal
#SBATCH -N 8 # number of nodes
#SBATCH -n 8 # number of tasks
#SBATCH -o mpiproject.out
#SBATCH -e mpiproject.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=blackkai@oregonstate.edu

module load openmpi

# Output CSV header
# echo "Processors, Elements, MegaSumsPerSecond" > performance_data.csv

# Compile with MPI and link math library
mpic++ -o main main.cpp -lm

# Number of processors to test
PROCESSORS=(1 2 4 8 16 32)

# Test each processor count
for p in "${PROCESSORS[@]}"; do
    echo "Running with $p processors..."

    # Run MPI program and append the performance data to the CSV file
    mpiexec -mca btl self,tcp -np $p ./main # >> performance_data.csv 2>&1
done

echo "All tests completed!" # Results saved in performance_data.csv"
