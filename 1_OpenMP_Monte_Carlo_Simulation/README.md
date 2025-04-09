# OpenMP: Monte Carlo Simulation

This project implements a Monte Carlo simulation to determine the probability of amateur mercenaries successfully hitting a castle with a cannon, using OpenMP for parallel processing.

## Files

- `main.cpp` - C++ program that runs a parallelized Monte Carlo simulation using OpenMP.
- `build_and_run.sh` - Shell script to compile and execute the program with different thread and trial counts.

## Requirements

- GCC or Clang with OpenMP support.

## Compilation & Execution

Run the following command:

```sh
./build_and_run.sh
```

This will compile and run the simulation with various combinations of thread counts (1, 2, 4, 6, 8) and trial counts (1000, 10000, 50000, 100000, 500000, 1000000), collecting performance data in CSV format.

## Description

The program simulates cannonballs being fired at a castle on top of a cliff. Due to the amateur nature of the mercenaries, parameters like ground distance, cliff height, castle distance, initial velocity, and firing angle can only be determined within certain ranges. The simulation calculates the probability of a successful hit using parallel Monte Carlo methods.

For each run, the program:

1. Generates random values for each parameter within specified ranges
2. Simulates the cannonball trajectory using physics equations
3. Determines if the cannonball successfully hits the castle
4. Calculates the probability of success and measures performance in MegaTrials/second

## Analysis

The script generates CSV data for analyzing:

- Performance vs. number of Monte Carlo trials
- Performance vs. number of OpenMP threads
- Parallel fraction (Fp) and maximum theoretical speedup

The results demonstrate how Monte Carlo simulations can be efficiently parallelized across multiple cores, and how performance scales with varying thread counts and problem sizes.
