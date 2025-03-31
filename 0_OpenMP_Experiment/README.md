# OpenMP Experiment

This directory contains a simple experiment demonstrating parallelized array multiplication using OpenMP.

## Files

- `main.cpp` - C++ program that multiplies two large arrays using OpenMP and measures performance.
- `build_and_run.sh` - Shell script to compile and execute the program.

## Requirements

- GCC or Clang with OpenMP support.

## Compilation & Execution

Run the following command:

```sh
./build_and_run.sh
```

## Description

The program initializes two large arrays and performs element-wise multiplication using OpenMP with a configurable number of threads. It measures execution time over multiple trials to determine peak performance.

## Experiment

Modify `NUMT` to compare performance between different thread counts:

- `#define NUMT 1` (Single-threaded execution)
- `#define NUMT 4` (Multi-threaded execution)

Observe the speedup achieved by increasing the number of threads.
