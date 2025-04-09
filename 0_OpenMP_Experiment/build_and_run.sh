#!/bin/bash

# Build with OpenMP enabled
g++ -fopenmp main.cpp -o main

# Run the program
./main
