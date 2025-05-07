#!/bin/bash

# Set this to true to use Intel intrinsics, false to use assembly
USE_INTRINSICS=true

# Set implementation name for file naming
if $USE_INTRINSICS; then
    COMPILE_FLAGS="-DUSE_INTRINSICS"
    echo "Running with Intel Intrinsics implementation"
else
    COMPILE_FLAGS=""
    echo "Running with inline Assembly implementation"
fi

# Array sizes from 1K to 8M
SIZES=(1024 4096 16384 65536 262144 1048576 4194304 8388608)

# Create results file with headers
echo "ArraySize,NonSimdMul(MM/sec),SimdMul(MM/sec),SpeedupMul,NonSimdMulSum(MM/sec),SimdMulSum(MM/sec),SpeedupMulSum" > results.csv

# Run the experiment for each array size
for size in "${SIZES[@]}"
do
    echo "Testing array size: $size"
    g++ -DARRAYSIZE=$size $COMPILE_FLAGS -o main main.cpp -fopenmp
    
    # Run the program and capture the output
    ./main 2>> results.csv
done

echo "Results saved to results.csv"

# Create visualization using gnuplot
cat > plot_script.plt << EOL
set terminal pngcairo enhanced size 1200,800
set output "plot.png"
set title "SIMD/non-SIMD Speed-up vs. Array Size"
set xlabel "Array Size (elements)"
set ylabel "Speed-Up"
set key outside
set grid
set logscale x 2
set format x "%.0s%c"
set grid mxtics
set datafile separator ","
set style data lines

# Plot the data
plot 'results.csv' using 1:4 title "SpeedupMul" with lines lw 2, \
     'results.csv' using 1:7 title "SpeedupMulSum" with lines lw 2
EOL

# Run gnuplot
gnuplot plot_script.plt

echo "Visualization of SIMD/non-SIMD Speed-up vs. Array Size saved to plot.png"
