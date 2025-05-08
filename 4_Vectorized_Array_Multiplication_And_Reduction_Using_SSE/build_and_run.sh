#!/bin/bash

# Set implementation to:
# "assembly" for inline assembly (no flags)
# "sse" for SSE intrinsics (-DUSE_INTRINSICS)
# "avx" for AVX/AVX2 intrinsics (-DUSE_AVX -mavx2)
IMPLEMENTATION="sse"

# Set compiler flags based on implementation
if [ "$IMPLEMENTATION" = "avx" ]; then
    IMPL_NAME="AVX"
    COMPILE_FLAGS="-DUSE_AVX -mavx2"
    echo "Running with AVX/AVX2 Intrinsics implementation"
elif [ "$IMPLEMENTATION" = "sse" ]; then
    IMPL_NAME="SSE"
    COMPILE_FLAGS="-DUSE_INTRINSICS"
    echo "Running with SSE Intrinsics implementation"
else
    IMPL_NAME="ASM"
    COMPILE_FLAGS=""
    echo "Running with inline Assembly implementation"
fi

# Array sizes from 1K to 8M
SIZES=(1024 4096 16384 65536 262144 1048576 4194304 8388608)

# Create results file with headers
RESULTS_FILE="results_${IMPL_NAME}.csv"
echo "ArraySize,NonSimdMul(MM/sec),SimdMul(MM/sec),SpeedupMul,NonSimdMulSum(MM/sec),SimdMulSum(MM/sec),SpeedupMulSum" > $RESULTS_FILE

# Run the experiment for each array size
for size in "${SIZES[@]}"
do
    echo "Testing array size: $size"
    g++ -DARRAYSIZE=$size $COMPILE_FLAGS -o main main.cpp -fopenmp
    
    # Run the program and capture the output
    ./main 2>> $RESULTS_FILE
done

echo "Results saved to $RESULTS_FILE"

# Create visualization using gnuplot
PLOT_FILE="plot_${IMPL_NAME}.png"
cat > plot_script.plt << EOL
set terminal pngcairo enhanced size 1200,800
set output "$PLOT_FILE"
set title "SIMD/non-SIMD Speed-up vs. Array Size ($IMPL_NAME implementation)"
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
plot '$RESULTS_FILE' using 1:4 title "SpeedupMul" with lines lw 2, \
     '$RESULTS_FILE' using 1:7 title "SpeedupMulSum" with lines lw 2
EOL

# Run gnuplot
gnuplot plot_script.plt

echo "Visualization of SIMD/non-SIMD Speed-up saved to $PLOT_FILE"
