#!/bin/bash

# Compile with OpenMP support
g++ -fopenmp -o main main.cpp -lm

# Run the simulation
./main > simulation_data.csv

# Convert data to metric units for better visualization
awk -F, 'BEGIN {OFS=","; print "Month,Year,Temp(C),Precip(cm),Height(cm),Deer,WeedDensity"}
         NR>1 {
           celsius = (5.0/9.0)*($3-32);
           precip_cm = $4 * 2.54;
           height_cm = $5 * 2.54;
           print $1,$2,celsius,precip_cm,height_cm,$6,$7
         }' simulation_data.csv > simulation_metric.csv

# Create visualization using gnuplot
cat > plot_script.plt << EOL
set terminal pngcairo enhanced size 1200,800
set output "simulation_plot.png"
set title "Grain-Deer-Weed Simulation Results"
set xlabel "Month"
set ylabel "Value"
set key outside
set grid
set datafile separator ","
set style data lines
plot 'simulation_metric.csv' using 0:3 title "Temperature (°C)" with lines lw 2, \
     'simulation_metric.csv' using 0:4 title "Precipitation (cm)" with lines lw 2, \
     'simulation_metric.csv' using 0:5 title "Grain Height (cm)" with lines lw 2, \
     'simulation_metric.csv' using 0:6 title "Deer Population" with lines lw 2, \
     'simulation_metric.csv' using 0:7 title "Weed Density" with lines lw 2
EOL

# Run gnuplot
gnuplot plot_script.plt
