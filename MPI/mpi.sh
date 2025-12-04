#!/bin/bash

echo "Procs | Time (ms) | Speedup | Accuracy"
echo "------|-----------|---------|----------"

base_time=0

process_list="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"

for p in $process_list
do
    output=$(mpirun -np $p --hostfile hosts ./mpi_knn iris_train.csv iris_test.csv 10 2>&1)
    time_ms=$(echo "$output" | grep "Execution time:" | awk '{print $3}')
    accuracy=$(echo "$output" | grep "Accuracy:" | awk '{print $2}' | sed 's/%//')

    if [ "$p" -eq 1 ]; then
        base_time=$time_ms
        speedup=1.00
    else
        speedup=$(awk "BEGIN {printf \"%.2f\", $base_time / $time_ms}")
    fi

    printf "  %2d  | %9s | %7s | %7s%%\n" $p $time_ms $speedup $accuracy
done
