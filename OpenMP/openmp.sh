#!/bin/bash

echo "Threads | Time (ms) | Speedup | Accuracy"
echo "--------|-----------|---------|----------"

# Store baseline time
base_time=0

# Test with different thread counts
for threads in 1 2 3 4 5 6 8 16
do
    # Run and capture output (suppress progress output)
    output=$(./knn_openmp iris_train.csv iris_test.csv 10 $threads 2>&1)
    
    # Extract time and accuracy from output
    time_ms=$(echo "$output" | grep "Execution time:" | awk '{print $3}')
    accuracy=$(echo "$output" | grep "Accuracy:" | awk '{print $2}' | sed 's/%//')
    
    # Calculate speedup (compared to 1 thread)
    if [ $threads -eq 1 ]; then
        base_time=$time_ms
        speedup=1.00
    else
        # Use awk for calculations to avoid bc dependency issues
        speedup=$(awk "BEGIN {printf \"%.2f\", $base_time / $time_ms}")
    fi
    
    printf "  %2d    | %9s | %7s | %7s%%\n" $threads $time_ms $speedup $accuracy
done