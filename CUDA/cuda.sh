#!/bin/bash

# CUDA KNN Script - Multiple Configuration Testing
# This script compiles and runs the CUDA KNN implementation with different block configurations

# Configuration
TRAIN_FILE="iris_train.csv"
TEST_FILE="iris_test.csv"
K_VALUE=10
OUTPUT_BINARY="knn_cuda"
NUM_ITERATIONS=3

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Block configurations to test (X, Y)
BLOCK_CONFIGS=(
    "4 4"       # 16 threads
    "8 8"       # 64 threads
    "16 16"     # 256 threads
    "32 32"     # 1024 threads
    "8 4"       # 32 threads
    "16 8"      # 128 threads
    "32 16"     # 512 threads
    "64 4"      # 256 threads
    "128 2"     # 256 threads
    "4 8"       # 32 threads
    "8 16"      # 128 threads
    "16 32"     # 512 threads
    "4 64"      # 256 threads
    "2 128"     # 256 threads
)

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}            CUDA KNN Classification - Performance Testing            ${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: nvcc (CUDA compiler) not found!${NC}"
    echo "Please install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Check CUDA version
echo -e "${YELLOW}CUDA Compiler Version:${NC}"
nvcc --version | grep "release"
echo ""

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}GPU Information:${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
fi

# Check if source file exists
if [ ! -f "knn_cuda.cu" ]; then
    echo -e "${RED}Error: knn_cuda.cu not found!${NC}"
    exit 1
fi

# Check if training data exists
if [ ! -f "$TRAIN_FILE" ]; then
    echo -e "${RED}Error: Training file '$TRAIN_FILE' not found!${NC}"
    exit 1
fi

# Check if test data exists
if [ ! -f "$TEST_FILE" ]; then
    echo -e "${RED}Error: Test file '$TEST_FILE' not found!${NC}"
    exit 1
fi

# Compile the CUDA program
echo -e "${YELLOW}Compiling CUDA program...${NC}"
nvcc -O3 -o $OUTPUT_BINARY knn_cuda.cu -lm

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Compilation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Compilation successful!${NC}"
echo ""

# Create results file
RESULTS_FILE="cuda_results_k${K_VALUE}.csv"
echo "Config,Block_X,Block_Y,Threads_Per_Block,Iteration,Execution_Time_ms,Accuracy" > $RESULTS_FILE

echo -e "${CYAN}======================================================================${NC}"
echo -e "${CYAN}Testing ${#BLOCK_CONFIGS[@]} configurations with K=$K_VALUE (${NUM_ITERATIONS} iterations each)${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo ""

config_num=0
total_configs=${#BLOCK_CONFIGS[@]}

# Track baseline for speedup calculation
baseline_time=""

for config in "${BLOCK_CONFIGS[@]}"; do
    config_num=$((config_num + 1))
    read BLOCK_X BLOCK_Y <<< "$config"
    THREADS=$((BLOCK_X * BLOCK_Y))
    
    echo -e "${BLUE}[$config_num/$total_configs] Configuration: ${BLOCK_X}x${BLOCK_Y} (${THREADS} threads)${NC}"
    echo "--------------------------------------------------------------------"
    
    declare -a times
    declare -a accuracies
    
    for iter in $(seq 1 $NUM_ITERATIONS); do
        echo -e "  Iteration $iter/$NUM_ITERATIONS..."
        
        # Run the program and capture output
        OUTPUT=$(./$OUTPUT_BINARY $TRAIN_FILE $TEST_FILE $K_VALUE $BLOCK_X $BLOCK_Y 2>&1)
        
        # Extract execution time and accuracy using grep and awk
        TIME=$(echo "$OUTPUT" | grep "RESULTS:" | awk -F'Time=' '{print $2}' | awk '{print $1}')
        ACCURACY=$(echo "$OUTPUT" | grep "RESULTS:" | awk -F'Accuracy=' '{print $2}' | awk -F'%' '{print $1}')
        
        if [ -n "$TIME" ] && [ -n "$ACCURACY" ]; then
            times+=($TIME)
            accuracies+=($ACCURACY)
            echo "${BLOCK_X}x${BLOCK_Y},$BLOCK_X,$BLOCK_Y,$THREADS,$iter,$TIME,$ACCURACY" >> $RESULTS_FILE
            echo -e "    Time: ${TIME} ms, Accuracy: ${ACCURACY}%"
        else
            echo -e "    ${RED}Failed to extract results${NC}"
        fi
    done
    
    # Calculate average
    if [ ${#times[@]} -gt 0 ]; then
        avg_time=$(printf '%s\n' "${times[@]}" | awk '{sum+=$1} END {print sum/NR}')
        avg_accuracy=$(printf '%s\n' "${accuracies[@]}" | awk '{sum+=$1} END {print sum/NR}')
        
        # Calculate standard deviation
        std_dev=$(printf '%s\n' "${times[@]}" | awk -v avg=$avg_time '{sum+=($1-avg)^2} END {print sqrt(sum/NR)}')
        
        # Set baseline from first config
        if [ -z "$baseline_time" ]; then
            baseline_time=$avg_time
            speedup="1.000"
        else
            speedup=$(echo "scale=3; $baseline_time / $avg_time" | bc)
        fi
        
        echo -e "${GREEN}  Average: ${avg_time} ± ${std_dev} ms | Accuracy: ${avg_accuracy}% | Speedup: ${speedup}x${NC}"
    fi
    
    echo ""
    unset times
    unset accuracies
done

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}                    Performance Testing Complete                      ${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo -e "${YELLOW}Results saved to: $RESULTS_FILE${NC}"
echo ""

# Generate summary
echo -e "${CYAN}Summary Statistics:${NC}"
echo "--------------------------------------------------------------------"

# Find best and worst configurations (skip header line)
best_config=$(tail -n +2 $RESULTS_FILE | awk -F',' '{sum[$1]+=$6; count[$1]++} END {for (c in sum) print sum[c]/count[c], c}' | sort -n | head -1)
worst_config=$(tail -n +2 $RESULTS_FILE | awk -F',' '{sum[$1]+=$6; count[$1]++} END {for (c in sum) print sum[c]/count[c], c}' | sort -n | tail -1)

echo -e "${GREEN}Best Configuration:${NC}"
echo "  $(echo $best_config | awk '{print $2}') - Avg Time: $(echo $best_config | awk '{printf "%.4f", $1}') ms"

echo -e "${RED}Worst Configuration:${NC}"
echo "  $(echo $worst_config | awk '{print $2}') - Avg Time: $(echo $worst_config | awk '{printf "%.4f", $1}') ms"

# Calculate improvement
improvement=$(echo "scale=2; ($(echo $worst_config | awk '{print $1}') - $(echo $best_config | awk '{print $1}')) / $(echo $worst_config | awk '{print $1}') * 100" | bc)
echo ""
echo -e "${CYAN}Performance Improvement: ${improvement}%${NC}"
echo -e "${CYAN}(Best config is ${improvement}% faster than worst config)${NC}"
echo ""
echo -e "${GREEN}======================================================================${NC}"
