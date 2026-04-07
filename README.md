# KNN Optimization Project (Serial, OpenMP, MPI, CUDA)

This project implements the K-Nearest Neighbors (KNN) classification algorithm for the Iris dataset using multiple parallelization strategies and compares their performance.

## What Was Done In This Project

- Built a baseline **serial** KNN implementation in C.
- Built a **shared-memory parallel** version using OpenMP.
- Built a **distributed-memory parallel** version using MPI.
- Built a **GPU-accelerated** version using CUDA.
- Added shell scripts to benchmark multiple configurations:
	- Thread scaling for OpenMP.
	- Process scaling for MPI.
	- CUDA block-size sweeps and repeated runs.
- Added train/test datasets in each folder so each implementation can run independently.

## Core Concepts Used

### 1. KNN Classification

- For each test sample, compute distance to all training samples.
- Select the nearest `k` neighbors.
- Predict class by majority vote.

Distance metric used:

$$
d(\mathbf{x},\mathbf{y}) = \sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
$$

### 2. Parallel Computing Models

- **Serial (CPU):** one process, one execution stream.
- **OpenMP:** multi-threaded execution on a single machine (shared memory).
- **MPI:** multiple processes (can be multi-node), each handling a chunk of test samples.
- **CUDA:** GPU kernel computes distances in parallel; host performs sorting and voting.

### 3. Performance Metrics

- Execution time in milliseconds.
- Classification accuracy.
- Speedup from scripts:

$$
	ext{Speedup} = \frac{T_1}{T_p}
$$

where $T_1$ is baseline time and $T_p$ is parallel time.

## Frameworks, APIs, and Tools Used

- **C (GCC/Clang):** core algorithm implementations.
- **OpenMP (`omp.h`):** shared-memory parallelism.
- **MPI (`mpi.h`, `mpicc`, `mpirun`):** process-level parallelism.
- **CUDA (`nvcc`, `cuda_runtime.h`):** GPU parallel computing.
- **Bash scripting:** automation for benchmark sweeps.

## Project Structure

```
.
├── knn_serial.c
├── iris_train.csv
├── iris_test.csv
├── iris_dataset.csv
├── iris_extended.csv
├── OpenMP/
│   ├── knn_openmp.c
│   ├── openmp.sh
│   ├── iris_train.csv
│   ├── iris_test.csv
│   └── Small-dataset/
├── MPI/
│   ├── knn_mpi.c
│   ├── mpi.sh
│   ├── iris_train.csv
│   └── iris_test.csv
└── CUDA/
		├── knn_cuda.cu
		├── cuda.sh
		├── knn_cuda.ipynb
		├── iris_train.csv
		└── iris_test.csv
```

## Dataset Format

Each CSV row is:

`feature_1,feature_2,...,feature_n,label`

For Iris here:

- 4 numeric features
- 1 integer class label

## Requirements

- C compiler (`gcc` or `clang`)
- Math library (`-lm`)
- OpenMP support (for OpenMP version)
- MPI runtime and compiler (for MPI version)
- NVIDIA GPU + CUDA Toolkit (for CUDA version)

### macOS Notes

- OpenMP with Apple Clang usually requires `libomp`:
	- `brew install libomp`
- MPI can be installed with:
	- `brew install open-mpi`
- CUDA is generally **not supported on Apple Silicon Macs** and most modern macOS setups; run the CUDA part on a Linux/Windows machine with NVIDIA GPU.

## Build and Run Instructions

### 1. Serial Version

From project root:

```bash
gcc -O3 -o knn_serial knn_serial.c -lm
./knn_serial iris_train.csv iris_test.csv 10
```

Arguments:

- `train_file`
- `test_file`
- `k`

If omitted, defaults are used (`iris_train.csv`, `iris_test.csv`, `k=3`).

### 2. OpenMP Version

From `OpenMP` folder (Linux/GCC example):

```bash
cd OpenMP
gcc -O3 -fopenmp -o knn_openmp knn_openmp.c -lm
./knn_openmp iris_train.csv iris_test.csv 10 8
```

Arguments:

- `train_file`
- `test_file`
- `k`
- `thread_count`

Run benchmark sweep:

```bash
cd OpenMP
chmod +x openmp.sh
./openmp.sh
```

### 3. MPI Version

From `MPI` folder:

```bash
cd MPI
mpicc -O3 -o mpi_knn knn_mpi.c -lm
mpirun -np 4 ./mpi_knn iris_train.csv iris_test.csv 10
```

Arguments:

- `train_file`
- `test_file`
- `k`

Run benchmark sweep script:

```bash
cd MPI
chmod +x mpi.sh
./mpi.sh
```

Note: `mpi.sh` uses `--hostfile hosts`. If no hostfile is configured, run `mpirun` manually (as above) or update the script.

### 4. CUDA Version

From `CUDA` folder:

```bash
cd CUDA
nvcc -O3 -o knn_cuda knn_cuda.cu -lm
./knn_cuda iris_train.csv iris_test.csv 10 16 16
```

Arguments:

- `train_file`
- `test_file`
- `k`
- `block_size_x`
- `block_size_y`

Run automated CUDA configuration benchmarking:

```bash
cd CUDA
chmod +x cuda.sh
./cuda.sh
```

This generates a CSV report like `cuda_results_k10.csv`.

## Output

All implementations report:

- Correct predictions
- Accuracy (%)
- Execution time (ms)
- Key runtime configuration (`k`, threads/processes/block size)

## Suggested Comparison Workflow

1. Run serial baseline.
2. Run OpenMP thread sweep.
3. Run MPI process sweep.
4. Run CUDA block-size sweep.
5. Compare accuracy consistency and speedup across models.