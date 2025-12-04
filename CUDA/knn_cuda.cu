#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Structure to hold a single data point
typedef struct {
    double *features;
    int label;
    int id;
} DataPoint;

// Structure to hold distance-label pairs for sorting
typedef struct {
    double distance;
    int label;
} DistanceLabel;

// Structure to hold a dataset
typedef struct {
    DataPoint *points;
    int num_points;
    int num_features;
} Dataset;

// Function prototypes
Dataset* load_csv(const char *filename);
void free_dataset(Dataset *dataset);
int find_max_label(Dataset *dataset);
int compare_distance(const void *a, const void *b);

// CUDA kernel to calculate distances for multiple test points with configurable block size
__global__ void calculate_distances_batch_kernel(
    double *train_features,
    double *test_features,
    double *distances,
    int num_train,
    int num_test,
    int num_features
) {
    int test_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int train_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (test_idx < num_test && train_idx < num_train) {
        double sum = 0.0;
        for (int f = 0; f < num_features; f++) {
            double diff = train_features[train_idx * num_features + f] - 
                         test_features[test_idx * num_features + f];
            sum += diff * diff;
        }
        distances[test_idx * num_train + train_idx] = sqrt(sum);
    }
}

// Batch prediction function using GPU with configurable block dimensions
void knn_predict_batch_gpu(
    double *d_train_features,
    int *d_train_labels,
    double *h_test_features,
    int *h_predictions,
    int num_train,
    int num_test,
    int num_features,
    int k,
    int num_classes,
    int block_dim_x,
    int block_dim_y
) {
    double *d_test_features, *d_distances;
    CUDA_CHECK(cudaMalloc(&d_test_features, num_test * num_features * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_distances, num_test * num_train * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_test_features, h_test_features, 
                         num_test * num_features * sizeof(double), 
                         cudaMemcpyHostToDevice));
    
    // Use configurable block dimensions
    dim3 threads_per_block(block_dim_x, block_dim_y);
    dim3 num_blocks((num_train + block_dim_x - 1) / block_dim_x, 
                    (num_test + block_dim_y - 1) / block_dim_y);
    
    calculate_distances_batch_kernel<<<num_blocks, threads_per_block>>>(
        d_train_features, d_test_features, d_distances, 
        num_train, num_test, num_features
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    double *h_distances = (double*)malloc(num_test * num_train * sizeof(double));
    CUDA_CHECK(cudaMemcpy(h_distances, d_distances, 
                         num_test * num_train * sizeof(double), 
                         cudaMemcpyDeviceToHost));
    
    int *h_labels = (int*)malloc(num_train * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_labels, d_train_labels, 
                         num_train * sizeof(int), cudaMemcpyDeviceToHost));
    
    for (int t = 0; t < num_test; t++) {
        DistanceLabel *dist_labels = (DistanceLabel*)malloc(num_train * sizeof(DistanceLabel));
        for (int i = 0; i < num_train; i++) {
            dist_labels[i].distance = h_distances[t * num_train + i];
            dist_labels[i].label = h_labels[i];
        }
        
        qsort(dist_labels, num_train, sizeof(DistanceLabel), compare_distance);
        
        int *votes = (int*)calloc(num_classes, sizeof(int));
        for (int i = 0; i < k; i++) {
            int label = dist_labels[i].label;
            if (label >= 0 && label < num_classes) {
                votes[label]++;
            }
        }
        
        int max_votes = -1;
        int predicted_label = -1;
        for (int i = 0; i < num_classes; i++) {
            if (votes[i] > max_votes) {
                max_votes = votes[i];
                predicted_label = i;
            }
        }
        
        h_predictions[t] = predicted_label;
        
        free(dist_labels);
        free(votes);
    }
    
    free(h_distances);
    free(h_labels);
    CUDA_CHECK(cudaFree(d_test_features));
    CUDA_CHECK(cudaFree(d_distances));
}

int compare_distance(const void *a, const void *b) {
    DistanceLabel *dl_a = (DistanceLabel*)a;
    DistanceLabel *dl_b = (DistanceLabel*)b;
    
    if (dl_a->distance < dl_b->distance) return -1;
    if (dl_a->distance > dl_b->distance) return 1;
    return 0;
}

int main(int argc, char *argv[]) {
    char *train_file = "iris_train.csv";
    char *test_file = "iris_test.csv";
    int k = 3;
    int block_dim_x = 16;
    int block_dim_y = 16;
    
    // Parse command line arguments
    if (argc >= 4) {
        train_file = argv[1];
        test_file = argv[2];
        k = atoi(argv[3]);
    }
    if (argc >= 6) {
        block_dim_x = atoi(argv[4]);
        block_dim_y = atoi(argv[5]);
    }
    
    // Check CUDA device
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }
    
    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, 0));
    printf("========================================\n");
    printf("CUDA Configuration\n");
    printf("========================================\n");
    printf("Device: %s\n", device_prop.name);
    printf("Compute capability: %d.%d\n", device_prop.major, device_prop.minor);
    printf("Max threads per block: %d\n", device_prop.maxThreadsPerBlock);
    printf("Max block dimensions: (%d, %d, %d)\n", 
           device_prop.maxThreadsDim[0], 
           device_prop.maxThreadsDim[1], 
           device_prop.maxThreadsDim[2]);
    printf("Max grid dimensions: (%d, %d, %d)\n", 
           device_prop.maxGridSize[0], 
           device_prop.maxGridSize[1], 
           device_prop.maxGridSize[2]);
    printf("Warp size: %d\n", device_prop.warpSize);
    printf("========================================\n\n");
    
    printf("Execution Parameters:\n");
    printf("  Train file: %s\n", train_file);
    printf("  Test file: %s\n", test_file);
    printf("  k = %d\n", k);
    printf("  Block dimensions: (%d, %d)\n", block_dim_x, block_dim_y);
    printf("  Threads per block: %d\n\n", block_dim_x * block_dim_y);
    
    // Load datasets
    Dataset *train_data = load_csv(train_file);
    if (!train_data) {
        fprintf(stderr, "Failed to load training data\n");
        return 1;
    }
    
    Dataset *test_data = load_csv(test_file);
    if (!test_data) {
        fprintf(stderr, "Failed to load test data\n");
        free_dataset(train_data);
        return 1;
    }
    
    printf("Dataset loaded successfully!\n");
    printf("Training samples: %d\n", train_data->num_points);
    printf("Test samples: %d\n", test_data->num_points);
    printf("Features per sample: %d\n", train_data->num_features);
    
    int max_train_label = find_max_label(train_data);
    int max_test_label = find_max_label(test_data);
    int num_classes = (max_train_label > max_test_label ? max_train_label : max_test_label) + 1;
    printf("Number of classes: %d\n\n", num_classes);
    
    if (train_data->num_features != test_data->num_features) {
        fprintf(stderr, "Error: Feature dimension mismatch!\n");
        free_dataset(train_data);
        free_dataset(test_data);
        return 1;
    }
    
    int num_train = train_data->num_points;
    int num_test = test_data->num_points;
    int num_features = train_data->num_features;
    
    // Flatten data
    double *h_train_features = (double*)malloc(num_train * num_features * sizeof(double));
    int *h_train_labels = (int*)malloc(num_train * sizeof(int));
    double *h_test_features = (double*)malloc(num_test * num_features * sizeof(double));
    int *h_test_labels = (int*)malloc(num_test * sizeof(int));
    
    for (int i = 0; i < num_train; i++) {
        for (int j = 0; j < num_features; j++) {
            h_train_features[i * num_features + j] = train_data->points[i].features[j];
        }
        h_train_labels[i] = train_data->points[i].label;
    }
    
    for (int i = 0; i < num_test; i++) {
        for (int j = 0; j < num_features; j++) {
            h_test_features[i * num_features + j] = test_data->points[i].features[j];
        }
        h_test_labels[i] = test_data->points[i].label;
    }
    
    // Allocate device memory
    double *d_train_features;
    int *d_train_labels;
    CUDA_CHECK(cudaMalloc(&d_train_features, num_train * num_features * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_train_labels, num_train * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_train_features, h_train_features, 
                         num_train * num_features * sizeof(double), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_train_labels, h_train_labels, 
                         num_train * sizeof(int), 
                         cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Start timing
    CUDA_CHECK(cudaEventRecord(start));
    
    // Perform KNN classification with specified block dimensions
    int *h_predictions = (int*)malloc(num_test * sizeof(int));
    knn_predict_batch_gpu(d_train_features, d_train_labels, h_test_features, 
                         h_predictions, num_train, num_test, num_features, 
                         k, num_classes, block_dim_x, block_dim_y);
    
    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float execution_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&execution_time, start, stop));
    
    // Calculate accuracy
    int correct_predictions = 0;
    for (int i = 0; i < num_test; i++) {
        if (h_predictions[i] == h_test_labels[i]) {
            correct_predictions++;
        }
    }
    
    double accuracy = (double)correct_predictions / num_test * 100.0;
    
    printf("========================================\n");
    printf("Results\n");
    printf("========================================\n");
    printf("Configuration: Block(%d,%d) = %d threads\n", 
           block_dim_x, block_dim_y, block_dim_x * block_dim_y);
    printf("Correct predictions: %d/%d\n", correct_predictions, num_test);
    printf("Accuracy: %.2f%%\n", accuracy);
    printf("Execution time: %.4f ms\n", execution_time);
    printf("========================================\n");
    
    // Output for easy parsing by benchmark script
    printf("\nCSV_OUTPUT:%d,%d,%d,%.4f,%.2f\n", 
           block_dim_x, block_dim_y, block_dim_x * block_dim_y, 
           execution_time, accuracy);
    
    // Clean up
    free(h_train_features);
    free(h_train_labels);
    free(h_test_features);
    free(h_test_labels);
    free(h_predictions);
    CUDA_CHECK(cudaFree(d_train_features));
    CUDA_CHECK(cudaFree(d_train_labels));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free_dataset(train_data);
    free_dataset(test_data);
    
    return 0;
}

// CSV loading function
Dataset* load_csv(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return NULL;
    }
    
    int num_lines = 0;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), file)) {
        num_lines++;
    }
    rewind(file);
    
    if (num_lines == 0) {
        fclose(file);
        return NULL;
    }
    
    Dataset *dataset = (Dataset*)malloc(sizeof(Dataset));
    if (!dataset) {
        fclose(file);
        return NULL;
    }
    
    dataset->points = (DataPoint*)malloc(num_lines * sizeof(DataPoint));
    if (!dataset->points) {
        free(dataset);
        fclose(file);
        return NULL;
    }
    
    dataset->num_points = 0;
    dataset->num_features = 0;
    
    while (fgets(buffer, sizeof(buffer), file)) {
        buffer[strcspn(buffer, "\n")] = 0;
        
        if (strlen(buffer) == 0) {
            continue;
        }
        
        int features_in_line = 1;
        for (int i = 0; buffer[i] != '\0'; i++) {
            if (buffer[i] == ',') {
                features_in_line++;
            }
        }
        
        if (dataset->num_features == 0) {
            dataset->num_features = features_in_line - 1;
        }
        
        if (features_in_line != dataset->num_features + 1) {
            continue;
        }
        
        dataset->points[dataset->num_points].features = 
            (double*)malloc(dataset->num_features * sizeof(double));
        
        if (!dataset->points[dataset->num_points].features) {
            for (int i = 0; i < dataset->num_points; i++) {
                free(dataset->points[i].features);
            }
            free(dataset->points);
            free(dataset);
            fclose(file);
            return NULL;
        }
        
        char *token = strtok(buffer, ",");
        int feature_idx = 0;
        
        while (token != NULL && feature_idx < dataset->num_features) {
            dataset->points[dataset->num_points].features[feature_idx] = atof(token);
            token = strtok(NULL, ",");
            feature_idx++;
        }
        
        if (token != NULL) {
            dataset->points[dataset->num_points].label = atoi(token);
        } else {
            dataset->points[dataset->num_points].label = -1;
        }
        
        dataset->points[dataset->num_points].id = dataset->num_points;
        dataset->num_points++;
    }
    
    fclose(file);
    return dataset;
}

int find_max_label(Dataset *dataset) {
    if (!dataset || dataset->num_points == 0) {
        return -1;
    }
    
    int max_label = dataset->points[0].label;
    for (int i = 1; i < dataset->num_points; i++) {
        if (dataset->points[i].label > max_label) {
            max_label = dataset->points[i].label;
        }
    }
    
    return max_label;
}

void free_dataset(Dataset *dataset) {
    if (dataset) {
        if (dataset->points) {
            for (int i = 0; i < dataset->num_points; i++) {
                if (dataset->points[i].features) {
                    free(dataset->points[i].features);
                }
            }
            free(dataset->points);
        }
        free(dataset);
    }
}
