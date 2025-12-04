#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

// Structure to hold a single data point
typedef struct {
    double *features;  // Array of feature values
    int label;         // Class label (0, 1, or 2)
} DataPoint;

// Structure to hold distance-label pairs for sorting
typedef struct {
    double distance;
    int label;
} DistanceLabel;

// Structure to hold a dataset
typedef struct {
    DataPoint *points;   // Array of data points
    int num_points;      // Number of data points
    int num_features;    // Number of features per point
} Dataset;

// Function prototypes
Dataset* load_csv(const char *filename);
double euclidean_distance(double *point1, double *point2, int num_features);
int compare_distance(const void *a, const void *b);
int knn_predict_omp(Dataset *train_data, double *test_point, int k, int thread_count);
int majority_vote(int *labels, int k);
void free_dataset(Dataset *dataset);
int find_max_label(Dataset *dataset);

// Main function with OpenMP support
int main(int argc, char *argv[]) {
    // Default parameters
    char *train_file = "iris_train.csv";
    char *test_file = "iris_test.csv";
    int k = 10;
    int thread_count = 4;  // Default OpenMP threads
    
    // Parse command line arguments
    if (argc >= 5) {
        train_file = argv[1];
        test_file = argv[2];
        k = atoi(argv[3]);
        thread_count = atoi(argv[4]);
    } else {
        printf("Using default parameters:\n");
        printf("  Train file: %s\n", train_file);
        printf("  Test file: %s\n", test_file);
        printf("  k = %d\n", k);
        printf("  Threads = %d\n\n", thread_count);
    }
    
    // Set number of OpenMP threads
    omp_set_num_threads(thread_count);
    
    // Load training and test datasets
    printf("Loading training data from %s\n", train_file);
    Dataset *train_data = load_csv(train_file);
    if (!train_data) {
        fprintf(stderr, "Failed to load training data\n");
        return 1;
    }
    
    printf("Loading test data from %s\n", test_file);
    Dataset *test_data = load_csv(test_file);
    if (!test_data) {
        fprintf(stderr, "Failed to load test data\n");
        free_dataset(train_data);
        return 1;
    }
    
    // Detect number of classes from training data
    int max_train_label = find_max_label(train_data);
    int max_test_label = find_max_label(test_data);
    int num_classes = (max_train_label > max_test_label ? max_train_label : max_test_label) + 1;
    
    printf("Training samples: %d\n", train_data->num_points);
    printf("Test samples: %d\n", test_data->num_points);
    printf("Features per sample: %d\n", train_data->num_features);
    printf("Number of classes: %d\n", num_classes);
    printf("k = %d\n", k);
    printf("OpenMP Threads: %d\n\n", thread_count);
    
    // Verify feature dimensions match
    if (train_data->num_features != test_data->num_features) {
        fprintf(stderr, "Error: Feature dimension mismatch!\n");
        free_dataset(train_data);
        free_dataset(test_data);
        return 1;
    }
    
    // Start timing (using OpenMP's high-resolution timer)
    double start_time = omp_get_wtime();
    
    // Allocate arrays to store results (avoid critical section)
    int *predictions = (int*)malloc(test_data->num_points * sizeof(int));
    int *actuals = (int*)malloc(test_data->num_points * sizeof(int));
    
    // Perform KNN classification
    int correct_predictions = 0;
    
    // Parallel region for test samples
    #pragma omp parallel reduction(+:correct_predictions)
    {
        // Each thread gets its own workspace to avoid race conditions
        // Allocate ONCE per thread, not per iteration
        DistanceLabel *local_distances = (DistanceLabel*)malloc(train_data->num_points * sizeof(DistanceLabel));
        int *nearest_labels = (int*)malloc(k * sizeof(int));
        int *votes = (int*)malloc(num_classes * sizeof(int));
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < test_data->num_points; i++) {
            // Get the test point
            double *test_point = test_data->points[i].features;
            int true_label = test_data->points[i].label;
            
            // Predict using KNN with OpenMP
            int predicted_label = -1;
            
            // Calculate distances to all training points
            for (int j = 0; j < train_data->num_points; j++) {
                local_distances[j].distance = euclidean_distance(test_point, 
                                                                 train_data->points[j].features, 
                                                                 train_data->num_features);
                local_distances[j].label = train_data->points[j].label;
            }
            
            // Sort distances in ascending order
            qsort(local_distances, train_data->num_points, sizeof(DistanceLabel), compare_distance);
            
            // Extract labels of k nearest neighbors
            for (int j = 0; j < k; j++) {
                nearest_labels[j] = local_distances[j].label;
            }
            
            // Get majority vote - reset votes array
            for (int j = 0; j < num_classes; j++) {
                votes[j] = 0;
            }
            
            // Count votes for each class
            for (int j = 0; j < k; j++) {
                int label = nearest_labels[j];
                if (label >= 0 && label < num_classes) {
                    votes[label]++;
                }
            }
            
            // Find label with maximum votes
            int max_votes = -1;
            predicted_label = -1;
            for (int j = 0; j < num_classes; j++) {
                if (votes[j] > max_votes) {
                    max_votes = votes[j];
                    predicted_label = j;
                }
            }
            
            // Store results (no critical section needed!)
            predictions[i] = predicted_label;
            actuals[i] = true_label;
            
            // Check if prediction is correct
            if (predicted_label == true_label) {
                correct_predictions++;
            }
        }
        
        // Free thread-local memory
        free(votes);
        free(local_distances);
        free(nearest_labels);
    }
    
    // End timing
    double end_time = omp_get_wtime();
    double execution_time = (end_time - start_time) * 1000.0; // Convert to milliseconds
    
    // Calculate accuracy
    double accuracy = (double)correct_predictions / test_data->num_points * 100.0;
    
    printf("OpenMP Results:\n");
    printf("  Correct predictions: %d/%d\n", correct_predictions, test_data->num_points);
    printf("  Accuracy: %.2f%%\n", accuracy);
    printf("  Execution time: %.4f ms\n", execution_time);
    printf("  k value: %d\n", k);
    printf("  Threads used: %d\n", thread_count);
    
    // Clean up
    free(predictions);
    free(actuals);
    free_dataset(train_data);
    free_dataset(test_data);
    
    return 0;
}

// Function to load CSV file into Dataset structure
Dataset* load_csv(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return NULL;
    }
    
    // Count lines in file to allocate memory
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
    
    // Allocate dataset structure
    Dataset *dataset = (Dataset*)malloc(sizeof(Dataset));
    if (!dataset) {
        fclose(file);
        return NULL;
    }
    
    // Allocate array for data points
    dataset->points = (DataPoint*)malloc(num_lines * sizeof(DataPoint));
    if (!dataset->points) {
        free(dataset);
        fclose(file);
        return NULL;
    }
    
    dataset->num_points = 0;
    dataset->num_features = 0; // Will be determined from first line
    
    // Read each line
    int line_num = 0;
    while (fgets(buffer, sizeof(buffer), file)) {
        // Remove newline character
        buffer[strcspn(buffer, "\n")] = 0;
        
        // Skip empty lines
        if (strlen(buffer) == 0) {
            continue;
        }
        
        // Count features in this line (commas + 1)
        int features_in_line = 1;
        for (int i = 0; buffer[i] != '\0'; i++) {
            if (buffer[i] == ',') {
                features_in_line++;
            }
        }
        
        // Initialize num_features from first valid line
        if (dataset->num_features == 0) {
            dataset->num_features = features_in_line - 1; // Last column is label
        }
        
        // Check consistency
        if (features_in_line != dataset->num_features + 1) {
            fprintf(stderr, "Warning: Line %d has %d features, expected %d. Skipping.\n",
                   line_num + 1, features_in_line - 1, dataset->num_features);
            continue;
        }
        
        // Allocate features array
        dataset->points[dataset->num_points].features = 
            (double*)malloc(dataset->num_features * sizeof(double));
        
        if (!dataset->points[dataset->num_points].features) {
            // Clean up on allocation failure
            for (int i = 0; i < dataset->num_points; i++) {
                free(dataset->points[i].features);
            }
            free(dataset->points);
            free(dataset);
            fclose(file);
            return NULL;
        }
        
        // Parse the line
        char *token = strtok(buffer, ",");
        int feature_idx = 0;
        
        while (token != NULL && feature_idx < dataset->num_features) {
            dataset->points[dataset->num_points].features[feature_idx] = atof(token);
            token = strtok(NULL, ",");
            feature_idx++;
        }
        
        // Last token is the label
        if (token != NULL) {
            dataset->points[dataset->num_points].label = atoi(token);
        } else {
            dataset->points[dataset->num_points].label = -1; // Invalid label
        }
        
        dataset->num_points++;
        line_num++;
    }
    
    fclose(file);
    
    return dataset;
}

// Calculate Euclidean distance between two points
double euclidean_distance(double *point1, double *point2, int num_features) {
    double sum = 0.0;
    for (int i = 0; i < num_features; i++) {
        double diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Comparator function for qsort (ascending order by distance)
int compare_distance(const void *a, const void *b) {
    DistanceLabel *dl_a = (DistanceLabel*)a;
    DistanceLabel *dl_b = (DistanceLabel*)b;
    
    if (dl_a->distance < dl_b->distance) return -1;
    if (dl_a->distance > dl_b->distance) return 1;
    return 0;
}

// Find maximum label value in dataset to determine number of classes
int find_max_label(Dataset *dataset) {
    if (!dataset || dataset->num_points == 0) {
        return -1;
    }
    
    int max_label = dataset->points[0].label;
    
    // Parallelize the search for maximum label using reduction
    #pragma omp parallel for reduction(max:max_label)
    for (int i = 0; i < dataset->num_points; i++) {
        if (dataset->points[i].label > max_label) {
            max_label = dataset->points[i].label;
        }
    }
    
    return max_label;
}

// Free memory allocated for dataset
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