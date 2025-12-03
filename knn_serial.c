#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Structure to hold a single data point
typedef struct {
    double *features;  // Array of feature values
    int label;         // Class label (0, 1, or 2)
    int id;            // Optional: sample ID
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
int knn_predict(Dataset *train_data, double *test_point, int k);
void free_dataset(Dataset *dataset);
int majority_vote(int *labels, int k);

// Main function
int main(int argc, char *argv[]) {
    // Default parameters
    char *train_file = "iris_train.csv";
    char *test_file = "iris_test.csv";
    int k = 3;
    
    // Parse command line arguments
    if (argc >= 4) {
        train_file = argv[1];
        test_file = argv[2];
        k = atoi(argv[3]);
    } else {
        printf("Using default parameters:\n");
        printf("  Train file: %s\n", train_file);
        printf("  Test file: %s\n", test_file);
        printf("  k = %d\n\n", k);
    }
    
    // Load training and test datasets
    printf("Loading training data from %s...\n", train_file);
    Dataset *train_data = load_csv(train_file);
    if (!train_data) {
        fprintf(stderr, "Failed to load training data\n");
        return 1;
    }
    
    printf("Loading test data from %s...\n", test_file);
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
    printf("k = %d\n\n", k);
    
    // Verify feature dimensions match
    if (train_data->num_features != test_data->num_features) {
        fprintf(stderr, "Error: Feature dimension mismatch!\n");
        free_dataset(train_data);
        free_dataset(test_data);
        return 1;
    }
    
    // Start timing
    clock_t start_time = clock();
    
    // Perform KNN classification
    int correct_predictions = 0;
    printf("Starting KNN classification...\n");
    
    for (int i = 0; i < test_data->num_points; i++) {
        // Get the test point
        double *test_point = test_data->points[i].features;
        int true_label = test_data->points[i].label;
        
        // Predict using KNN
        int predicted_label = knn_predict(train_data, test_point, k);
        
        // Check if prediction is correct
        int is_correct = (predicted_label == true_label);
        if (is_correct) {
            correct_predictions++;
        }
        
        // Print result for this test sample
        printf("Test sample %3d: Predicted = %d, Actual = %d %s\n",
               i + 1, predicted_label, true_label,
               is_correct ? "✓" : "✗");
    }
    
    // End timing
    clock_t end_time = clock();
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    
    // Calculate accuracy
    double accuracy = (double)correct_predictions / test_data->num_points * 100.0;
    
    printf("Results:\n");
    printf("  Correct predictions: %d/%d\n", correct_predictions, test_data->num_points);
    printf("  Accuracy: %.2f%%\n", accuracy);
    printf("  Execution time: %.2f ms\n", execution_time);
    printf("  k value: %d\n", k);
    
    // Clean up
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
        
        // Assign ID
        dataset->points[dataset->num_points].id = dataset->num_points;
        
        dataset->num_points++;
        line_num++;
    }
    
    fclose(file);
    
    printf("Loaded %d samples with %d features each\n", 
           dataset->num_points, dataset->num_features);
    
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

// Find majority label among k nearest neighbors
int majority_vote(int *labels, int k) {
    int votes[3] = {0}; // For 3 classes (0, 1, 2)
    
    for (int i = 0; i < k; i++) {
        if (labels[i] >= 0 && labels[i] <= 2) {
            votes[labels[i]]++;
        }
    }
    
    // Find label with maximum votes
    int max_votes = -1;
    int predicted_label = -1;
    
    for (int i = 0; i < 3; i++) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            predicted_label = i;
        }
    }
    
    return predicted_label;
}

// KNN prediction for a single test point
int knn_predict(Dataset *train_data, double *test_point, int k) {
    // Validate k
    if (k <= 0 || k > train_data->num_points) {
        fprintf(stderr, "Invalid k value: %d\n", k);
        return -1;
    }
    
    // Allocate array for distances and labels
    DistanceLabel *distances = (DistanceLabel*)malloc(train_data->num_points * sizeof(DistanceLabel));
    if (!distances) {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }
    
    // Calculate distances to all training points
    for (int i = 0; i < train_data->num_points; i++) {
        distances[i].distance = euclidean_distance(test_point, 
                                                   train_data->points[i].features, 
                                                   train_data->num_features);
        distances[i].label = train_data->points[i].label;
    }
    
    // Sort distances in ascending order
    qsort(distances, train_data->num_points, sizeof(DistanceLabel), compare_distance);
    
    // Extract labels of k nearest neighbors
    int *nearest_labels = (int*)malloc(k * sizeof(int));
    if (!nearest_labels) {
        free(distances);
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }
    
    for (int i = 0; i < k; i++) {
        nearest_labels[i] = distances[i].label;
    }
    
    // Get majority vote
    int predicted_label = majority_vote(nearest_labels, k);
    
    // Clean up
    free(distances);
    free(nearest_labels);
    
    return predicted_label;
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