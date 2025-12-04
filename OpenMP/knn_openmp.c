#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <omp.h>

typedef struct {
    double *features;  
    int label;         
} DataPoint;

typedef struct {
    double distance;
    int label;
} DistanceLabel;

typedef struct {
    DataPoint *points;   
    int num_points;      
    int num_features;    
} Dataset;

Dataset* load_csv(const char *filename);
double euclidean_distance(double *point1, double *point2, int num_features);
int compare_distance(const void *a, const void *b);
int knn_predict_omp(Dataset *train_data, double *test_point, int k, int thread_count);
int majority_vote(int *labels, int k);
void free_dataset(Dataset *dataset);
int find_max_label(Dataset *dataset);

int main(int argc, char *argv[]) {
    char *train_file = "iris_train.csv";
    char *test_file = "iris_test.csv";
    int k = 10;
    int thread_count = 4;  
    
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
    
    omp_set_num_threads(thread_count);
    
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
    
    int max_train_label = find_max_label(train_data);
    int max_test_label = find_max_label(test_data);
    int num_classes = (max_train_label > max_test_label ? max_train_label : max_test_label) + 1;
    
    printf("Training samples: %d\n", train_data->num_points);
    printf("Test samples: %d\n", test_data->num_points);
    printf("Features per sample: %d\n", train_data->num_features);
    printf("Number of classes: %d\n", num_classes);
    printf("k = %d\n", k);
    printf("OpenMP Threads: %d\n\n", thread_count);
    
    if (train_data->num_features != test_data->num_features) {
        fprintf(stderr, "Error: Feature dimension mismatch!\n");
        free_dataset(train_data);
        free_dataset(test_data);
        return 1;
    }
    
    double start_time = omp_get_wtime();
    
    int *predictions = (int*)malloc(test_data->num_points * sizeof(int));
    int *actuals = (int*)malloc(test_data->num_points * sizeof(int));
    
    int correct_predictions = 0;
    
    #pragma omp parallel reduction(+:correct_predictions)
    {
        DistanceLabel *local_distances = (DistanceLabel*)malloc(train_data->num_points * sizeof(DistanceLabel));
        int *nearest_labels = (int*)malloc(k * sizeof(int));
        int *votes = (int*)malloc(num_classes * sizeof(int));
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < test_data->num_points; i++) {
            double *test_point = test_data->points[i].features;
            int true_label = test_data->points[i].label;
            
            int predicted_label = -1;
            
            for (int j = 0; j < train_data->num_points; j++) {
                local_distances[j].distance = euclidean_distance(test_point, 
                                                                 train_data->points[j].features, 
                                                                 train_data->num_features);
                local_distances[j].label = train_data->points[j].label;
            }
            
            qsort(local_distances, train_data->num_points, sizeof(DistanceLabel), compare_distance);
            
            for (int j = 0; j < k; j++) {
                nearest_labels[j] = local_distances[j].label;
            }
            
            for (int j = 0; j < num_classes; j++) {
                votes[j] = 0;
            }
            
            for (int j = 0; j < k; j++) {
                int label = nearest_labels[j];
                if (label >= 0 && label < num_classes) {
                    votes[label]++;
                }
            }
            
            int max_votes = -1;
            predicted_label = -1;
            for (int j = 0; j < num_classes; j++) {
                if (votes[j] > max_votes) {
                    max_votes = votes[j];
                    predicted_label = j;
                }
            }
            
            predictions[i] = predicted_label;
            actuals[i] = true_label;
            
            if (predicted_label == true_label) {
                correct_predictions++;
            }
        }
        
        free(votes);
        free(local_distances);
        free(nearest_labels);
    }
    
    double end_time = omp_get_wtime();
    double execution_time = (end_time - start_time) * 1000.0; 
    
    double accuracy = (double)correct_predictions / test_data->num_points * 100.0;
    
    printf("OpenMP Results:\n");
    printf("  Correct predictions: %d/%d\n", correct_predictions, test_data->num_points);
    printf("  Accuracy: %.2f%%\n", accuracy);
    printf("  Execution time: %.4f ms\n", execution_time);
    printf("  k value: %d\n", k);
    printf("  Threads used: %d\n", thread_count);
    
    free(predictions);
    free(actuals);
    free_dataset(train_data);
    free_dataset(test_data);
    
    return 0;
}

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
    
    int line_num = 0;
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
            fprintf(stderr, "Warning: Line %d has %d features, expected %d. Skipping.\n",
                   line_num + 1, features_in_line - 1, dataset->num_features);
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
        
        dataset->num_points++;
        line_num++;
    }
    
    fclose(file);
    
    return dataset;
}

double euclidean_distance(double *point1, double *point2, int num_features) {
    double sum = 0.0;
    for (int i = 0; i < num_features; i++) {
        double diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

int compare_distance(const void *a, const void *b) {
    DistanceLabel *dl_a = (DistanceLabel*)a;
    DistanceLabel *dl_b = (DistanceLabel*)b;
    
    if (dl_a->distance < dl_b->distance) return -1;
    if (dl_a->distance > dl_b->distance) return 1;
    return 0;
}

int find_max_label(Dataset *dataset) {
    if (!dataset || dataset->num_points == 0) {
        return -1;
    }
    
    int max_label = dataset->points[0].label;
    
    #pragma omp parallel for reduction(max:max_label)
    for (int i = 0; i < dataset->num_points; i++) {
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