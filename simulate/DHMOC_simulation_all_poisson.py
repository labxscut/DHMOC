import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
import os
import pickle

def generate_means_and_d_values(num_categories, num_features):
    # Initialize the mean matrix
    means = np.zeros((num_categories, num_features))

    # Generate the mean for each category-feature combination
    for j in range(num_categories):
        np.random.seed(123 * (j + 1))  # Set the random seed for reproducibility
        # Generate means from a Poisson distribution
        means[j, :] = np.random.poisson(lam=1.0, size=num_features)  # Set λ parameter to 1.0

    # Initialize the d_values matrix
    d_values = np.zeros((num_categories, num_features))

    # Calculate the d-value for each pair of categories
    for i in range(num_categories - 1):
        for j in range(i + 1, num_categories):
            d_value = np.sum(means[i, :] * means[j, :]) / (np.sqrt(np.sum(means[i, :] ** 2)) * np.sqrt(np.sum(means[j, :] ** 2)))
            d_values[i, j] = d_value
            d_values[j, i] = d_value  # Because the distance matrix is symmetric

    return means, d_values

# Define the list of sample sizes
samples = [500, 1000, 5000, 10000]
# Define the list of feature counts
features = [500, 1000, 5000, 10000]  # Fixed list of feature counts

# Define the list of category counts
classes = [3, 5, 7, 10]  # Category counts: 3, 5, 7, 10

# Dictionary to store results
results = {}

# Use a triple loop to iterate through samples, features, and classes
for sample in samples:
    for feature in features:  # Iterate through different feature counts
        for class_count in classes:
            if class_count == 1:
                # If the number of classes is 1, directly generate a dataset with all zeros
                simulated_data = np.hstack([np.ones((sample, 1)), np.zeros((sample, feature))])
                s3_tree = None
            else:
                means, d_values = generate_means_and_d_values(class_count, feature)
                results[(sample, class_count, feature)] = {
                    'means': means,
                    'd_values': d_values
                }

                # Create distance_matrix_s3
                distance_matrix_s3 = np.zeros((class_count, class_count))

                # Fill the distance_matrix_s3
                for i in range(class_count - 1):
                    for j in range(i + 1, class_count):
                        distance_matrix_s3[i, j] = d_values[i, j]
                        distance_matrix_s3[j, i] = d_values[j, i]

                # Convert distance_matrix_s3 to condensed form for hierarchical clustering
                condensed_distance_matrix = squareform(distance_matrix_s3)

                # Perform hierarchical clustering
                hc = linkage(condensed_distance_matrix, method='single')

                # Convert the hierarchical clustering result to a tree structure
                s3_tree = to_tree(hc)

                # Define the list of standard deviations
                sd = [1/100, 1/10, 1/6, 1/4, 1/3, 1, 3]
                std_devs = [np.full((class_count, feature), s) for s in sd]

                # Generate simulated data
                s3 = []
                for k, s in enumerate(sd):
                    simulated_data_list = []
                    for i in range(class_count):
                        simulated_data_i = np.zeros((sample // class_count, feature))
                        for j in range(feature):
                            np.random.seed(123 * j)
                            # Generate data from a Poisson distribution
                            lambda_param = s  # Poisson distribution λ parameter
                            simulated_data_i[:, j] = np.random.poisson(lam=lambda_param, size=sample // class_count)
                        simulated_data_list.append(simulated_data_i)

                    simulated_data = np.vstack(simulated_data_list)
                    simulated_data = np.hstack([np.repeat(np.arange(1, class_count + 1), sample // class_count).reshape(-1, 1), simulated_data])
                    s3.append(simulated_data)

                    # Save the simulated data
                    data_filename = f"/work2/wlp/DHMOC/DHMOC/DHMOC_new/result/simulation/all_poisson/DHMOC_poisson_{sample}_{class_count}_{feature}_{s:.4f}.csv"
                    os.makedirs(os.path.dirname(data_filename), exist_ok=True)
                    pd.DataFrame(simulated_data).to_csv(data_filename, index=False)
                    print(f"Saved simulation data to: {data_filename}")

            # Print results
            print(f"Sample size: {sample}, Number of classes: {class_count}, Number of features: {feature}")
            if class_count > 1:
                print("Distance Matrix S3:")
                print(pd.DataFrame(distance_matrix_s3, index=range(1, class_count + 1), columns=range(1, class_count + 1)))
            print("\nSimulated Data (first few rows):")
            print(simulated_data[:5, :])  # Print the first few rows of simulated data for the first standard deviation
            print("\n" + "-"*80 + "\n")
