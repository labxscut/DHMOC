import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import os
# from Bio import Phylo


def generate_means_and_d_values(num_categories, num_features):
    # Initialize the means matrix
    means = np.zeros((num_categories, num_features))

    # Generate means for each category-feature combination
    for j in range(num_categories):
        np.random.seed(123 * (j + 1))  # Set random seed for reproducibility
        means[j, :] = np.random.uniform(0.5, 1.5, num_features)

    # Initialize the d_values matrix
    d_values = np.zeros((num_categories, num_features))

    # Calculate d values for each pair of categories
    for i in range(num_categories - 1):
        for j in range(i + 1, num_categories):
            d_value = np.sum(means[i, :] * means[j, :]) / (np.sqrt(np.sum(means[i, :] ** 2)) * np.sqrt(np.sum(means[j, :] ** 2)))
            d_values[i, j] = d_value
            d_values[j, i] = d_value  # Because the distance matrix is symmetric

    return means, d_values


# Define a list of sample sizes
samples = [500, 1000, 5000, 10000]
# Define a list of feature counts
features = [500, 1000, 5000, 10000]
# Define a list of class counts
classes =  [3, 5, 7, 10]  # Class counts are 3, 5, 7, 10

# Dictionary to store results
results = {}

# Double loop over samples and classes
for sample in samples:
    for feature in features:
        for class_count in classes:
            if class_count == 1:
                # If the number of classes is 1, generate a dataset of all zeros
                simulated_data = np.hstack([np.ones((sample, 1)), np.zeros((sample, feature))])
                s3_tree = None
            else:
                means, d_values = generate_means_and_d_values(class_count, feature)
                results[(sample, class_count)] = {
                    'means': means,
                    'd_values': d_values
                }

                # Convert to distance matrix
                distance_matrix = 1 - d_values  # Convert similarity matrix to distance matrix

                # Create distance_matrix_s3
                distance_matrix_s3 = np.zeros((class_count, class_count))

                # Fill distance_matrix_s3
                for i in range(class_count - 1):
                    for j in range(i + 1, class_count):
                        distance_matrix_s3[i, j] = distance_matrix[i, j]
                        distance_matrix_s3[j, i] = distance_matrix[j, i]

                # Perform hierarchical clustering (directly using distance matrix without inversion)
                linkage_matrix = linkage(squareform(distance_matrix_s3), method='single')

                # Plot dendrogram
                plt.figure(figsize=(10, 5))
                dendrogram(linkage_matrix, labels=[str(i + 1) for i in range(class_count)])
                plt.show()

                # Define a list of standard deviations
                sd = [1/100, 1/10, 1/6, 1/4, 1/3, 1, 3]
                std_devs = [np.full((class_count, feature), s) for s in sd]

                # Generate simulated data
                s3 = []
                for k, std in enumerate(sd):
                    simulated_data_list = []
                    for i in range(class_count):
                        simulated_data_i = np.zeros((sample // class_count, feature))
                        for j in range(feature):
                            np.random.seed(123 * j)
                            simulated_data_i[:, j] = np.random.normal(means[i, j], std_devs[k][i, j], size=sample // class_count)
                        
                        simulated_data_list.append(simulated_data_i)

                    simulated_data = np.vstack(simulated_data_list)
                    simulated_data = np.hstack([np.repeat(np.arange(1, class_count + 1), sample // class_count).reshape(-1, 1), simulated_data])

                    # Save simulated data
                    data_filename = f"/work2/wlp/DHMOC/DHMOC/DHMOC_new/result/simulation/unif/DHMOC_unif_{sample}_{feature}_{class_count}_{std:.4f}.csv"
                    os.makedirs(os.path.dirname(data_filename), exist_ok=True)
                    pd.DataFrame(simulated_data).to_csv(data_filename, index=False)
                    print(f"Saved simulation data to: {data_filename}")

            # Print results
            print(f"Sample size: {sample}, Number of classes: {class_count}")
            if class_count > 1:
                print("Distance Matrix S3:")
                print(pd.DataFrame(distance_matrix_s3, index=range(1, class_count + 1), columns=range(1, class_count + 1)))
            print("\nSimulated Data (first few rows):")
            print(simulated_data[:5, :])  # Print the first few rows of simulated data for the first standard deviation
            print("\n" + "-"*80 + "\n")
