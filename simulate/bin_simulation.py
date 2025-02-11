import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
import os
import pickle

def generate_means_and_d_values(num_categories, num_features):
    # Initialize the means matrix
    means = np.zeros((num_categories, num_features))

    # Generate means for each category-feature pair
    for j in range(num_categories):
        np.random.seed(123 * (j + 1))  # Set the random seed to ensure reproducibility
        # Generate means from a negative binomial distribution
        n = 10  # Parameter n of the negative binomial distribution
        p = 0.5  # Parameter p of the negative binomial distribution
        means[j, :] = np.random.negative_binomial(n, p, size=num_features)

    # Initialize the d_values matrix
    d_values = np.zeros((num_categories, num_categories))

    # Calculate the d value for each pair of categories
    for i in range(num_categories - 1):
        for j in range(i + 1, num_categories):
            d_value = np.sum(means[i, :] * means[j, :]) / (np.sqrt(np.sum(means[i, :] ** 2)) * np.sqrt(np.sum(means[j, :] ** 2)))
            d_values[i, j] = d_value
            d_values[j, i] = d_value  # The distance matrix is symmetric

    return means, d_values 

# List of sample sizes
samples = [500, 1000, 5000, 10000]
# samples = [ 10000]
# List of feature counts
features = [500, 1000, 5000, 10000]
# features = [5000, 10000]

# List of class counts
classes_list = [3, 5, 7, 10]  # Number of classes: 3, 5, 7, 10

# Dictionary to store results
results = {}

# Triple loop to iterate over samples, features, and classes
for sample in samples:
    for feature in features:
        for class_count in classes_list:
            if class_count == 1:
                # If the number of classes is 1, generate a dataset with all zeros
                simulated_data = np.hstack([np.ones((sample, 1)), np.zeros((sample, feature))])
                s3_tree = None
            else:
                means, d_values = generate_means_and_d_values(class_count, feature)
                results[(sample, class_count)] = {
                    'means': means,
                    'd_values': d_values
                }

                # Create the distance_matrix_s3
                distance_matrix_s3 = np.zeros((class_count, class_count))

                # Fill in the distance_matrix_s3
                for i in range(class_count - 1):
                    for j in range(i + 1, class_count):
                        distance_matrix_s3[i, j] = d_values[i, j]
                        distance_matrix_s3[j, i] = d_values[j, i]

                # Convert the distance_matrix_s3 to condensed form for hierarchical clustering
                condensed_distance_matrix = squareform(distance_matrix_s3)

                # Perform hierarchical clustering
                hc = linkage(condensed_distance_matrix, method='single')

                # Convert the hierarchical clustering result into a tree structure
                s3_tree = to_tree(hc)

                # Define a list of standard deviations
                sd = [1/100, 1/10, 1/6, 1/4, 1/3, 1, 3]

                # Generate simulated data
                s3 = []
                for k, std in enumerate(sd):
                    simulated_data_list = []
                    for i in range(class_count):
                        simulated_data_i = np.zeros((sample // class_count, feature))
                        for j in range(feature):
                            np.random.seed(123 * j)
                            # Generate data from a negative binomial distribution
                            n = 10  # Parameter n of the negative binomial distribution
                            p = 1 / (1 + means[i, j])  # Parameter p of the negative binomial distribution
                            simulated_data_i[:, j] = np.random.negative_binomial(n, p, size=sample // class_count)

                        # Add Poisson noise
                        poisson_noise = np.random.poisson(lam=std, size=simulated_data_i.shape)  # Poisson noise with standard deviation equal to the current std
                        simulated_data_i += poisson_noise    
                        simulated_data_list.append(simulated_data_i)

                    simulated_data = np.vstack(simulated_data_list)
                    simulated_data = np.hstack([np.repeat(np.arange(1, class_count + 1), sample // class_count).reshape(-1, 1), simulated_data])
                    s3.append(simulated_data)

                    # Save the simulated data
                    data_filename = f"/work2/wlp/DHMOC/DHMOC/DHMOC_new/result/simulation/negative_binomial_distribution/DHMOC_negative_binomial_{sample}_{feature}_{class_count}_{std:.4f}.csv"
                    os.makedirs(os.path.dirname(data_filename), exist_ok=True)
                    pd.DataFrame(simulated_data).to_csv(data_filename, index=False)
                    print(f"Saved simulation data to: {data_filename}")

                    # # Save s3_tree
                    # if s3_tree is not None:
                    #     tree_filename = f"/home/wlp/project/data/simulation/Negative_binomial/distance_matrix/DHMOC_negative_binomial_s3tree_{sample}_{feature}_{class_count}_{std:.4f}.pkl"
                    #     with open(tree_filename, 'wb') as f:
                    #         pickle.dump(s3_tree, f)
                    #     print(f"Saved s3_tree to: {tree_filename}")

            # Print results
            print(f"Sample size: {sample}, Feature count: {feature}, Number of classes: {class_count}")
            if class_count > 1:
                print("Distance Matrix S3:")
                print(pd.DataFrame(distance_matrix_s3, index=range(1, class_count + 1), columns=range(1, class_count + 1)))
            print("\nSimulated Data (first few rows):")
            print(simulated_data[:5, :])  # Print the first few rows of simulated data for the first standard deviation
            print("\n" + "-"*80 + "\n")
