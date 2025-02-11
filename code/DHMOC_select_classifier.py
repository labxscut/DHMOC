import os
import sys
from sklearn.preprocessing import LabelEncoder  # Add this line
sys.path.append('/work2/wlp/DHMOC/DHMOC/DHMOC_new/code')
from DHMOC_selecte_classifer_2 import LazyClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Create a directory to save the results
output_dir = '/work2/wlp/DHMOC/DHMOC/DHMOC_new/result'
os.makedirs(output_dir, exist_ok=True)

# Define data file paths
data_files = [
    '/work2/wlp/GAENLRR/data/HCC.csv'
]

# Process each dataset
for data_file in data_files:
    dataset_name = os.path.splitext(os.path.basename(data_file))[0]
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Read the CSV file
    df = pd.read_csv(data_file)
    
    X = df.iloc[:, 4:].values  # Use columns from the 4th to the last column as features
    y = df.iloc[:, 2].values   # Use the second column as the label (type)

    cell_unique_types = np.unique(y)
    # Convert labels to numeric values
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    # Use LazyClassifier to compare models
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(dataset_name, cell_unique_types, X_train, X_test, y_train, y_test)
    
    # Format the results
    models = models.sort_values(['Accuracy', 'Time Taken'], ascending=[False, True])

    # Directly create the result filename using dataset_name
    result_filename = f"{dataset_name}_3_results.csv"  # e.g., 'Faecal_results.csv'
    result_file_path = os.path.join(output_dir, result_filename)
    print(models)
    
    # Save the result to CSV
    models.to_csv(result_file_path)

    print(f"Results saved to: {result_file_path}")
