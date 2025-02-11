import os
import sys
from sklearn.preprocessing import LabelEncoder 
sys.path.append('/work2/wlp/DHMOC/DHMOC/DHMOC_new/code')
from DHMOC_classifer_select import LazyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
from scipy import sparse

# Create output directory to save results
output_dir = '/work2/wlp/DHMOC/DHMOC/DHMOC_new/code'
os.makedirs(output_dir, exist_ok=True)

# Define data file paths
data_files = [
    '/work2/wlp/DHMOC/compare_method/lazypredict/myself/data/xc_data/Tintori_stage.csv',
    # '/work2/wlp/GAENLRR/single-cell data/Deep single cell/scziDesk_h5/Adam/data.h5',
    # '/work2/wlp/GAENLRR/data/Faecal.csv',
    # '/work2/wlp/GAENLRR/data/GC.csv',
    # '/work2/wlp/GAENLRR/data/BRCA.csv',
    # '/work2/wlp/GAENLRR/data/NCI-RNA.csv',
    # '/work2/wlp/GAENLRR/data/Lymphoid.csv'
]

# Define classifier class
# class RandomForestWrapper(RandomForestClassifier):
#     def __init__(self):
#         super().__init__(n_estimators=100)

# class LogisticRegressionWrapper(LogisticRegression):
#     def __init__(self):
#         super().__init__(max_iter=1000)

# class LGBMWrapper(LGBMClassifier):
#     def __init__(self):
#         super().__init__(n_estimators=100)

# class AdaBoostWrapper(AdaBoostClassifier):
#     def __init__(self):
#         super().__init__(n_estimators=100)

class ExtraTreesWrapper(ExtraTreesClassifier):
    def __init__(self):
        super().__init__(n_estimators=100)

# Specific classifiers to be used
specific_classifiers = [
    # ('RandomForestClassifier', lambda: RandomForestClassifier(n_estimators=100)),
    # ('LogisticRegression', lambda: LogisticRegression(max_iter=1000)),
    # ('LGBMClassifier', lambda: LGBMClassifier(n_estimators=100)),
    # ('AdaBoostClassifier', lambda: AdaBoostClassifier(n_estimators=100)),
    # ('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis),
    ('ExtraTreesClassifier', lambda: ExtraTreesClassifier(n_estimators=100))
]

# Process each dataset
for data_file in data_files:
    dataset_name = os.path.splitext(os.path.basename(data_file))[0]
    print(f"\nProcessing dataset: {dataset_name}")
    if dataset_name == 'data':
        dataset_name = os.path.basename(os.path.dirname(data_file))
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Read h5 file
        with h5py.File(data_file, 'r') as f:
            # Build sparse matrix
            data = sparse.csr_matrix((f['exprs/data'][:], 
                                    f['exprs/indices'][:], 
                                    f['exprs/indptr'][:]),
                                    shape=f['exprs/shape'][:])
            
            # Convert to dense matrix
            X = data.toarray()
            
            # Read cell type labels
            cell_types = np.array(f['obs/cell_ontology_class']).astype(str)
            cell_unique_types = np.unique(cell_types)
            y = f['obs/cell_ontology_class'][:]
            # If y is of byte type, convert to string
            if isinstance(y[0], bytes):
                y = [x.decode() for x in y]
    elif dataset_name == 'Tintori_stage':
        # Read CSV file
        df = pd.read_csv(data_file)
        
        X = df.iloc[:, 1:].values  # Use columns from the 2nd column to the last column as features
        y = df.iloc[:, 0].values   # Use the first column as labels

        cell_unique_types = np.unique(y)
        
    else:
        # Read CSV file
        df = pd.read_csv(data_file)
        
        X = df.iloc[:, 2:].values  # Use columns from the 3rd column to the last column as features
        y = df.iloc[:, 1].values   # Use the second column as labels
        cell_unique_types = np.unique(y)

    # Convert labels to numeric
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
    # Initialize LazyClassifier with specific classifiers
    clf = LazyClassifier(verbose=0, 
                        ignore_warnings=True, 
                        custom_metric=None,
                        classifiers=specific_classifiers)
    # models, predictions = clf.fit(dataset_name,X_train, X_test, y_train, y_test)
    models, predictions = clf.fit(dataset_name,cell_unique_types,X_train, X_test, y_train, y_test)
    # Sort results by accuracy and time taken
    models = models.sort_values(['Accuracy', 'Time Taken'], ascending=[False, True])

    # Create result file name directly from dataset_name
    result_filename = f"{dataset_name}_results.csv"  # e.g. 'Faecal_results.csv'
    result_file_path = os.path.join(output_dir, result_filename)
    print(models)
    
    # Save the result to CSV
    models.to_csv(result_file_path)

    print(f"Results saved to: {result_file_path}")
