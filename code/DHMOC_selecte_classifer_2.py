"""
Supervised Models
"""
import textwrap
# Author: Shankar Rao Pandala <shankar.pandala@live.com>
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import os
# Add necessary imports
from scipy import io as sio

import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
import time
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,  # Added
    recall_score,    # Added
    auc,            # Added
    roc_curve,       # Added
    r2_score,
    mean_squared_error,
)
import warnings
import xgboost
import lightgbm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Filter warnings
warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

def wrap_labels_logically(labels):
    """Intelligently wrap labels based on spaces"""
    wrapped = []
    for label in labels:
        # If it's a numeric type, convert to string
        if isinstance(label, (int, np.integer, float, np.floating)):
            label = str(label)  # Ensure it's a string type
        # Split words by spaces
        words = label.split()
        
        if len(words) <= 2:  # If there are only 1-2 words, don't wrap
            wrapped_label = label
        else:  # If more than 2 words, wrap every 2 words
            wrapped_label = ''
            for i in range(0, len(words), 2):
                if i + 2 <= len(words):
                    wrapped_label += ' '.join(words[i:i+2]) + '\n'
                else:
                    wrapped_label += ' '.join(words[i:])
            wrapped_label = wrapped_label.rstrip('\n')  # Remove the last newline
            
        wrapped.append(wrapped_label)
    return wrapped


def get_dataset_colors(Z, cell_unique_types):
    # Define a soft color palette
    color_palette = [
        '#F4A7A1',  # Soft red
        '#F1BE98',  # Soft orange
        '#A8D8E8',  # Soft blue
        '#F3E5AB',  # Soft yellow
        '#98D7C2',  # Soft green
        '#B4C7E7',  # Soft purple-blue
        '#E6E6E6',  # Soft gray
        '#D7B5D8',  # Soft purple
        '#BBE3A7',  # Soft light green
        '#FFB6C1',  # Light pink
        '#DDA0DD'   # Plum
    ]
    # Ensure there are enough colors
    if len(cell_unique_types) > len(color_palette):
        # If the number of cell types exceeds predefined colors, cycle through them
        color_palette = color_palette * (len(cell_unique_types) // len(color_palette) + 1)
    
    # Create a mapping from cell types to colors
    colors = dict(zip(cell_unique_types, color_palette[:len(cell_unique_types)]))
    
    # Define a function to get colors
    def get_color(k):
        # If it's a leaf node
        if k < len(cell_unique_types):
            return colors[cell_unique_types[k]]
        
        # If it's an internal node, use the color of the left child node
        left = int(Z[k-len(cell_unique_types), 0])
        return get_color(left)
    
    return colors, get_color

# List of removed classifiers and regressors (no changes here)
removed_classifiers = [
    "LightGBM",
    "ClassifierChain",
    "ComplementNB",
    "GradientBoostingClassifier",
    "GaussianProcessClassifier",
    "HistGradientBoostingClassifier",
    "MLPClassifier",
    "LogisticRegressionCV", 
    "MultiOutputClassifier", 
    "MultinomialNB", 
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "RadiusNeighborsClassifier",
    "VotingClassifier",
]

removed_regressors = [
    "TheilSenRegressor",
    "ARDRegression", 
    "CCA", 
    "IsotonicRegression", 
    "StackingRegressor",
    "MultiOutputRegressor", 
    "MultiTaskElasticNet", 
    "MultiTaskElasticNetCV", 
    "MultiTaskLasso", 
    "MultiTaskLassoCV", 
    "PLSCanonical", 
    "PLSRegression", 
    "RadiusNeighborsRegressor", 
    "RegressorChain", 
    "VotingRegressor", 
]

# Define classifiers and regressors
CLASSIFIERS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], ClassifierMixin) and (est[0] not in removed_classifiers))
]

REGRESSORS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], RegressorMixin) and (est[0] not in removed_regressors))
]

# Add new classifiers and regressors
REGRESSORS.append(("XGBRegressor", xgboost.XGBRegressor))
REGRESSORS.append(("LGBMRegressor", lightgbm.LGBMRegressor))
# REGRESSORS.append(('CatBoostRegressor',catboost.CatBoostRegressor))

CLASSIFIERS.append(("XGBClassifier", xgboost.XGBClassifier))
CLASSIFIERS.append(("LGBMClassifier", lightgbm.LGBMClassifier))
# CLASSIFIERS.append(('CatBoostClassifier',catboost.CatBoostClassifier))

# Add Lasso-Logistic and SVM classifiers
CLASSIFIERS.append(("Lasso-Logistic", LogisticRegression(penalty='l1', solver='saga')))
CLASSIFIERS.append(("SVC", SVC()))

# Define transformers for data preprocessing
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

categorical_transformer_low = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

categorical_transformer_high = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OrdinalEncoder()),
    ]
)

def plot_roc_curves(y_true, y_pred_proba, labels, colors, dataset_name, 
                   classifier_name, save_dir):
    """Plot ROC curves"""
    plt.figure(figsize=(15, 6))
    
    # Training set ROC curve
    plt.subplot(1, 2, 1)
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, '-', 
                label=f'{label} (AUC = {roc_auc:.2f})',
                color=colors[label])
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Training Set ROC Curves - {name}')
    plt.legend(loc="lower right", fontsize='small')

    plt.close()
    return filename

# Helper function for splitting data (not fully shown in the code above)
def get_card_split(df, cols, n=11):
    """
    Splits categorical columns into 2 lists based on cardinality (i.e # of unique values)
    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame from which the cardinality of the columns is calculated.
    cols : list-like
        Categorical columns to list
    n : int, optional (default=11)
        The value of 'n' will be used to split columns.
    Returns
    -------
    card_low : list-like
        Columns with cardinality < n
    card_high : list-like
        Columns with cardinality >= n
        Distinguish between low and high cardinality features;
        Helps in choosing the appropriate encoding method (e.g., one-hot encoding is suitable for low cardinality features)
    """
    cond = df[cols].nunique() > n
    card_high = cols[cond]
    card_low = cols[~cond]
    return card_low, card_high

# Helper class for performing classification

####################### Classification ##################################
class LazyClassifier:
    """
    This module helps in fitting to all the classification algorithms that are available in Scikit-learn
    Parameters
    ----------
    verbose : int, optional (default=0)
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    ignore_warnings : bool, optional (default=True)
        When set to True, the warning related to algorithms that are not able to run are ignored.
    custom_metric : function, optional (default=None)
        When function is provided, models are evaluated based on the custom evaluation metric provided.
    prediction : bool, optional (default=False)
        When set to True, the predictions of all the models are returned as a dataframe.
    classifiers : list, optional (default="all")
        When function is provided, trains the chosen classifier(s).

    Examples
    --------
    >>> from lazypredict.Supervised import LazyClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_breast_cancer()
    >>> X = data.data
    >>> y= data.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)
    >>> clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    >>> models,predictions = clf.fit(X_train, X_test, y_train, y_test)
    >>> model_dictionary = clf.provide_models(X_train,X_test,y_train,y_test)
    >>> models
    | Model                          |   Accuracy |   Balanced Accuracy |   ROC AUC |   F1 Score |   Time Taken |
    |:-------------------------------|-----------:|--------------------:|----------:|-----------:|-------------:|
    | LinearSVC                      |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0150008 |
    | SGDClassifier                  |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0109992 |
    | MLPClassifier                  |   0.985965 |            0.986904 |  0.986904 |   0.985994 |    0.426     |
    | Perceptron                     |   0.985965 |            0.984797 |  0.984797 |   0.985965 |    0.0120046 |
    | LogisticRegression             |   0.985965 |            0.98269  |  0.98269  |   0.985934 |    0.0200036 |
    | LogisticRegressionCV           |   0.985965 |            0.98269  |  0.98269  |   0.985934 |    0.262997  |
    | SVC                            |   0.982456 |            0.979942 |  0.979942 |   0.982437 |    0.0140011 |
    | CalibratedClassifierCV         |   0.982456 |            0.975728 |  0.975728 |   0.982357 |    0.0350015 |
    | PassiveAggressiveClassifier    |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0130005 |
    | LabelPropagation               |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0429988 |
    | LabelSpreading                 |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0310006 |
    | RandomForestClassifier         |   0.97193  |            0.969594 |  0.969594 |   0.97193  |    0.033     |
    | GradientBoostingClassifier     |   0.97193  |            0.967486 |  0.967486 |   0.971869 |    0.166998  |
    | QuadraticDiscriminantAnalysis  |   0.964912 |            0.966206 |  0.966206 |   0.965052 |    0.0119994 |
    | HistGradientBoostingClassifier |   0.968421 |            0.964739 |  0.964739 |   0.968387 |    0.682003  |
    | RidgeClassifierCV              |   0.97193  |            0.963272 |  0.963272 |   0.971736 |    0.0130029 |
    | RidgeClassifier                |   0.968421 |            0.960525 |  0.960525 |   0.968242 |    0.0119977 |
    | AdaBoostClassifier             |   0.961404 |            0.959245 |  0.959245 |   0.961444 |    0.204998  |
    | ExtraTreesClassifier           |   0.961404 |            0.957138 |  0.957138 |   0.961362 |    0.0270066 |
    | KNeighborsClassifier           |   0.961404 |            0.95503  |  0.95503  |   0.961276 |    0.0560005 |
    | BaggingClassifier              |   0.947368 |            0.954577 |  0.954577 |   0.947882 |    0.0559971 |
    | BernoulliNB                    |   0.950877 |            0.951003 |  0.951003 |   0.951072 |    0.0169988 |
    | LinearDiscriminantAnalysis     |   0.961404 |            0.950816 |  0.950816 |   0.961089 |    0.0199995 |
    | GaussianNB                     |   0.954386 |            0.949536 |  0.949536 |   0.954337 |    0.0139935 |
    | NuSVC                          |   0.954386 |            0.943215 |  0.943215 |   0.954014 |    0.019989  |
    | DecisionTreeClassifier         |   0.936842 |            0.933693 |  0.933693 |   0.936971 |    0.0170023 |
    | NearestCentroid                |   0.947368 |            0.933506 |  0.933506 |   0.946801 |    0.0160074 |
    | ExtraTreeClassifier            |   0.922807 |            0.912168 |  0.912168 |   0.922462 |    0.0109999 |
    | CheckingClassifier             |   0.361404 |            0.5      |  0.5      |   0.191879 |    0.0170043 |
    | DummyClassifier                |   0.512281 |            0.489598 |  0.489598 |   0.518924 |    0.0119965 |
    """

    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        random_state=42,
        classifiers="all",
    ):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state
        self.classifiers = classifiers

    def fit(self, dataset_name, cell_unique_types, X_train, X_test, y_train, y_test):
        """Fit Classification algorithms to X_train and y_train, predict and score on X_test, y_test.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        scores : Pandas DataFrame
            Returns metrics of all the models in a Pandas DataFrame.
        predictions : Pandas DataFrame
            Returns predictions of all the models in a Pandas DataFrame.
        """
        Accuracy = []
        B_Accuracy = []
        ROC_AUC = []
        F1 = []
        Precision = []  # Added
        Recall = []     # Added
        names = []
        TIME = []
        predictions = {}
        # Get dataset information
       
        if cell_unique_types is None:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        if self.custom_metric is not None:
            CUSTOM_METRIC = []

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = get_card_split(
            X_train, categorical_features
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )

        if self.classifiers == "all":
            self.classifiers = CLASSIFIERS
        else:
            try:
                temp_list = []
                for classifier in self.classifiers:
                    full_name = (classifier.__name__, classifier)
                    temp_list.append(full_name)
                self.classifiers = temp_list
            except Exception as exception:
                print(exception)
                print("Invalid Classifier(s)")
        
        for name, model in tqdm(self.classifiers):
            start = time.time()
            try:
                # Create model instance (only create once)
                if "random_state" in model().get_params().keys():
                    # For models that support random_state
                    if name == "Lasso-Logistic":
                        model_instance = model(random_state=self.random_state, penalty='l1', solver='saga')
                    elif name == "SVC":
                        model_instance = model(random_state=self.random_state)
                    else:
                        model_instance = model()
                else:
                    # For models that do not support random_state
                    if name == "Lasso-Logistic":
                        model_instance = model(penalty='l1', solver='saga')
                    elif name == "SVC":
                        model_instance = model()
                    else:
                        model_instance = model()    
                # Use the created model instance
                pipe = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("classifier", model_instance)
                    ]
                )

                pipe.fit(X_train, y_train)
                self.models[name] = pipe
                y_pred = pipe.predict(X_test)
                # Use predict method to get predictions
                # 2. If probability prediction is needed
                train_pred_proba = pipe.predict_proba(X_train)  # Shape should be (n_samples, n_classes)
                test_pred_proba = pipe.predict_proba(X_test)
                all_predictions = np.vstack([train_pred_proba, test_pred_proba])
                final_predictions = np.argmax(all_predictions, axis=1)  # Now axis=1 makes sense
                # Combine true labels of training and test sets
                true_labels = np.concatenate([y_train, y_test])
                
                le = LabelEncoder()
                le.fit(cell_unique_types)  # Train encoder with fixed order of labels
                print("Label encoding mapping:")
                for i, label in enumerate(le.classes_):
                    print(f"{label} -> {i}")

                # 1. Get label names
                label_names = le.classes_  # Get label names from LabelEncoder

                # Calculate confusion matrix
                # conf_mat = confusion_matrix(true_labels, final_predictions)
                # Normalize each row (divide by the max value of the row)
                normalized_A = conf_mat / conf_mat.max(axis=1)[:, np.newaxis]

                # Construct symmetric similarity matrix
                symmetric_affinity_matrix = (normalized_A + normalized_A.T) / 2

                # Round to 3 decimal places
                symmetric_affinity_matrix = np.round(symmetric_affinity_matrix, decimals=3)
                affinity_matrix = symmetric_affinity_matrix

                affinity_matrix[affinity_matrix == 0] = 0.0001
                # Calculate distance matrix
                distance_matrix = 1 / affinity_matrix
                normalized_distance_matrix = distance_matrix / distance_matrix.max(axis=1)[:, np.newaxis]
                # Construct symmetric distance matrix
                symmetric_distance_matrix = (normalized_distance_matrix + normalized_distance_matrix.T) / 2
                symmetric_distance_matrix = np.round(symmetric_distance_matrix, decimals=3)
                # Set diagonal to 0
                np.fill_diagonal(symmetric_distance_matrix, 0)
                print(symmetric_distance_matrix)

                # Perform hierarchical clustering
                # Note: scipy's linkage function requires a condensed distance matrix format
                # Since the matrix is symmetric, we only need the upper triangle
                condensed_dist = symmetric_distance_matrix[np.triu_indices(symmetric_distance_matrix.shape[0], k=1)]
                Z = linkage(condensed_dist, method='average')
                
                fig = plt.figure(figsize=(12, 8), dpi=1200)  # Set high resolution

                colors, get_color = get_dataset_colors(Z, cell_unique_types)
                
                # Plot dendrogram
                plt.figure(figsize=(12, 8))
                dendrogram(
                    Z,
                    labels=wrap_labels_logically(cell_unique_types),  # Ensure correct labels are used
                    leaf_rotation=0,  # No rotation for labels
                    leaf_font_size=12,  # Adjust font size for clarity
                    truncate_mode='level',  # Truncate mode
                    p=5,  # Show the last p merged clusters
                    link_color_func=get_color,  # Apply color function
                    above_threshold_color='black',  # Set color above threshold line to black
                    distance_sort=True,
                    no_plot=False,
                    count_sort=False
                )
                dataset_part = dataset_name.lower().replace('-', '_')
                classifier_part = name.lower().replace(' ', '_')  # name is the classifier name
                plt.title(f'{dataset_part}_({classifier_part})')

                plt.tight_layout()  # Automatically adjust subplot parameters to fill the figure area
                plt.show()
                
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                # ax.get_xaxis().set_visible(False)        # Hide x-axis
                ax.get_yaxis().set_visible(False)        # Hide y-axis
                # # Hide ticks but keep labels
                ax.tick_params(left=False, bottom=False, labelbottom=True)  # Hide tick marks but keep bottom labels

                # 1. Set the full save path
                save_dir = '/work2/wlp/DHMOC/DHMOC/DHMOC_new/result/'
                
                dendrogram_filename = os.path.join(save_dir, f'{dataset_part}_{classifier_part}_dendrogram.pdf')
                plt.savefig(dendrogram_filename, format='pdf', bbox_inches='tight', dpi=300)
                plt.close()
                print(f"Dendrogram saved to: {dendrogram_filename}")

                # Get prediction probabilities for training and test sets
                train_probs = pipe.predict_proba(X_train)
                test_probs = pipe.predict_proba(X_test)

                # If cell_unique_types is a numpy array, convert to list
                cell_unique_types_list = cell_unique_types.tolist() if isinstance(cell_unique_types, np.ndarray) else cell_unique_types

                # Prepare MATLAB format data structure
                mat_data = {
                    'train_data': {
                        'fpr': [],
                        'tpr': [],
                        'auc': [],
                        'labels': cell_unique_types_list
                    },
                    'test_data': {
                        'fpr': [],
                        'tpr': [],
                        'auc': [],
                        'labels': cell_unique_types_list
                    },
                    'classifier_name': name
                }

                # Calculate and store ROC data for training and test sets
                for i, label in enumerate(cell_unique_types_list):
                    # Training set data
                    fpr_train, tpr_train, _ = roc_curve(
                        (y_train == i).astype(int),
                        train_probs[:, i]
                    )
                    train_auc = auc(fpr_train, tpr_train)
                    
                    mat_data['train_data']['fpr'].append(fpr_train)
                    mat_data['train_data']['tpr'].append(tpr_train)
                    mat_data['train_data']['auc'].append(train_auc)
                    
                    # Test set data
                    fpr_test, tpr_test, _ = roc_curve(
                        (y_test == i).astype(int),
                        test_probs[:, i]
                    )
                    test_auc = auc(fpr_test, tpr_test)
                    
                    mat_data['test_data']['fpr'].append(fpr_test)
                    mat_data['test_data']['tpr'].append(tpr_test)
                    mat_data['test_data']['auc'].append(test_auc)

                # # Convert to numpy arrays to ensure MATLAB compatibility
                mat_data['train_data']['fpr'] = np.array(mat_data['train_data']['fpr'], dtype=object)
                mat_data['train_data']['tpr'] = np.array(mat_data['train_data']['tpr'], dtype=object)
                mat_data['train_data']['auc'] = np.array(mat_data['train_data']['auc'])
                mat_data['test_data']['fpr'] = np.array(mat_data['test_data']['fpr'], dtype=object)
                mat_data['test_data']['tpr'] = np.array(mat_data['test_data']['tpr'], dtype=object)
                mat_data['test_data']['auc'] = np.array(mat_data['test_data']['auc'])

                # Plot graphs
                plt.figure(figsize=(15, 6))

                # 1. ROC curve for training set (left plot)
                plt.subplot(1, 2, 1)
                for i, label in enumerate(cell_unique_types_list):
                    plt.plot(mat_data['train_data']['fpr'][i], 
                            mat_data['train_data']['tpr'][i], '-',
                            label=f'{label} (AUC = {mat_data["train_data"]["auc"][i]:.2f})',
                            color=colors[label])

                plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'Training Set ROC Curves - {name}')
                plt.legend(loc="lower right", fontsize='small')

                # 2. ROC curve for test set (right plot)
                plt.subplot(1, 2, 2)
                for i, label in enumerate(cell_unique_types_list):
                    plt.plot(mat_data['test_data']['fpr'][i], 
                            mat_data['test_data']['tpr'][i], '-',
                            label=f'{label} (AUC = {mat_data["test_data"]["auc"][i]:.2f})',
                            color=colors[label])

                plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'Testing Set ROC Curves - {name}')
                plt.legend(loc="lower right", fontsize='small')

                plt.tight_layout()
                roc_filename = os.path.join(save_dir, f'{dataset_part}_{classifier_part}_roc.pdf')
                plt.savefig(roc_filename, format='pdf', bbox_inches='tight', dpi=300)
                plt.close()
                print(f"ROC plot saved to: {roc_filename}")

                print("Calculating evaluation metrics")
                accuracy = accuracy_score(y_test, y_pred, normalize=True)
                b_accuracy = balanced_accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                precision = precision_score(y_test, y_pred, average='weighted')  # Added
                recall = recall_score(y_test, y_pred, average='weighted') 
                print("Evaluation metrics calculated, next calculate ROC AUC")
                try:
                    if len(np.unique(y_test)) == 2:  # Binary classification
                        roc_auc = roc_auc_score(y_test, y_pred)
                    else:  # Multi-class classification
                        roc_auc = roc_auc_score(y_test, pipe.predict_proba(X_test), multi_class='ovr')
                except Exception as exception:
                    roc_auc = None
                    if self.ignore_warnings is False:
                        print("ROC AUC couldn't be calculated for " + name)
                        print(exception)
                names.append(name)
                Accuracy.append(accuracy)
                B_Accuracy.append(b_accuracy)
                ROC_AUC.append(roc_auc)
                F1.append(f1)
                Precision.append(precision)  # Added
                Recall.append(recall)        # Added
                TIME.append(time.time() - start)
                if self.custom_metric is not None:
                    custom_metric = self.custom_metric(y_test, y_pred)
                    CUSTOM_METRIC.append(custom_metric)
                if self.verbose > 0:
                    if self.custom_metric is not None:
                        print(
                            {
                                "Model": name,
                                "Accuracy": accuracy,
                                "Balanced Accuracy": b_accuracy,
                                "ROC AUC": roc_auc,
                                "F1 Score": f1,
                                "Precision": precision,
                                "Recall": recall,
                                self.custom_metric.__name__: custom_metric,
                                "Time taken": time.time() - start,
                            }
                        )
                    else:
                        print(
                            {
                                "Model": name,
                                "Accuracy": accuracy,
                                "Balanced Accuracy": b_accuracy,
                                "ROC AUC": roc_auc,
                                "F1 Score": f1,
                                "Precision": precision,
                                "Recall": recall,
                                "Time taken": time.time() - start,
                            }
                        )
                if self.predictions:
                    predictions[name] = y_pred
            except Exception as exception:
                if self.ignore_warnings is False:
                    print(name + " model failed to execute")
                    print(exception)
        if self.custom_metric is None:
            scores = pd.DataFrame(
                {
                    "Model": names,
                    "Accuracy": Accuracy,
                    "Balanced Accuracy": B_Accuracy,
                    "ROC AUC": ROC_AUC,
                    "F1 Score": F1,
                    "Precision": Precision,
                    "Recall": Recall,
                    "Time Taken": TIME,
                }
            )
        else:
            scores = pd.DataFrame(
                {
                    "Model": names,
                    "Accuracy": Accuracy,
                    "Balanced Accuracy": B_Accuracy,
                    "ROC AUC": ROC_AUC,
                    "F1 Score": F1,
                    "Precision": Precision,
                    "Recall": Recall,
                    self.custom_metric.__name__: CUSTOM_METRIC,
                    "Time Taken": TIME,
                }
            )
        scores = scores.sort_values(by="Balanced Accuracy", ascending=False).set_index(
            "Model"
        )

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions)
        return scores, predictions_df if self.predictions is True else scores

    def provide_models(self, X_train, X_test, y_train, y_test):
        """
        This function returns all the model objects trained in fit function.
        If fit is not called already, then we call fit and then return the models.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        models: dict-object,
            Returns a dictionary with each model pipeline as value 
            with key as name of models.
        """
        if len(self.models.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test)

        return self.models

def adjusted_rsquared(r2, n, p):
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))

# Helper class for performing classification

####################### Regression ##################################
class LazyRegressor:
    """
    This module helps in fitting regression models that are available in Scikit-learn
    Parameters
    ----------
    verbose : int, optional (default=0)
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    ignore_warnings : bool, optional (default=True)
        When set to True, the warning related to algorithms that are not able to run are ignored.
    custom_metric : function, optional (default=None)
        When function is provided, models are evaluated based on the custom evaluation metric provided.
    prediction : bool, optional (default=False)
        When set to True, the predictions of all the models are returned as a dataframe.
    regressors : list, optional (default="all")
        When function is provided, trains the chosen regressor(s).

    Examples
    --------
    >>> from lazypredict.Supervised import LazyRegressor
    >>> from sklearn import datasets
    >>> from sklearn.utils import shuffle
    >>> import numpy as np

    >>> diabetes  = datasets.load_diabetes()
    >>> X, y = shuffle(diabetes.data, diabetes.target, random_state=13)
    >>> X = X.astype(np.float32)

    >>> offset = int(X.shape[0] * 0.9)
    >>> X_train, y_train = X[:offset], y[:offset]
    >>> X_test, y_test = X[offset:], y[offset:]

    >>> reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    >>> models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    >>> model_dictionary = reg.provide_models(X_train, X_test, y_train, y_test)
    >>> models
    | Model                         |   Adjusted R-Squared |   R-Squared |     RMSE |   Time Taken |
    |:------------------------------|---------------------:|------------:|---------:|-------------:|
    | ExtraTreesRegressor           |           0.378921   |  0.520076   |  54.2202 |   0.121466   |
    | OrthogonalMatchingPursuitCV   |           0.374947   |  0.517004   |  54.3934 |   0.0111742  |
    | Lasso                         |           0.373483   |  0.515873   |  54.457  |   0.00620174 |
    | LassoLars                     |           0.373474   |  0.515866   |  54.4575 |   0.0087235  |
    | LarsCV                        |           0.3715     |  0.514341   |  54.5432 |   0.0160234  |
    | LassoCV                       |           0.370413   |  0.513501   |  54.5903 |   0.0624897  |
    | PassiveAggressiveRegressor    |           0.366958   |  0.510831   |  54.7399 |   0.00689793 |
    | LassoLarsIC                   |           0.364984   |  0.509306   |  54.8252 |   0.0108321  |
    | SGDRegressor                  |           0.364307   |  0.508783   |  54.8544 |   0.0055306  |
    | RidgeCV                       |           0.363002   |  0.507774   |  54.9107 |   0.00728202 |
    | Ridge                         |           0.363002   |  0.507774   |  54.9107 |   0.00556874 |
    | BayesianRidge                 |           0.362296   |  0.507229   |  54.9411 |   0.0122972  |
    | LassoLarsCV                   |           0.361749   |  0.506806   |  54.9646 |   0.0175984  |
    | TransformedTargetRegressor    |           0.361749   |  0.506806   |  54.9646 |   0.00604773 |
    | LinearRegression              |           0.361749   |  0.506806   |  54.9646 |   0.00677514 |
    | Lars                          |           0.358828   |  0.504549   |  55.0903 |   0.00935149 |
    | ElasticNetCV                  |           0.356159   |  0.502486   |  55.2048 |   0.0478678  |
    | HuberRegressor                |           0.355251   |  0.501785   |  55.2437 |   0.0129263  |
    | RandomForestRegressor         |           0.349621   |  0.497434   |  55.4844 |   0.2331     |
    | AdaBoostRegressor             |           0.340416   |  0.490322   |  55.8757 |   0.0512381  |
    | LGBMRegressor                 |           0.339239   |  0.489412   |  55.9255 |   0.0396187  |
    | HistGradientBoostingRegressor |           0.335632   |  0.486625   |  56.0779 |   0.0897055  |
    | PoissonRegressor              |           0.323033   |  0.476889   |  56.6072 |   0.00953603 |
    | ElasticNet                    |           0.301755   |  0.460447   |  57.4899 |   0.00604224 |
    | KNeighborsRegressor           |           0.299855   |  0.458979   |  57.5681 |   0.00757337 |
    | OrthogonalMatchingPursuit     |           0.292421   |  0.453235   |  57.8729 |   0.00709486 |
    | BaggingRegressor              |           0.291213   |  0.452301   |  57.9223 |   0.0302746  |
    | GradientBoostingRegressor     |           0.247009   |  0.418143   |  59.7011 |   0.136803   |
    | TweedieRegressor              |           0.244215   |  0.415984   |  59.8118 |   0.00633955 |
    | XGBRegressor                  |           0.224263   |  0.400567   |  60.5961 |   0.339694   |
    | GammaRegressor                |           0.223895   |  0.400283   |  60.6105 |   0.0235181  |
    | RANSACRegressor               |           0.203535   |  0.38455    |  61.4004 |   0.0653253  |
    | LinearSVR                     |           0.116707   |  0.317455   |  64.6607 |   0.0077076  |
    | ExtraTreeRegressor            |           0.00201902 |  0.228833   |  68.7304 |   0.00626636 |
    | NuSVR                         |          -0.0667043  |  0.175728   |  71.0575 |   0.0143399  |
    | SVR                           |          -0.0964128  |  0.152772   |  72.0402 |   0.0114729  |
    | DummyRegressor                |          -0.297553   | -0.00265478 |  78.3701 |   0.00592971 |
    | DecisionTreeRegressor         |          -0.470263   | -0.136112   |  83.4229 |   0.00749898 |
    | GaussianProcessRegressor      |          -0.769174   | -0.367089   |  91.5109 |   0.0770502  |
    | MLPRegressor                  |          -1.86772    | -1.21597    | 116.508  |   0.235267   |
    | KernelRidge                   |          -5.03822    | -3.6659     | 169.061  |   0.0243919  |
    """

    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        random_state=42,
        regressors="all",
    ):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state
        self.regressors = regressors

    def fit(self, X_train, X_test, y_train, y_test):
        """Fit Regression algorithms to X_train and y_train, predict and score on X_test, y_test.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        scores : Pandas DataFrame
            Returns metrics of all the models in a Pandas DataFrame.
        predictions : Pandas DataFrame
            Returns predictions of all the models in a Pandas DataFrame.
        """
        R2 = []
        ADJR2 = []
        RMSE = []
        # WIN = []
        names = []
        TIME = []
        predictions = {}

        if self.custom_metric:
            CUSTOM_METRIC = []

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = get_card_split(
            X_train, categorical_features
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )

        if self.regressors == "all":
            self.regressors = REGRESSORS
        else:
            try:
                temp_list = []
                for regressor in self.regressors:
                    full_name = (regressor.__name__, regressor)
                    temp_list.append(full_name)
                self.regressors = temp_list
            except Exception as exception:
                print(exception)
                print("Invalid Regressor(s)")

        for name, model in tqdm(self.regressors):
            start = time.time()
            try:
                if "random_state" in model().get_params().keys():
                    pipe = Pipeline(
                        steps=[
                            ("preprocessor", preprocessor),
                            ("regressor", model(random_state=self.random_state)),
                        ]
                    )
                else:
                    pipe = Pipeline(
                        steps=[("preprocessor", preprocessor), ("regressor", model())]
                    )

                pipe.fit(X_train, y_train)
                self.models[name] = pipe
                y_pred = pipe.predict(X_test)

                r_squared = r2_score(y_test, y_pred)
                adj_rsquared = adjusted_rsquared(
                    r_squared, X_test.shape[0], X_test.shape[1]
                )
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                names.append(name)
                R2.append(r_squared)
                ADJR2.append(adj_rsquared)
                RMSE.append(rmse)
                TIME.append(time.time() - start)

                if self.custom_metric:
                    custom_metric = self.custom_metric(y_test, y_pred)
                    CUSTOM_METRIC.append(custom_metric)

                if self.verbose > 0:
                    scores_verbose = {
                        "Model": name,
                        "R-Squared": r_squared,
                        "Adjusted R-Squared": adj_rsquared,
                        "RMSE": rmse,
                        "Time taken": time.time() - start,
                    }

                    if self.custom_metric:
                        scores_verbose[self.custom_metric.__name__] = custom_metric

                    print(scores_verbose)
                if self.predictions:
                    predictions[name] = y_pred
            except Exception as exception:
                if self.ignore_warnings is False:
                    print(name + " model failed to execute")
                    print(exception)

        scores = {
            "Model": names,
            "Adjusted R-Squared": ADJR2,
            "R-Squared": R2,
            "RMSE": RMSE,
            "Time Taken": TIME,
        }

        if self.custom_metric:
            scores[self.custom_metric.__name__] = CUSTOM_METRIC

        scores = pd.DataFrame(scores)
        scores = scores.sort_values(by="Adjusted R-Squared", ascending=False).set_index(
            "Model"
        )

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions)
        return scores, predictions_df if self.predictions is True else scores

    def provide_models(self, X_train, X_test, y_train, y_test):
        """
        This function returns all the model objects trained in fit function.
        If fit is not called already, then we call fit and then return the models.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        models: dict-object,
            Returns a dictionary with each model pipeline as value 
            with key as name of models.
        """
        if len(self.models.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test)

        return self.models

Regression = LazyRegressor
Classification = LazyClassifier
