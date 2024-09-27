# Import packages

# Packages for data manipulation
import pandas as pd
import numpy as np

# Display all of the columns in dataframes
pd.set_option('display.max_columns', None)

# Visualization packages 
import matplotlib.pyplot as plt
import seaborn as sns

# Data modeling
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

from xgboost import XGBClassifier
from xgboost import plot_importance

# Metrics 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
    f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score

# Statistical manipulations
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Serializing and deserializing Python objects 
import pickle



# Define a function that would collect model performance data
def make_results(model_name: str, model_object, X_test, y_test, res):
    '''
    In:
        model_name:   name of the model as a string (e.g., "Logistic Regression", "Random Forest")
        model_object: a trained model object with a 'predict' method and optionally a 'predict_proba' method
        X_test:       test data features to evaluate the model on
        y_test:       actual labels for the test data
        res:          existing DataFrame to append the model performance metrics to

    Out:
        Updated DataFrame with new row containing the model's performance metrics:
            - Accuracy
            - Precision
            - Recall
            - F1 score
            - AUC (if applicable, otherwise None)
    '''
    # Predict the probabilities if model supports it, else just predict class labels
    if hasattr(model_object, 'predict_proba'):
        y_pred_proba = model_object.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class (1)
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        auc = None  # AUC is not applicable if the model doesn't support predict_proba
    
    # Predict the class labels
    y_pred = model_object.predict(X_test)
    
    # Calculate the metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Create a new DataFrame with the metrics
    new_result = pd.DataFrame({
        'Model': [model_name],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1': [f1],
        'AUC': [auc]
    })
    
    # Use pd.concat to add the new results to the res dataframe
    res = pd.concat([res, new_result], ignore_index=True)
    
    return res



# Define functions to pickle the model and read in the model.
def write_pickle(model_object, save_as:str):
    '''
    In: 
        model_object: a model you want to pickle
        save_as:      filename for how you want to save the model

    Out: A call to pickle the model in the folder indicated
    '''    

    with open('models/' + save_as + '.pickle', 'wb') as to_write:
        pickle.dump(model_object, to_write)
        
        
def read_pickle(saved_model_name:str):
    '''
    In: 
        saved_model_name: filename of pickled model you want to read in

    Out: 
        model: the pickled model 
    '''
    with open('models/' + saved_model_name + '.pickle', 'rb') as to_read:
        model = pickle.load(to_read)

    return model