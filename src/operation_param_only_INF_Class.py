import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger().setLevel(logging.WARNING)

import seaborn as sns
import os
import re
import random
import math
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

# Read in validated data
df = pd.read_csv("data_processed.csv")

# data processing for the training and test model

# Define basic features
basic_features = ['Year', 'Month', 'DISCHARGE VOLUME', 'INFLUENT VOLUME', 'Industrial_Total', 'Ammonia, Total (as N)_INF',
'Biochemical Oxygen Demand (BOD) (5-day @ 20 Deg. C)_INF', 'Carbonaceous Biochemical Oxygen Demand (CBOD) (5-day @ 20 Deg. C)_INF',
'Flow_INF', 'Total Dissolved Solids (TDS)_INF', 'Total Organic Carbon (TOC)_INF', 'Total Suspended Solids (TSS)_INF', 'pH_INF', 'PFAS_total_INF',
'PFAS_total_EFF', 'PFAS_total_BIO']

#
# df_inf = df.copy()
df_inf = df.copy()[basic_features]

# label data PFAS_total_INF-> INF_label
df_inf.loc[(df_inf.PFAS_total_INF < 70), 'INF_label'] = 0
df_inf.loc[(df_inf.PFAS_total_INF >= 70 ),'INF_label'] = 1

df_inf = df_inf.drop(columns=['PFAS_total_INF', 'PFAS_total_EFF', 'PFAS_total_BIO'])  # 'PFAS_total_EFF',
print(df_inf['INF_label'].value_counts())   # get the category & their numbers in the label col
df_inf = df_inf.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# split into X and y matrices
num_col = df_inf.shape[1]   # 1 - return number of cols
print('num_col', num_col)
X = df_inf.iloc[:, :-1]     # delete .values for rf importance
y = df_inf.iloc[:, -1]

# Function to train and evaluate CatBoost model with cross-validation
def train_evaluate_catboost_cv(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dealing with Imbalance of Classes in the Data
    print('Original class distribution:', Counter(y_train))
    oversample = SMOTE(random_state=42)
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    print('After SMOTE:', Counter(y_train))

    # Define CatBoost classifier and parameter grid
    catboost_model = CatBoostClassifier(silent=True, random_state=42)
    param_grid = {
        'iterations': [500, 1000, 1500],
        'learning_rate': [0.05, 0.1, 0.5, 0.8]
    }

    # Perform GridSearchCV
    clf = GridSearchCV(catboost_model, param_grid, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)

    print(f"Best parameters for CatBoost EFF: {clf.best_params_}")

    # Perform 5-fold cross-validation
    cv_results = cross_validate(
        clf.best_estimator_, X, y, cv=5, scoring=['accuracy', 'recall_macro', 'precision_macro', 'f1_macro', 'roc_auc']
    )

    # Display cross-validation results
    print("5-Fold Cross-Validation Results:")
    for metric in cv_results:
        print(f"{metric}: {cv_results[metric].mean():.4f} (+/- {cv_results[metric].std():.4f})")

    # Train the model on full training data with best params
    clf_best = clf.best_estimator_
    clf_best.fit(X_train, y_train)

    # Make predictions
    y_pred = clf_best.predict(X_test)
    y_proba = clf_best.predict_proba(X_test)[:, 1]

    # Calculate and print final metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    auroc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)

    print("Final Scores for CatBoost EFF:")
    print(report)
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1 Score: {f1}")
    print(f"AUROC: {auroc}")

    # Save the trained model
    with open('CatBoost_inf_web_operation_param.pkl', 'wb') as f:
        pickle.dump(clf_best, f)

# Train and evaluate CatBoost with cross-validation for EFF
X = df_inf.iloc[:, :-1]
y = df_inf.iloc[:, -1]
train_evaluate_catboost_cv(X, y)

print('done training and saved the best model')



