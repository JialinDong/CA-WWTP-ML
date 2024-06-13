'''
Pipeline:
Train a binary classifier for the first label.
Evaluate the model and print the metrics.
Use the predictions of the first model as a feature to train the next model for the second label.
Evaluate the second model and print the metrics.
Continue this process for all labels.
Finally, evaluate the overall multi-label model performance.
The following Python code demonstrates this process:
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import hamming_loss, precision_recall_fscore_support
import warnings
import configargparse

warnings.filterwarnings('ignore')  # Suppress warnings

def load_and_prepare_data(file_name):
    data = pd.read_csv(file_name)
    # Drop the 'Date' column
    data.drop(columns=['Date'], inplace=True)
    # Identifying the feature and label columns
    feature_cols = data.columns[:-39]
    label_cols = data.columns[-39:]
    # Extracting features and labels
    X = data[feature_cols].fillna(0)  # Handling missing values
    y = data[label_cols]
    # Normalizing/Standardizing the feature data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.select_dtypes(include=['float64', 'int64']))
    return X_scaled, y, label_cols

def train_and_evaluate_model(X_train, X_test, y_train, y_test, label_cols, model, param_grid, file):
    y_pred_df = pd.DataFrame(index=y_test.index, columns=label_cols)
    for label in label_cols:
        if y_pred_df.columns.get_loc(label) != 0:
            X_train = np.hstack((X_train, y_train.iloc[:, :y_pred_df.columns.get_loc(label)].values))
            X_test = np.hstack((X_test, y_pred_df.iloc[:, :y_pred_df.columns.get_loc(label)].values))

        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train[label])
        best_clf = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print_evaluation_metrics(best_clf, X_test, y_test[label], label, best_params, file)
        y_pred_df[label] = best_clf.predict(X_test)

    evaluate_overall_model(y_test, y_pred_df, file)

def train_and_evaluate_model_noTuning(X_train, X_test, y_train, y_test, label_cols, model, file):
    y_pred_df = pd.DataFrame(index=y_test.index, columns=label_cols)
    for label in label_cols:
        unique_classes = y_train[label].nunique()
        if unique_classes < 2:
            print(f"Skipping training for label '{label}' as it has less than 2 unique classes in the training set.", file=file)
            y_pred_df[label] = 0  # or handle it differently based on your requirement
            continue

        if y_pred_df.columns.get_loc(label) != 0:
            X_train = np.hstack((X_train, y_train.iloc[:, :y_pred_df.columns.get_loc(label)].values))
            X_test = np.hstack((X_test, y_pred_df.iloc[:, :y_pred_df.columns.get_loc(label)].values))

        # Print default hyperparameters
        default_params = model.get_params()
        print(f"Default hyperparameters for label {label}: {default_params}", file=file)

        # Train the model with default or predefined parameters
        model.fit(X_train, y_train[label])

        print_evaluation_metrics_noTuning(model, X_test, y_test[label], label, file)
        y_pred_df[label] = model.predict(X_test)

    evaluate_overall_model(y_test, y_pred_df, file)

def print_evaluation_metrics(model, X_test, y_test_label, label, best_params, file):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) > 1 else np.zeros(len(X_test))
    print(f"Evaluation metrics for label: {label}", file=file)
    print(f"Best hyperparameters: {best_params}", file=file)  # Print best hyperparameters
    print("Confusion Matrix:\n", confusion_matrix(y_test_label, y_pred), file=file)
    print("Accuracy: ", accuracy_score(y_test_label, y_pred), file=file)
    print("Recall: ", recall_score(y_test_label, y_pred), file=file)
    print("Precision: ", precision_score(y_test_label, y_pred), file=file)
    print("F1 Score: ", f1_score(y_test_label, y_pred), file=file)
    if len(np.unique(y_test_label)) == 2:
        print("ROC AUC: ", roc_auc_score(y_test_label, y_pred_proba), file=file)
    else:
        print("ROC AUC is not defined for this label (only one class present).", file=file)
    print("\n", file=file)

def print_evaluation_metrics_noTuning(model, X_test, y_test_label, label, file):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) > 1 else np.zeros(len(X_test))
    print(f"Evaluation metrics for label: {label}", file=file)
    print("Confusion Matrix:\n", confusion_matrix(y_test_label, y_pred), file=file)
    print("Accuracy: ", accuracy_score(y_test_label, y_pred), file=file)
    print("Recall: ", recall_score(y_test_label, y_pred), file=file)
    print("Precision: ", precision_score(y_test_label, y_pred), file=file)
    print("F1 Score: ", f1_score(y_test_label, y_pred), file=file)
    if len(np.unique(y_test_label)) == 2:
        print("ROC AUC: ", roc_auc_score(y_test_label, y_pred_proba), file=file)
    else:
        print("ROC AUC is not defined for this label (only one class present).", file=file)
    print("\n", file=file)

def evaluate_overall_model(y_test, y_pred_df, file):
    print("Overall Multi-label Model Evaluation", file=file)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_df),
        'Hamming Loss': hamming_loss(y_test, y_pred_df),
        'Macro Precision': precision_recall_fscore_support(y_test, y_pred_df, average='macro')[0],
        'Macro Recall': precision_recall_fscore_support(y_test, y_pred_df, average='macro')[1],
        'Macro F1 Score': precision_recall_fscore_support(y_test, y_pred_df, average='macro')[2],
        'Micro Precision': precision_recall_fscore_support(y_test, y_pred_df, average='micro')[0],
        'Micro Recall': precision_recall_fscore_support(y_test, y_pred_df, average='micro')[1],
        'Micro F1 Score': precision_recall_fscore_support(y_test, y_pred_df, average='micro')[2],
        'Weighted Precision': precision_recall_fscore_support(y_test, y_pred_df, average='weighted')[0],
        'Weighted Recall': precision_recall_fscore_support(y_test, y_pred_df, average='weighted')[1],
        'Weighted F1 Score': precision_recall_fscore_support(y_test, y_pred_df, average='weighted')[2]
    }
    for metric, value in metrics.items():
        print(f'{metric}: {value}', file=file)

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--which_data", type=str, default='LGBM_4', help='')
    # RF_1 XGB_2 catboost_3 LGBM_4
    return parser

def main():
    parser = config_parser()
    args = parser.parse_args()

    file_name = args.which_data
    print('file_name', file_name)

    X_scaled, y, label_cols = load_and_prepare_data("INF_MultiLabel.csv")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    with open('INF_4_models_catboost .txt', 'w') as file:
        if file_name == 'RF_1':
            print("INF_RF", file=file)
            param_grid = {
                'n_estimators': [10, 50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            train_and_evaluate_model(X_train, X_test, y_train, y_test, label_cols, RandomForestClassifier(random_state=42), param_grid, file)

        elif file_name == 'XGB_2':
            print("INF_XGB", file=file)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'subsample': [0.7, 0.8, 0.9]
            }
            train_and_evaluate_model(X_train, X_test, y_train, y_test, label_cols, XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), param_grid, file)

        elif file_name == 'catboost_3':
            print("INF_Catboost", file=file)
            # CatBoost parameter grid
            param_grid = {
                'iterations': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [4, 6, 8],
                'l2_leaf_reg': [1, 3, 5]
            }
            # CatBoost training and evaluation - NO TUNING (using default)
            train_and_evaluate_model_noTuning(X_train, X_test, y_train, y_test, label_cols, CatBoostClassifier(random_seed=42, verbose=False), file)

        elif file_name == 'LGBM_4':
            print("INF_LightGBM", file=file)
            # LightGBM parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 62, 127],
                'max_depth': [3, 6, 9],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            # LightGBM training and evaluation
            train_and_evaluate_model(X_train, X_test, y_train, y_test, label_cols,LGBMClassifier(random_state=42, verbosity=-1), param_grid, file)

if __name__ == "__main__":
    main()

print("----------------------------------------------------------------------------")



