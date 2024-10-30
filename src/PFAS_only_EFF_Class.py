
import seaborn as sns
import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


# Read in validated data
df = pd.read_csv("data_processed.csv")

# data processing for the training and test model

## Keep only limited basic features
basic_features = ['PFBA_INF', 'PFPeA_INF', 'PFHxA_INF', 'PFHpA_INF', 'PFOA_INF', 'PFNA_INF',
 'PFDA_INF', 'PFUnA_INF', 'PFDoA_INF', 'PFTrDA_INF', 'PFTA_INF', 'PFHxDA_INF',
    'PFODA_INF', 'FTCA_33_INF', 'FTCA_53_INF', 'FTCA_73_INF', 'FTS_42_INF',
    'FTS_62_INF', 'FTS_82_INF', 'FTS_102_INF', 'PFBS_INF', 'PFPeS_INF',
    'PFHxS_INF', 'PFHpS_INF', 'PFOS_INF', 'PFNS_INF', 'PFDS_INF', 'PFDoS_INF',
    'FOSA_INF', 'MeFOSA_INF', 'EtFOSA_INF', 'MeFOSE_INF', 'EtFOSE_INF',
    'NMeFOSAA_INF', 'NEtFOSAA_INF', 'ADONA_INF', 'HFPO_DA_INF', 'ClPF3OUDS_11_INF', 'ClPF3ONS_9_INF',
    'PFAS_total_EFF',	'PFAS_total_BIO']

# df_eff = df.copy()
df_eff = df.copy()[basic_features]
# df_eff = df_eff.drop(columns=list_bio)     # drop the pfas bio cols
# df_eff = df_eff.drop(columns=list_eff) # drop the pfas eff cols

# label data PFAS_total_EFF-> Eff_label
df_eff.loc[(df_eff.PFAS_total_EFF < 70), 'Eff_label'] = 0
df_eff.loc[(df_eff.PFAS_total_EFF >= 70 ),'Eff_label'] = 1
# df_eff.loc[((df_eff.PFAS_total_EFF >= 70 ) & (df_eff.PFAS_total_EFF <= 120 )),'Eff_label'] = 1
# df_eff.loc[(df_eff.PFAS_total_EFF > 120), 'Eff_label'] = 2
df_eff = df_eff.drop(columns=['PFAS_total_EFF', 'PFAS_total_BIO'])  # 'PFAS_total_EFF',
print(df_eff['Eff_label'].value_counts())   # get the category & their numbers in the label col

df_eff = df_eff.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# split into X and y matrices
num_col = df_eff.shape[1]   # 1 - return number of cols
print('num_col', num_col)
X = df_eff.iloc[:, :-1]     # delete .values for rf importance
y = df_eff.iloc[:, -1]


# Assuming X and y are already defined (y should be categorical for classification)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the classifiers and their respective parameter grids
classifiers = {
    'CatBoost': {
        'model': CatBoostClassifier(random_state=42, verbose=0),
        'params': {
            'iterations': [100, 500, 1000],
            'learning_rate': [0.01, 0.1, 0.3],
            'depth': [4, 6, 8]
        }
    },
}

results = []

for classifier_name, classifier_info in classifiers.items():
    clf = GridSearchCV(classifier_info['model'], classifier_info['params'], cv=5, n_jobs=-1, scoring='f1_macro')

    if classifier_name in ['SVC', 'LogisticRegression']:
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_train_pred = clf.predict(X_train_scaled)
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_train_pred = clf.predict(X_train)

    # Cross-validation scores for various metrics
    cv_accuracy = cross_val_score(clf.best_estimator_, X_train, y_train, cv=5, scoring='accuracy')
    cv_f1 = cross_val_score(clf.best_estimator_, X_train, y_train, cv=5, scoring='f1_macro')
    cv_precision = cross_val_score(clf.best_estimator_, X_train, y_train, cv=5, scoring='precision_macro')
    cv_recall = cross_val_score(clf.best_estimator_, X_train, y_train, cv=5, scoring='recall_macro')

    # Calculate means and standard errors
    accuracy_mean = np.mean(cv_accuracy)
    accuracy_se = np.std(cv_accuracy) / np.sqrt(len(cv_accuracy))

    f1_mean = np.mean(cv_f1)
    f1_se = np.std(cv_f1) / np.sqrt(len(cv_f1))

    precision_mean = np.mean(cv_precision)
    precision_se = np.std(cv_precision) / np.sqrt(len(cv_precision))

    recall_mean = np.mean(cv_recall)
    recall_se = np.std(cv_recall) / np.sqrt(len(cv_recall))

    print(f"Best parameters for {classifier_name}: {clf.best_params_}")

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]) if hasattr(clf.best_estimator_,
                                                                                'predict_proba') else "N/A"

    print(f"Scores for {classifier_name}:")
    print(f"Accuracy: {accuracy} ± {accuracy_se}")
    print(f"F1 Score: {f1} ± {f1_se}")
    print(f"Precision: {precision} ± {precision_se}")
    print(f"Recall: {recall} ± {recall_se}")
    print(f"ROC AUC: {roc_auc}")
    print("\n")

    # Save the trained model
    with open(f'{classifier_name}_model_eff_web.pkl', 'wb') as f:
        pickle.dump(clf, f)

    # Append results to the results list
    results.append({
        'Model': classifier_name,
        'Accuracy': accuracy,
        'Accuracy_SE': accuracy_se,
        'F1 Score': f1,
        'F1_SE': f1_se,
        'Precision': precision,
        'Precision_SE': precision_se,
        'Recall': recall,
        'Recall_SE': recall_se,
        'ROC AUC': roc_auc,
    })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Sort the DataFrame by F1 Score (you can change this to sort by a different metric if you prefer)
results_df = results_df.sort_values('F1 Score', ascending=False)

# Display the results table
print("Results Table:")
print(results_df.to_string(index=False))



