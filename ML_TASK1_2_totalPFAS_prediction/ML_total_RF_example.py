# binary classification  --  RandomForestClassifier
# SMOTE - 合成少数类过采样技术

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

from numpy import mean
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import accuracy_score, r2_score  # accuracy
from sklearn.metrics import mean_squared_error    # Mean Squared Error (MSE)
from sklearn.metrics import mean_absolute_error
import sklearn.metrics as metrics                 # f-i score
from sklearn.metrics import accuracy_score
import random
import math

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

import imblearn       # 使用不平衡学习 Python 库提供的实现
print(imblearn.__version__)

# Generate and plot a synthetic imbalanced classification dataset
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN   # 自适应合成采样 (ADASYN)
from sklearn.preprocessing import StandardScaler
# 调参
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# 二分类数据的ROC曲线可视化
import matplotlib
matplotlib.rcParams['axes.unicode_minus']=False
import seaborn as sns
sns.set(font= "Kaiti",style="ticks",font_scale=1.4)
import pandas as pd
pd.set_option("max_colwidth", 200)
from sklearn.metrics import *



def ML_evaluation_1(model, y_test, y_pred):
    print(model)
    from sklearn.metrics import confusion_matrix, classification_report
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: ', accuracy)
    conf_result = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix')
    print(conf_result)
    class_report = classification_report(y_test, y_pred)
    print('Classification Report')
    print(class_report)

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    # add the parameters needed to change
    parser.add_argument("--which_data", type=str, default='WWTP_EFF_RF_opt',help='')
    # WWTP_EFF_RF_opt, WWTP_BIO_RF_opt
    return parser

def main( ):

    # parameters
    parser = config_parser()
    args = parser.parse_args()

    file_name = args.which_data
    print('file_name',file_name)

    # PFAS list
    pfas_list_1_1 = ["PFBA", "PFPeA", "PFHxA", "PFHpA", "PFOA", "PFNA", 'PFDA', "PFUnA", "PFDoA", "PFTrDA", "PFTA","PFHxDA", "PFODA"]  # "PFODA"  PFBA=PFBTA (13)
    pfas_list_2_1 = ["FTCA_33", "FTCA_53", "FTCA_73", "FTS_42", "FTS_62", "FTS_82","FTS_102"]  # "10:2FTS", "7:3FTCA"  "3:3FTCA", "5:3FTCA",  (7)
    pfas_list_3_1 = ["PFBS", "PFPeS", "PFHxS", "PFHpS", "PFOS", "PFNS", "PFDS", "PFDoS"]  # PFDOS_A (7)
    pfas_list_4_1 = ["FOSA", "MeFOSA", "EtFOSA", "MeFOSE", "EtFOSE", "NMeFOSAA", "NEtFOSAA"]  # (7)
    pfas_list_5_1 = ["ADONA", "HFPO_DA", "ClPF3OUDS_11", "ClPF3ONS_9"]  # "PFBTA"  # (4)
    pfas_list_all_1 = pfas_list_1_1 + pfas_list_2_1 + pfas_list_3_1 + pfas_list_4_1 + pfas_list_5_1  # total pfas: 39

    list = pfas_list_all_1
    list_inf = [i + '_INF' for i in list]
    list_eff = [i + '_EFF' for i in list]
    list_bio = [i + '_BIO' for i in list]
    list_pfas = list_inf + list_eff + list_bio


    ### final rf optimization -- eff
    if file_name == 'WWTP_EFF_RF_opt':
        df = pd.read_csv("WWTP_0ML.csv")
        df = df.drop(columns=list_bio)     # drop the pfas bio col
        df = df.drop(columns=list_eff)
        # label data PFAS_total_EFF-> Eff_label
        df.loc[(df.PFAS_total_EFF < 70), 'Eff_label'] = 0
        df.loc[((df.PFAS_total_EFF >= 70 ) & (df.PFAS_total_EFF <= 120 )),'Eff_label'] = 1
        df.loc[(df.PFAS_total_EFF > 120), 'Eff_label'] = 2
        df = df.drop(columns=['PFAS_total_INF', 'PFAS_total_EFF', 'PFAS_total_BIO','Date'])  # 'PFAS_total_EFF',
        print(df['Eff_label'].value_counts())   # get the category & their numbers in the label col
        # split Training And Test Data
        num_col = df.shape[1]   # 1 - return number of cols
        print('num_col', num_col)
        X = df.iloc[:, :-1]     # delete .values for rf importance
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # # standard the variables -- comment out for te best result
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.fit_transform(X_test)

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2'],
        }

        # Create the random forest classifier
        rf_classifier = RandomForestClassifier(random_state=42)

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_
        print(f'Best hyperparameters: {best_params}')

        # Train the random forest classifier with the best hyperparameters
        best_rf_classifier = RandomForestClassifier(**best_params, random_state=42)
        best_rf_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = best_rf_classifier.predict(X_test)

        # Evaluate the classifier performance
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.4f}')

        ML_evaluation_1('RF result',y_test, y_pred)



    ### final rf optimization -- bio
    elif file_name == 'WWTP_BIO_RF_opt':
        df = pd.read_csv("WWTP_0ML.csv")
        df = df.drop(columns=list_bio)     # drop the pfas bio col
        # label data PFAS_total_EFF-> Eff_label
        df.loc[(df.PFAS_total_BIO <= 0), 'BIO_label'] = 0
        df.loc[((df.PFAS_total_BIO > 0 ) & (df.PFAS_total_BIO <= 110000 )),'BIO_label'] = 1
        df.loc[(df.PFAS_total_BIO > 110000), 'BIO_label'] = 2
        df = df.drop(columns=['PFAS_total_INF', 'PFAS_total_EFF', 'PFAS_total_BIO','Date'])  # 'PFAS_total_EFF',
        print(df['BIO_label'].value_counts())    # get the category & their numbers in the label col
        # split Training And Test Data
        num_col = df.shape[1]   # 1 - return number of cols
        print('num_col', num_col)
        X = df.iloc[:, :-1]      # delete .values for rf importance
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # standard the variables
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2'],
        }

        # Create the random forest classifier
        rf_classifier = RandomForestClassifier(random_state=42)

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_
        print(f'Best hyperparameters: {best_params}')

        # Train the random forest classifier with the best hyperparameters
        best_rf_classifier = RandomForestClassifier(**best_params, random_state=42)
        best_rf_classifier.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = best_rf_classifier.predict(X_test)

        # Evaluate the classifier performance
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.4f}')

        ML_evaluation_1('RF result',y_test, y_pred)



    ### Dimensionality reduction - PCA - acc did not increase
    elif file_name == 'WWTP_data_EFF_rf_PCA':
        df = pd.read_csv("WWTP_0ML.csv")
        df = df.drop(columns=list_bio)     # drop the pfas bio col
        df = df.drop(columns=list_eff)
        # label data PFAS_total_EFF-> Eff_label
        df.loc[(df.PFAS_total_EFF < 70), 'Eff_label'] = 0
        df.loc[((df.PFAS_total_EFF >= 70 ) & (df.PFAS_total_EFF <= 120 )),'Eff_label'] = 1
        df.loc[(df.PFAS_total_EFF > 120), 'Eff_label'] = 2
        df = df.drop(columns=['PFAS_total_INF', 'PFAS_total_EFF', 'PFAS_total_BIO','Date'])  # 'PFAS_total_EFF',
        print(df['Eff_label'].value_counts())   # get the category & their numbers in the label col
        # split Training And Test Data
        num_col = df.shape[1]   # 1 - return number of cols
        print('num_col', num_col)
        X = df.iloc[:, :-1]     # delete .values for rf importance
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # # standard the variables
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.fit_transform(X_test)

        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=0.95)  # Retain 95% of the variance in the dataset
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Train the random forest classifier
        rf_classifier = RandomForestClassifier(random_state=1)   # n_estimators=100,
        rf_classifier.fit(X_train_pca, y_train)

        # Make predictions on the test set
        y_pred = rf_classifier.predict(X_test_pca)

        ML_evaluation_1('RF', y_test, y_pred)


    ### feature selection - did not work well - acc did not increase
    elif file_name == 'WWTP_data_EFF_featureS_rf_regression':
        df = pd.read_csv("WWTP_0ML.csv")
        df = df.drop(columns=list_bio)     # drop the pfas bio col
        df = df.drop(columns=list_eff)
        # label data PFAS_total_EFF-> Eff_label
        df.loc[(df.PFAS_total_EFF < 70), 'Eff_label'] = 0
        df.loc[((df.PFAS_total_EFF >= 70 ) & (df.PFAS_total_EFF <= 120 )),'Eff_label'] = 1
        df.loc[(df.PFAS_total_EFF > 120), 'Eff_label'] = 2
        df = df.drop(columns=['PFAS_total_INF', 'PFAS_total_EFF', 'PFAS_total_BIO','Date'])  # 'PFAS_total_EFF',
        print(df['Eff_label'].value_counts())   # get the category & their numbers in the label col
        # split Training And Test Data
        num_col = df.shape[1]   # 1 - return number of cols
        print('num_col', num_col)
        X = df.iloc[:, :-1]     # delete .values for rf importance
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # standard the variables
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        ### RF
        # RF - Train
        clf = RandomForestClassifier(random_state=1)  # n_estimators=200, max_depth=52, max_features=35,
        clf.fit(X_train, y_train)   # .ravel()
        clf.score(X_test, y_test)
        # Evaluate model
        y_pred = clf.predict(X_test)
        ML_evaluation_1('RF',y_test, y_pred)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        print(model.feature_importances_)  # use inbuilt class feature_importances
        # plot graph of feature importances for better visualization
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        # feat_importances.nlargest(30).plot(kind='barh')  # plot the top 30
        feat_importances = feat_importances.to_frame()
        feat_importances_plot = feat_importances[feat_importances[0]>0.005]
        feat_importances_plot.plot.bar(rot=90)
        plt.tight_layout()
        plt.show()

        # RF - Train & test again
        # only used the select features
        features = feat_importances_plot.index.tolist()
        select_col = features + ['Eff_label']
        df = df[select_col]
        X = df.iloc[:, :-1]     # delete .values for rf importance
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # standard the variables
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        # use the same model to train and test
        clf = RandomForestClassifier(random_state=1)  # n_estimators=200, max_depth=52, max_features=35,
        clf.fit(X_train, y_train)   # .ravel()
        clf.score(X_test, y_test)
        # Evaluate model
        y_pred = clf.predict(X_test)
        ML_evaluation_1('RF',y_test, y_pred)


        # https://zhuanlan.zhihu.com/p/141010878

    elif file_name == 'WWTP_data_EFF_featureS_rf_Classification':
        df = pd.read_csv("WWTP_0ML.csv")
        df = df.drop(columns=list_bio)     # drop the pfas bio col
        df = df.drop(columns=list_eff)
        # label data PFAS_total_EFF-> Eff_label
        df.loc[(df.PFAS_total_EFF < 70), 'Eff_label'] = 0
        df.loc[((df.PFAS_total_EFF >= 70 ) & (df.PFAS_total_EFF <= 120 )),'Eff_label'] = 1
        df.loc[(df.PFAS_total_EFF > 120), 'Eff_label'] = 2
        df = df.drop(columns=['PFAS_total_INF', 'PFAS_total_EFF', 'PFAS_total_BIO','Date'])  # 'PFAS_total_EFF',
        print(df['Eff_label'].value_counts())   # get the category & their numbers in the label col
        # split Training And Test Data
        num_col = df.shape[1]   # 1 - return number of cols
        print('num_col', num_col)
        X = df.iloc[:, :-1]     # delete .values for rf importance
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # standard the variables
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        ### RF
        # RF - Train
        clf = RandomForestClassifier(random_state=1)  # n_estimators=200, max_depth=52, max_features=35,
        clf.fit(X_train, y_train)   # .ravel()
        clf.score(X_test, y_test)
        # Evaluate model
        y_pred = clf.predict(X_test)
        ML_evaluation_1('RF',y_test, y_pred)

        # plot graph of feature importances for better visualization
        feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
        # feat_importances.nlargest(30).plot(kind='barh')  # plot the top 30
        feat_importances = feat_importances.to_frame()
        weight_var = 0.0   ### modify
        print('weight larger than: ', weight_var)
        feat_importances_plot = feat_importances[feat_importances[0]>weight_var]   # >0.005 - 46 variables
        feat_importances_plot.plot.bar(rot=90)
        plt.tight_layout()
        plt.show()

        # RF - Train & test again
        # only used the select features
        features = feat_importances_plot.index.tolist()
        print('variable numbers: ', len(features))
        select_col = features + ['Eff_label']
        df = df[select_col]
        X = df.iloc[:, :-1]     # delete .values for rf importance
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # standard the variables
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        # use the same model to train and test
        clf = RandomForestClassifier(random_state=1)  # n_estimators=200, max_depth=52, max_features=35,
        clf.fit(X_train, y_train)   # .ravel()
        clf.score(X_test, y_test)
        # Evaluate model
        y_pred = clf.predict(X_test)
        ML_evaluation_1('RF',y_test, y_pred)

    elif file_name == 'WWTP_data_EFF_featureSelection_DT':
        df = pd.read_csv("WWTP_0ML.csv")
        df = df.drop(columns=list_bio)     # drop the pfas bio col
        df = df.drop(columns=list_eff)
        # label data PFAS_total_EFF-> Eff_label
        df.loc[(df.PFAS_total_EFF < 70), 'Eff_label'] = 0
        df.loc[((df.PFAS_total_EFF >= 70 ) & (df.PFAS_total_EFF <= 120 )),'Eff_label'] = 1
        df.loc[(df.PFAS_total_EFF > 120), 'Eff_label'] = 2
        df = df.drop(columns=['PFAS_total_INF', 'PFAS_total_EFF', 'PFAS_total_BIO','Date'])  # 'PFAS_total_EFF',
        print(df['Eff_label'].value_counts())   # get the category & their numbers in the label col
        # split Training And Test Data
        num_col = df.shape[1]   # 1 - return number of cols
        print('num_col', num_col)
        X = df.iloc[:, :-1]     # delete .values for rf importance
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # standard the variables
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        ### RF
        # RF - Train
        clf = RandomForestClassifier(random_state=1)  # n_estimators=200, max_depth=52, max_features=35,
        clf.fit(X_train, y_train)   # .ravel()
        clf.score(X_test, y_test)
        # Evaluate model
        y_pred = clf.predict(X_test)
        ML_evaluation_1('RF',y_test, y_pred)

        from sklearn import feature_selection
        # 筛选前20%的特征,使用相同配置的决策树模型进行预测,并且评估性能
        fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=20)
        X_train_fs = fs.fit_transform(X_train, y_train)
        clf.fit(X_train_fs, y_train)
        X_test_fs = fs.transform(X_test)
        clf.score(X_test_fs, y_test)
        # Evaluate model
        y_pred = clf.predict(X_test)
        ML_evaluation_1('RF2',y_test, y_pred)

        # https://zhuanlan.zhihu.com/p/34940911


    ### USGS RF examples
    elif file_name == 'USGS_data1':
        print('pfas dataset')
        # Load Data
        df = pd.read_csv("GW_MAIN_PFAS_14_subset_pfas_total.csv")
        # Training And Test Data -- drop location cols
        df = df.drop(columns=['gm_gis_dwr_region'])     # 'latitude','longitude',
        X = df.iloc[:, 0:113].values      # delete .values for rf importance
        #y = df.iloc[:, 113:].values        # .values for XBBoost, catb, lightb
        y = np.array(df["Label_PFAS_total"])


        # 1 SMOTE Oversampling for Binary classification
        # summarize class distribution
        counter = Counter(y)   # Counter({0: 19121, 1: 7779})
        print(counter)
        # scatter plot of examples by class label
        for label, _ in counter.items():
            row_ix = where(y == label)[0]
            pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
        pyplot.legend()
        pyplot.show()

        # SMOTE 对少数类进行过采样并绘制转换后的数据集
        print('SMOTE')
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)    # Counter({0: 19121, 1: 19121})
        # split to train, test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        print('Random forest')
        clf = RandomForestClassifier(n_estimators=200, random_state=1)  # max_depth=15
        clf.fit(X_train, y_train)   # .ravel()
        y_pred = clf.predict(X_test)
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print('accuracy: ', accuracy)        # 0.9741142633023925

        # 自适应合成采样 (ADASYN) -- 生成与少数类中样本的密度成反比的合成样本
        print('ADASYN')
        X = df.iloc[:, 0:113].values      # delete .values for rf importance
        y = np.array(df["Label_PFAS_total"])
        oversample = ADASYN()
        X, y = oversample.fit_resample(X, y)    # Counter({0: 19121, 1: 19121})
        # split to train, test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        print('Random forest')
        clf = RandomForestClassifier(n_estimators=200, random_state=1)  # max_depth=15
        clf.fit(X_train, y_train)   # .ravel()
        y_pred = clf.predict(X_test)
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print('accuracy: ', accuracy)

        # 首先使用 SMOTE 对少数类进行过采样到大约 1:10 (sampling_strategy=0.1) 的比例，然后对多数类进行欠采样以达到大约 1:2 (sampling_strategy=0.5)的比例
        X = df.iloc[:, 0:113].values      # delete .values for rf importance
        y = np.array(df["Label_PFAS_total"])             # Counter({0: 19120, 1: 7779})
        under = RandomUnderSampler(sampling_strategy=0.5)
        X, y = under.fit_resample(X, y)
        counter = Counter(y)    # Counter({0: 15558, 1: 7779})
        print(counter)
        oversample = SMOTE(sampling_strategy=0.95)
        X, y = oversample.fit_resample(X, y)
        counter = Counter(y)    # Counter({0: 15558, 1: 14780})
        print(counter)
        # split to train, test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Train The Random Forest
        print('Random forest')
        clf = RandomForestClassifier(n_estimators=200, random_state=1)  # max_depth=15
        clf.fit(X_train, y_train)   # .ravel()
        # Apply RandomForestRegressor To Test Data
        y_pred = clf.predict(X_test)

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print('accuracy: ', accuracy)
        # train_score = clf.score(X_train, y_train)
        # test_score = clf.score(X_test, y_test)
        # r2 = r2_score(y_test, y_pred)
        # mae = mean_absolute_error(y_test, y_pred)
        # mse = mean_squared_error(y_test, y_pred)
        # rmae = np.sqrt(mean_squared_error(y_test, y_pred))
        # print('TEST Score： ', test_score)
        # print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
        # print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
        # print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

    elif file_name == 'USGS_data2':
        print('pfas ml training')
        # Load Data
        df = pd.read_csv("GW_MAIN_PFAS_14_subset_pfas_total.csv")
        # Training And Test Data -- drop location cols
        df = df.drop(columns=['gm_gis_dwr_region'])     # 'latitude','longitude',
        X = df.iloc[:, 0:113].values      # delete .values for rf importance
        #y = df.iloc[:, 113:].values        # .values for XBBoost, catb, lightb
        y = np.array(df["Label_PFAS_total"])

        # 自适应合成采样 (ADASYN) -- 生成与少数类中样本的密度成反比的合成样本
        print('ADASYN')
        oversample = ADASYN()
        X, y = oversample.fit_resample(X, y)    # Counter({0: 19121, 1: 19121})
        # split to train, test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Train The Random Forest
        # RF -- ML optimization
        print('Random forest')

        # search for the best n_estimators
        '''
        # n_estimators是影响程度最大的参数，我们先对其进行调整 # 调参，绘制学习曲线来调参n_estimators（对随机森林影响最大）
        score_lt = []
        # 每隔10步建立一个随机森林，获得不同n_estimators的得分
        for i in range(0, 200, 10):
            rfc = RandomForestClassifier(n_estimators=i + 1, random_state=90)
            score = cross_val_score(rfc, X_train, y_train, cv=10).mean()
            score_lt.append(score)
            print(i)
        score_max = max(score_lt)
        print('score_max：{}'.format(score_max),
              'n_estimators：{}'.format(score_lt.index(score_max) * 10 + 1))   # n_estimators 子树数量

        # 绘制学习曲线  --  find the best n_estimators
        x = np.arange(1, 241, 10)
        plt.subplot(111)
        plt.plot(x, score_lt, 'r-')
        plt.show()
        '''
        # 接下来的调参方向是使模型复杂度减小的方向，从而接近泛化误差最低点。我们使用能使模型复杂度减小，并且影响程度排第二的max_depth。
        '''
        # 建立n_estimators为200的随机森林
        rfc = RandomForestClassifier(n_estimators=200, random_state=90)

        # 用网格搜索调整max_depth
        param_grid = {'max_depth': np.arange(1, 20)}
        GS = GridSearchCV(rfc, param_grid, cv=10)
        GS.fit(X_train, y_train)

        best_param = GS.best_params_
        best_score = GS.best_score_
        print(best_param, best_score)

        # 用网格搜索调整max_features
        param_grid = {'max_features': np.arange(20, 51)}

        rfc = RandomForestClassifier(n_estimators=200
                                     , random_state=90
                                     , max_depth=52)
        GS = GridSearchCV(rfc, param_grid, cv=10)
        GS.fit(X_train, y_train)
        best_param = GS.best_params_
        best_score = GS.best_score_
        print(best_param, best_score)
        '''

        # RF - after optimization
        clf = RandomForestClassifier(n_estimators=200,max_depth=52,max_features=35,random_state=1)  #  n_estimators=200  max_depth=15
        clf.fit(X_train, y_train)   # .ravel()
        # Apply RandomForestRegressor To Test Data
        y_pred = clf.predict(X_test)
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print('accuracy: ', accuracy)
        # train_score = clf.score(X_train, y_train)
        # test_score = clf.score(X_test, y_test)
        # r2 = r2_score(y_test, y_pred)
        # mae = mean_absolute_error(y_test, y_pred)
        # mse = mean_squared_error(y_test, y_pred)
        # rmae = np.sqrt(mean_squared_error(y_test, y_pred))
        # print('TEST Score： ', test_score)
        # print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
        # print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
        # print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        # 结果可视化
        # AUROC Curve  -- https://zhuanlan.zhihu.com/p/364400255

        ## 可视化在验证集上的Roc曲线
        pre_y = clf.predict_proba(X_test)[:, 1]
        fpr_Nb, tpr_Nb, _ = roc_curve(y_test, pre_y)
        aucval = auc(fpr_Nb, tpr_Nb)  # 计算auc的取值
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_Nb, tpr_Nb, "r", linewidth=3)
        plt.grid()
        plt.xlabel("False Postive Rate")
        plt.ylabel("True Positive Rate")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title("RF ROC curve")
        plt.text(0.15, 0.9, "AUC = " + str(round(aucval, 4)))
        plt.show()


        '''
        # 将训练集结果可视化
        from matplotlib.colors import ListedColormap
        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
        plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c=ListedColormap(('red', 'green'))(i), label=j)
        plt.title('Random Forest Classification (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()
        # 将测试集结果可视化
        from matplotlib.colors import ListedColormap
        X_set, y_set = X_test, y_test
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
        plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c=ListedColormap(('red', 'green'))(i), label=j)
        plt.title('Random Forest Classification (Test set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()
        '''

        '''
        # importance
        col = list(X_train.columns.values)
        importances = clf.feature_importances_
        col_list = df.columns.values.tolist()
        x_columns = col_list[0:-1]
        # print("importances：", importances)
        # Returns the index value of the array from largest to smallest
        indices = np.argsort(importances)[::-1]
        list01 = []
        list02 = []
        for f in range(X_train.shape[1]):
            # For the final need to be sorted in reverse order, I think it is to do a value similar to decision tree backtracking,
            # from the leaf to the root, the root is more important than the leaf.
            print("%2d) %-*s %f" % (f + 1, 30, col[indices[f]], importances[indices[f]]))
            list01.append(col[indices[f]])
            list02.append(importances[indices[f]])

        c = {"columns": list01, "importances": list02}
        data_impts = DataFrame(c)
        # data_impts.to_excel('RF_data_importances.xlsx')

        importances = list(clf.feature_importances_)
        feature_list = list(X_train.columns)

        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(feature_list, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

        x_values = list(range(len(importances)))
        print(x_values)
        '''

    else:
        print('pass')

if __name__ == "__main__":
    main( )


print('\n--------------------------------------------------------------\n')



# https://zhuanlan.zhihu.com/p/440648816
# RF 调参 -- https://zhuanlan.zhihu.com/p/126288078


