# Machine Learning for Monitoring Per- and Polyfluoroalkyl Substance (PFAS) in California's Wastewater Treatment Plants

Although wastewater treatment plants (WWTPs) are considerable sources of Per- and Polyfluoroalkyl Substances (PFAS) pollution, comprehensive monitoring, and management outlining PFAS contamination in WWTPs remain insufficient. To address this issue, we compiled a statewide database (WWTP-PFAS-CA) and developed machine learning (ML) models to predict PFAS risk. To facilitate the establishment of an effective, data-driven monitoring framework, we developed the public WWTP-PFAS-CA statewide database (2020 - 2023), which encompasses information from over 200 WWTPs across California. This database detailed PFAS concentrations in influent, effluent, and biosolids and included data on sampling dates, wastewater sources, and treatment processes. Our analysis revealed that more than 80% of WWTPs exhibit increased total PFAS concentrations in the effluent, with over half of these facilities facing a significant risk of surpassing 70 ng/L threshold for PFAS levels in wastewater. Individual PFAS were positively correlated with each other within the same wastewater matrix. Differences in pollution sources can lead to varying PFAS species and concentrations, while transformation processes within WWTPs may result in higher PFAS concentrations in the effluent than in the influent, due to the presence of PFAS precursors. Additionally, we developed a data-driven ML tool to strengthen comprehensive PFAS monitoring (assessing total PFAS risk, individual PFAS occurrences, and predicting specific PFAS concentrations) in WWTPs. Our machine learning models achieved ~80% accuracy in predicting total PFAS risk in WWTPs and identified key influencers of PFAS fate in influent, effluent and biosolids, including WWTP size, wastewater source, county population, and GDP. Our research provides a data-driven perspective on PFAS behavior and offers a novel strategy for strengthening wastewater monitoring (pre-screening or prioritizing sampling). 

This repository provides a machine-learning framework for monitoring and classifying PFAS contamination risk in influent, effluent, and biosolids at California's wastewater treatment plants (WWTP).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JialinDong/ML-Monitoring-PFAS-Californias-WWTP.git

2. Navigate to the project directory:
   ```bash
   cd ML-Monitoring-PFAS-Californias-WWTP

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt

## Data

The original raw dataset is WWTP-PFAS-CA.csv, and the data processed for developing ML models is data_processed.csv, which includes PFAS measurements and other influent characteristics.

## Model development

We categorized PFAS levels in WWTP influent and effluent above 70 ng/L (a previous U.S. Environmental Protection Agency (EPA) health advisory level for PFOA and PFOS in drinking water) as high risk (1) and below as low risk (0), while biosolids were marked as high risk at any detectable level due to frequent non-detects. We developed distinct models to assess total PFAS risk in WWTPs based on different inputs. We included general models were limited to commonly monitored standard operational parameters of WWTPs (year, month, influent/effluent volumes, industrial wastewater intake, total organic carbon (TOC), ammonia, BOD, carbonaceous BOD (CBOD), flow rate, pH, total dissolved solids (TDS), and total suspended solids (TSS)) or only 39 PFASs’ concentrations in WWTP influent as inputs. This multiple inputs selection strategy was employed to better assess the risk of PFAS by accommodating scenarios with both comprehensive and limited data. 

Model Training and Optimization:

For each model:
Algorithm: We employed the CatBoost algorithm, which, in combination with SMOTE oversampling, consistently outperformed other models in predicting high PFAS concentrations in influent, effluent, and biosolids.
Cross-Validation: Each model underwent 5-fold cross-validation for robust performance estimation.
Hyperparameter Tuning: GridSearchCV was used to find the best hyperparameters for each model.

Model Evaluation:
Key metrics (accuracy, F1 score, precision, recall, and ROC AUC) are calculated and displayed with standard errors.
Cross-validation scores are computed to ensure robustness across different data splits.

Model Performance 

Using Operational Parameters:
Influent: CatBoost achieved 74.2% accuracy in predicting high PFAS concentrations.
Effluent: CatBoost achieved 73.7% accuracy.
Biosolids: CatBoost reached 78.0% accuracy.
Using PFAS Concentrations in Influent:
Effluent: CatBoost achieved the highest performance with an accuracy of 81.0%.
Biosolids: AdaBoost performed best with 66.7% accuracy.

Saving the Model:
The best model configuration for each classifier is saved as a .pkl file in the Models/ directory.
