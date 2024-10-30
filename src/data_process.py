import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn.preprocessing import OrdinalEncoder

# Read in validated data
df = pd.read_csv("WWTP-PFAS-CA.csv")

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

# drop rows Global_ID is NaN
df = df.dropna(subset=['Global_ID'])

# drop columns after the column named 'Government'
index = df.columns.get_loc('Government')  # Find the index of the 'Government' column
df = df.iloc[:, :index + 1]  # # Drop all columns after 'Government'

## Drop Latitude, Longitude, City, County Info
for feature in ['Latitude', 'Longitude', 'City', 'County']:
    del df[feature]

# encoder COL LIST
col_cate_list = ['TREATMENT LEVEL', 'Treatment Level Simple', 'Population BIN group']

# replace
df['Population BIN group'] = df['Population BIN group'].replace('Under 1,000', '1000-')
ord_enc = OrdinalEncoder()

# for i in col_cate_list:
#     df[[i]] = ord_enc.fit_transform(df[[i]])
df['TREATMENT_code'] = ord_enc.fit_transform(df[['TREATMENT LEVEL']])
# df['City_code'] = ord_enc.fit_transform(df[['City']])
df['TreatmentLevel_code'] = ord_enc.fit_transform(df[['Treatment Level Simple']])
df['Population_code'] = ord_enc.fit_transform(df[['Population BIN group']])
# df['County_code'] = ord_enc.fit_transform(df[['County']])

# fill NAs
values = {'TreatmentLevel_code': 0, 'Population_code': 6}
df = df.fillna(value=values)

# check the number & categories
df_treat = df[['TREATMENT_code', 'TREATMENT LEVEL']]  # select cols
df_treat = df_treat.drop_duplicates()
print('TREATMENT')
print(df_treat)

# df_city = df[['City_code', 'City']]  # select cols
# df_city = df_city.drop_duplicates()
# print('city')
# print(df_city)

df_t2 = df[['TreatmentLevel_code', 'Treatment Level Simple']]  # select cols
df_t2 = df_t2.drop_duplicates()
print('TreatmentLevel')

print(df_t2)
df_pop = df[['Population_code', 'Population BIN group']]  # select cols
df_pop = df_pop.drop_duplicates()
print('Population')
print(df_pop)

# df_county = df[['County_code', 'County']]  # select cols
# df_county = df_county.drop_duplicates()
# print('County')
# print(df_county)

# # re-order cols
# list_col = df.columns.values.tolist()
# other_list = col_cate_list
# refill_na_col = [i for i in list_col if i not in other_list]
# order_col = other_list + refill_na_col

# drop cols
df = df.drop(columns=col_cate_list)

# check df col types
print('column_types')
column_types = df.dtypes
print(column_types)

# check na for cols
na_num = df.isna().sum()
print(na_num)

df = df.drop(columns=['Global_ID'])  # DROP cols

# change date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# # drop cols with to many NAs
# drop_list = []
# value_limit = 100
# for col in df.columns:
#     na_num = df[col].isna().sum()
#     na_perc = (na_num / len(df))
#     value_num = (len(df) - na_num)
#     if value_num < value_limit:
#         drop_list.append(col)
#     # if na_perc > 0.8:
#     # drop_list.append(col)

# final_drop_list = [i for i in drop_list if i not in list_pfas]
# df = df.drop(columns=final_drop_list)

# check na for cols
na_num = df.isna().sum()
# print(na_num)

# drop cols
# df = df.drop(columns=['5b_1_Sewage_Sludge_Agricultural_Est_Vol_Percentage_2019', '5b_1_Biosolids_Agricultural_Est_Vol_Percentage_2019',
#                       '5b_2_Sewage_Sludge_Composting_Est_Vol_Percentage_2019','5b_2_Biosolids_Composting_Est_Vol_Percentage_2019',
#                       '5b_3_Sewage_Sludge_Forest_Est_Vol_Percentage_2019','5b_3_Biosolids_Forest_Est_Vol_Percentage_2019',
#                       '5b_4_Sewage_Sludge_Incineration_Est_Vol_Percentage_2019','5b_4_Biosolids_Incineration_Est_Vol_Percentage_2019'])  # DROP cols
# df = df.drop(columns=['5b_7_Sewage_Sludge_Mine_Rec_Est_Vol_Percentage_2019','5b_7_Biosolids_Mine_Rec_Est_Vol_Percentage_2019',
#                       '5b_8_Sewage_Sludge_Public_Dist_Est_Vol_Percentage_2019','5b_8_Biosolids_Public_Dist_Est_Vol_Percentage_2019',
#                       '5b_9_Sewage_Sludge_Public_Lands_Est_Vol_Percentage_2019','5b_9_Biosolids_Public_Lands_Est_Vol_Percentage_2019',
#                       '5b_10_Sewage_Sludge_Onsite_Dedicated_Land_Disp_Est_Vol_Percentage_2019',
#                       '5b_10_Biosolids_Onsite_Dedicated_Land_Disp_Est_Vol_Percentage_2019',
#                       '5b_11_Sewage_Sludge_Onsite_Long_Term_Storage_Est_Vol_Percentage_2019',
#                       '5b_11_Biosolids_Onsite_Long_Term_Storage_Est_Vol_Percentage_2019'])
# df = df.drop(columns=['Is loading considered when setting rates? If yes, specify the type. Response (Yes/No)',
#                       'Is loading considered when setting rates? If yes, specify the type. Biochemical Oxygen Demand (BOD)',
#                       'Is loading considered when setting rates? If yes, specify the type. Chemical Oxygen Demand (COD)',
#                       'Is loading considered when setting rates? If yes, specify the type. Suspended Solids (SS)',
#                       'Is loading considered when setting rates? If yes, specify the type. Other (please specify)'])

# replace string with number
df = df.replace('Yes ', 1)
df = df.replace(' ', 0)  # blanks

# fill NAs with mean
# selected_columns = df.loc[:, 'DISCHARGE VOLUME':'INFLUENT VOLUME'].columns.tolist()
# for col in selected_columns:
#     df[col] = pd.to_numeric(df[col], errors='coerce')

for column in df.columns[df.columns.get_loc('DISCHARGE VOLUME'):df.columns.get_loc('INFLUENT VOLUME') + 1]:
    df[column].fillna(df[column].mean(), inplace=True)

for column in df.columns[
              df.columns.get_loc('1a_Avg_Annual_Flow'):df.columns.get_loc('2a_Industrial_WW_Percentage') + 1]:
    df[column].fillna(df[column].mean(), inplace=True)

for column in df.columns[df.columns.get_loc('5a_Sewage_Sludge_Dry_Metric_Tons'):df.columns.get_loc(
        '5a_ClassB_Dry_Metric_Tons') + 1]:
    df[column].fillna(df[column].mean(), inplace=True)

for column in df.columns[df.columns.get_loc('Flow_INF'):df.columns.get_loc('Service Area Median Household Income') + 1]:
    df[column].fillna(df[column].mean(), inplace=True)

for column in df.columns[
              df.columns.get_loc('Wastewater Budgets Wastewater operation and maintenance budget'):df.columns.get_loc(
                      'Wastewater Budgets Wastewater capital expenditure budget') + 1]:
    df[column].fillna(df[column].mean(), inplace=True)

for column in df.columns[
              df.columns.get_loc('Agency Role/Responsibilities Average Dry Weather Flow (MGD)'):df.columns.get_loc(
                      'Agency Role/Responsibilities Design Flow Capacity (MGD)') + 1]:
    df[column].fillna(df[column].mean(), inplace=True)

# for column in df.columns[df.columns.get_loc('Biochemical Oxygen Demand (BOD) (5-day @ 20 Deg. C)_INF'):df.columns.get_loc('Flow_INF') + 1]:
#     df[column].fillna(df[column].mean(), inplace=True)

selected_columns = df.loc[:, 'City_POP2020':'Government'].columns.tolist()
for col in selected_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
for column in df.columns[df.columns.get_loc('City_POP2020'):df.columns.get_loc('Government') + 1]:
    df[column].fillna(df[column].mean(), inplace=True)

# df = df.fillna(df.mean()['DISCHARGE VOLUME':'INFLUENT VOLUME'])
# df = df.fillna(df.mean()['1a_Avg_Annual_Flow':'2a_Industrial_WW_Percentage'])
# df = df.fillna(df.mean()['5a_Sewage_Sludge_Dry_Metric_Tons':'5a_ClassB_Dry_Metric_Tons'])
# df = df.fillna(df.mean()['Flow_INF':'Service Area Median Household Income'])
# df = df.fillna(df.mean()['Wastewater Budgets Wastewater operation and maintenance budget':'Wastewater Budgets Wastewater capital expenditure budget'])
# df = df.fillna(df.mean()['Agency Role/Responsibilities Average Dry Weather Flow (MGD)':'Agency Role/Responsibilities Design Flow Capacity (MGD)'])
# df = df.fillna(df.mean()['Biochemical Oxygen Demand (BOD) (5-day @ 20 Deg. C)_INF':'Flow_INF'])
# df = df.fillna(df.mean()['City_POP2020':'Government'])

# fill with additional category: 2
values = {'FAC PRODUCE REC WATER': 2, 'Industrial_Total': 16, '2b_16_Est_Industrial_Total_Volume_Othercopy': 8.5}
df = df.fillna(value=values)
# col： 1a_Avg_Annual_Flow - 7b_5yrs_Volume_RO_Concentrate
# fill in: col average, assign value, 0 (for all the cols not mentioned above)
df = df.fillna(0)

na_num = df.isna().sum()  # check nan
# print(na_num)

# calculate the total pfas in inf, eff, bio
index_inf = df.columns.get_loc('PFBA_INF')  # Find the index of the 'PFBA_INF' column
index_eff = df.columns.get_loc('PFBA_EFF')
index_bio = df.columns.get_loc('PFBA_BIO')
index_Q = df.columns.get_loc('1a_Avg_Annual_Flow')
df['PFAS_total_INF'] = df.iloc[:, index_inf:index_eff].sum(axis=1)  # add new col
df['PFAS_total_EFF'] = df.iloc[:, index_eff:index_bio].sum(axis=1)  # add new col
df['PFAS_total_BIO'] = df.iloc[:, index_bio:index_Q].sum(axis=1)  # add new col

# Checking for any NaNs in the entire data
df.isna().sum().sum()

# Save DataFrame to a CSV file
df.to_csv('data_processed.csv', index=False)  # Set index=False to avoid saving row numbers




