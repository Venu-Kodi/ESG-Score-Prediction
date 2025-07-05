#!/usr/bin/env python
# coding: utf-8
ESG Score Prediction model building by Venu Kodi 
# In[3]:


import pandas as pd
import numpy as np

1. Data Loading and Inspection
# In[5]:


df = pd.read_csv("esg_scores_regression.csv")


# In[7]:


df


# 2. Exploratory Data Analysis

# 1). Dataset Overview

# Dataset Head -- To know the Top 5 rows

# In[12]:


df.head(5)


# Dataset Tail -- To know the Bottom 5 rows

# In[15]:


df.tail(5)


# Dataset Shape : To know number of Rows & Columns

# In[18]:


df.shape


# In[20]:


print("Number of Rows:", df.shape[0])


# In[22]:


print("Number of Columns:", df.shape[1])


# Dataset Information & Data types

# In[25]:


df.info()


# Dataset Description -- Statistical summary for both Numerical & Categorical columns

# In[28]:


df.describe(include='all')


# 2). Missing Values & their count percentages

# In[31]:


missing_count = df.isnull().sum()


# In[33]:


print(missing_count)


# In[35]:


percentage_missing = (missing_count/len(df))*100


# In[37]:


print(percentage_missing)


# 3). Target variable distribution plot -- Histogram
# -- Verifying the skewness and outliers

# In[40]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[41]:


target_col = 'ESG_Score'


# In[42]:


plt.figure(figsize=(8,5))
sns.histplot(df[target_col], kde=True, bins=30, color='skyblue')

plt.title(f'Distribution of {target_col}', fontsize=14)
plt.xlabel(target_col)
plt.ylabel('Frequency')
plt.show()


# 4). Numerical Features -- Univariate Analysis

# In[46]:


numerical_col = df.select_dtypes(include=['int64', 'float64'])


# In[49]:


print(numerical_col)


# In[51]:


for col in numerical_col:
    plt.figure(figsize=(8,5))
    sns.histplot(df[col], kde=True, color='skyblue')

    plt.title(f'Distribution of {col}', fontsize=10)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# 5). Categorical Features -- Univariate Analysis

# In[53]:


categorical_col = df.select_dtypes(include=['object', 'category'])


# In[54]:


categorical_col


# In[55]:


for col in categorical_col:
    plt.figure(figsize=(5,3))
    sns.countplot(df[col],color='skyblue',palette='Set2')

    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()


# 6). Duplicate data detection
# -- check duplicates
# -- view duplicates if needed

# In[57]:


#view duplicates, if needed
duplicates = df[df.duplicated()]
print(duplicates)


# In[60]:


#Check number of duplicates
df.duplicated().sum()


# 7). Outlier Detection using Boxplots

# In[63]:


for col in numerical_col:
    plt.figure(figsize=(5,3))
    sns.boxplot(df[col],color='lightgreen')

    plt.title(f'Boxplot of {col}', fontsize=8)
    plt.xlabel(col)
    plt.show()


# 8). Correlation Heatmap
# -- To identify highly correlated variables, causes Multicollinearity problems

# In[65]:


corr_matrix = numerical_col.corr()
print(corr_matrix)


# In[66]:


plt.figure(figsize=(7,4))
sns.heatmap(corr_matrix, annot=True,cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()


# 9). Multicollinearity Check -- Variance Inflation Factor

# In[68]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
X = numerical_col.dropna()
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor (X.values, i) for i in range(X.shape[1])]
print(vif_data)


# 10). Correlation with Target Variable

# In[70]:


correlation_with_target = numerical_col.corr()[target_col].sort_values(ascending=False)
print(correlation_with_target)


# 11). Bivariate Analysis -- Numerical features vs Target

# In[75]:


for col in numerical_col:
    if col != target_col:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[col], y=df[target_col])
        plt.title(f'{col} vs {target_col}')
        plt.show()


# 12). Bivariate Analysis -- Categorical features vs Target

# In[79]:


for col in categorical_col:
    if col != target_col:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[col], y=df[target_col])
        plt.title(f'{target_col} vs {col}')
        plt.xticks(rotation=45)
        plt.show()


# 13). Skewness and Kurtosis check

# In[84]:


for col in numerical_col:
    skewness = df[col].skew()
    kurtosis = df[col].kurtosis()
    print(f"{col} : Skewness = {skewness:.4f}, Kurtosis = {kurtosis : .4f}")


# In[85]:


for col in numerical_col:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f'{col} distribution\nSkewness: {df[col].skew():.2f}, Kurtosis: {df[col].kurtosis(): .2f}')
    plt.show()

3. Data Cleaning and Preprocessing
# 1). Handle duplicate records

# In[88]:


#No duplicates, if exist need to remove using first occurrence, last occurrence by keep=last or drop column wise


# 2). Consistent column names

# In[92]:


df.columns = df.columns.str.strip().str.lower().str.replace(' ','_')


# In[96]:


df.columns


# 3). Handle missing values

# In[99]:


# No missing values found, if present have to deal with percentages_count and replacement with Mean/Median/Mode based on criteria or drop them


# 4). Handle invalid or impossible values

# In[102]:


# No invalid or impossible values found, if present deal with removing or replacement of statistical methods


# 5). Correct data types

# In[105]:


#Replacing obj with category for memory usage purposes
df['sector'] = df['sector'].astype('category')
df['country'] = df['country'].astype('category')


# In[107]:


df.info()


# 6). Outlier treatment

# In[110]:


from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')


# In[112]:


df['market_cap_billions'] = pt.fit_transform(df[['market_cap_billions']])


# In[114]:


plt.figure(figsize=(12,5))
sns.boxplot(df['market_cap_billions'])
plt.title('Market cap after Transformations')
plt.show()


# In[115]:


df['ceo_compensation_ratio'] = pt.fit_transform(df[['ceo_compensation_ratio']])


# In[118]:


plt.figure(figsize=(12,5))
sns.boxplot(df['ceo_compensation_ratio'])
plt.title('ceo_compensation_ratio after Transformations')
plt.show()


# In[120]:


df['roe'] = pt.fit_transform(df[['roe']])


# In[122]:


plt.figure(figsize=(12,5))
sns.boxplot(df['roe'])
plt.title('roe after Transformations')
plt.show()


# In[124]:


df['esg_score'] = pt.fit_transform(df[['esg_score']])


# In[126]:


plt.figure(figsize=(12,5))
sns.boxplot(df['esg_score'])
plt.title('esg_score after Transformations')
plt.show()


# In[128]:


# Even after transformations, outliers are due to compression to the mean and extreme values. can be decoded with capping and transformations if needed


# 7). Encoding categorical variables & Feature Scaling in Pipelines steps

# In[131]:


df.columns


# In[133]:


# Feature engineering not required


# 8). Drop unncesseary columns  

# In[136]:


df = df.drop('company', axis=1)


# In[138]:


df.head(2)


# Train Test Split

# In[141]:


X = df.drop('esg_score', axis=1)
y = df['esg_score']


# In[143]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[145]:


numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()


# In[147]:


categorical_cols= X_train.select_dtypes(include=['object', 'category']).columns.tolist()


# Feature scaling & Encoding in Pipeline method at a time

# In[150]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# In[152]:


preprocessor = ColumnTransformer([('num', StandardScaler(), numerical_cols), 
                                               ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)])


# Multple model selection & their evaluations

# In[155]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


# A. Linear Regression

# In[158]:


from sklearn.linear_model import LinearRegression


# In[160]:


pipe_lr = Pipeline([('preprocessing', preprocessor),
                    ('model', LinearRegression())])
#Fit
pipe_lr.fit(X_train, y_train)


# In[162]:


#Predict
y_pred_lr = pipe_lr.predict(X_test)
#Evaluate
print("Linear Regression:")
print("RMSE:", mean_squared_error(y_test, y_pred_lr, squared=False))
print("R2 Score:", r2_score(y_test, y_pred_lr))


# B. Decision Tree Regressor

# In[165]:


from sklearn.tree import DecisionTreeRegressor


# In[168]:


pipe_dt = Pipeline([('preprocessing', preprocessor),
                    ('model', DecisionTreeRegressor(random_state=42))])
#Fit
pipe_dt.fit(X_train, y_train)


# In[170]:


#Predict
y_pred_dt = pipe_dt.predict(X_test)
#Evaluate
print("Decision Tree Regressor:")
print("RMSE:", mean_squared_error(y_test, y_pred_dt, squared=False))
print("R2 Score:", r2_score(y_test, y_pred_dt))


# C. Random Forest Regressor

# In[173]:


from sklearn.ensemble import RandomForestRegressor


# In[175]:


pipe_rf = Pipeline([('preprocessing', preprocessor),
                    ('model', RandomForestRegressor(random_state=42))])
#Fit
pipe_rf.fit(X_train, y_train)


# In[177]:


#Predict
y_pred_rf = pipe_rf.predict(X_test)
#Evaluate
print("Random Forest Regressor:")
print("RMSE:", mean_squared_error(y_test, y_pred_rf, squared=False))
print("R2 Score:", r2_score(y_test, y_pred_rf))


# D. Gradient Boosting Regressor

# In[180]:


from sklearn.ensemble import GradientBoostingRegressor


# In[182]:


pipe_gb = Pipeline([('preprocessing', preprocessor),
                    ('model', GradientBoostingRegressor(random_state=42))])
#Fit
pipe_gb.fit(X_train, y_train)


# In[184]:


#Predict
y_pred_gb = pipe_gb.predict(X_test)
#Evaluate
print("Gradient Boosting Regressor:")
print("RMSE:", mean_squared_error(y_test, y_pred_gb, squared=False))
print("R2 Score:", r2_score(y_test, y_pred_gb))


# E. XGBoost Regressor

# In[189]:


get_ipython().system('pip install xgboost')


# In[190]:


pip install xgboost


# In[193]:


from xgboost import XGBRegressor


# In[195]:


pipe_xgb = Pipeline([('preprocessing', preprocessor),
                    ('model', XGBRegressor(random_state=42))])
#Fit
pipe_xgb.fit(X_train, y_train)


# In[197]:


#Predict
y_pred_xgb = pipe_xgb.predict(X_test)
#Evaluate
print("XGBoost Regressor:")
print("RMSE:", mean_squared_error(y_test, y_pred_xgb, squared=False))
print("R2 Score:", r2_score(y_test, y_pred_xgb))


# F. LightGBM Regressor

# In[202]:


get_ipython().system('pip install lightgbm')


# In[204]:


from lightgbm import LGBMRegressor


# In[206]:


pipe_lgbm = Pipeline([('preprocessing', preprocessor),
                    ('model', LGBMRegressor(random_state=42))])
#Fit
pipe_lgbm.fit(X_train, y_train)


# In[208]:


#Predict
y_pred_lgbm = pipe_lgbm.predict(X_test)
#Evaluate
print("LightGBM Regressor:")
print("RMSE:", mean_squared_error(y_test, y_pred_lgbm, squared=False))
print("R2 Score:", r2_score(y_test, y_pred_lgbm))


# In[212]:


final = {'Linear Regression': ["RMSE:", mean_squared_error(y_test, y_pred_lr, squared=False), "R2 Score:", r2_score(y_test, y_pred_lr)],
         'Decision Tree': ["RMSE:", mean_squared_error(y_test, y_pred_dt, squared=False), "R2 Score:", r2_score(y_test, y_pred_dt)],
         'Random Forest': ["RMSE:", mean_squared_error(y_test, y_pred_rf, squared=False), "R2 Score:", r2_score(y_test, y_pred_rf)],
         'Gradient Boost':["RMSE:", mean_squared_error(y_test, y_pred_gb, squared=False), "R2 Score:", r2_score(y_test, y_pred_gb)],
         'XGBoost' : ["RMSE:", mean_squared_error(y_test, y_pred_xgb, squared=False), "R2 Score:", r2_score(y_test, y_pred_xgb)],
         'LightGBM': ["RMSE:", mean_squared_error(y_test, y_pred_lgbm, squared=False), "R2 Score:", r2_score(y_test, y_pred_lgbm)]}


# In[214]:


print(final)


# In[216]:


results = pd.DataFrame(final)
print(results)


# In[218]:


# As per RMSE we have to consider the lower=better model performance
# As per R2 we have to consider the higher=better model performance


# In[220]:


#Now, we can consider Linear, Gradient and LightGBM to tune and evaluate , selecting the best model


# Hyperparameter Tuning using Optuna

# In[223]:


get_ipython().system('pip install optuna')


# In[225]:


#For Linear Regression tuning not necessary


# In[227]:


import optuna


# 1. Gradient Boost tuning with optuna

# In[242]:


#Define objective function for optuna
def objective(trial):
    #Hyperparameters to tune
    n_estimators = trial.suggest_int('n_estimators', 50,100)
    max_depth = trial.suggest_int('max_depth',2,10)
    learning_rate = trial.suggest_float('learning_rate', 0.01,0.3, log=True)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    #Define model with suggest hyperparameters
    model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)
    #Define Pipeline
    pipe= Pipeline([('preprocessing', preprocessor), ('model', model)])
    #Fit on training data
    pipe.fit(X_train, y_train)
    #Predict on test data
    y_pred = pipe.predict(X_test)
    #Calculate RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse


# In[244]:


#Run optuna study
study= optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Best trial:")
print(study.best_trial)


# In[248]:


#Train final model with best hyperparameters
best_params = study.best_params
final_model = GradientBoostingRegressor(**best_params, random_state=42)
pipe_gb_tuned = Pipeline([('preprocessing', preprocessor),
                          ('model', final_model)])
pipe_gb_tuned.fit(X_train, y_train)
y_pred_gb_tuned = pipe_gb_tuned.predict(X_test)
print("Tuned Gradient Boosting Regressor:")
print("RMSE:", mean_squared_error(y_test, y_pred_gb_tuned, squared=False))
print("R2 Score:", r2_score(y_test, y_pred_gb_tuned))


# #Before Tuning
# RMSE: 0.44 , R2: 0.78
# #After Tuning
# RMSE: 0.41, R2: 0.81
# #Conclusion: After tuning performance enhanced

# 2. LightGBM tuning with optuna

# In[258]:


#Define objective function for optuna
def objective(trial):
    #Hyperparameters to tune
    n_estimators = trial.suggest_int('n_estimators', 50,1000)
    max_depth = trial.suggest_int('max_depth',3,10)
    learning_rate = trial.suggest_float('learning_rate', 0.005,0.3, log=True)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    num_leaves = trial.suggest_int('num_leaves', 20, 300)
    #Define model with suggest hyperparameters
    model = LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, num_leaves=num_leaves, random_state=42)
    #Define Pipeline
    pipe= Pipeline([('preprocessing', preprocessor), ('model', model)])
    #Fit on training data
    pipe.fit(X_train, y_train)
    #Predict on test data
    y_pred = pipe.predict(X_test)
    #Calculate RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse


# In[260]:


#Run optuna study
study= optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Best trial:")
print(study.best_trial)


# In[262]:


#Train final model with best hyperparameters
best_params = study.best_params
final_model = LGBMRegressor(**best_params, random_state=42)
pipe_lgbm_tuned = Pipeline([('preprocessing', preprocessor),
                          ('model', final_model)])
pipe_lgbm_tuned.fit(X_train, y_train)
y_pred_lgbm_tuned = pipe_lgbm_tuned.predict(X_test)
print("Tuned LGBM Regressor:")
print("RMSE:", mean_squared_error(y_test, y_pred_lgbm_tuned, squared=False))
print("R2 Score:", r2_score(y_test, y_pred_lgbm_tuned))


# #Before Tuning
# RMSE: 0.47 , R2: 0.75
# #After Tuning
# RMSE: 0.45, R2: 0.78
# #Conclusion: After tuning performance enhanced

# In[265]:


# Conclusion: Gradient Boosting performing best after tuning with low RMSE & High R2 Score, saving it for future predictions


# Model Saving & Versioning

# In[268]:


pip install mlflow


# In[270]:


#import mlflow
import mlflow
import mlflow.sklearn
#Log and register model
# start an mlflow run
with mlflow.start_run():
   #log parameters from best_params
   mlflow.log_params(best_params)
   #log metrics
   rmse = mean_squared_error(y_test,y_pred_gb_tuned, squared=False)
   r2 = r2_score(y_test, y_pred_gb_tuned)
   mlflow.log_metric("RMSE", rmse)
   mlflow.log_metric("R2", r2)
   #log the pipeline model
   mlflow.sklearn.log_model(sk_model=pipe_gb_tuned, artifact_path="gradient_boosting-pipeline_model", registered_model_name="GradientBoostingPipelineModel")
   print("Model saved to mlflow & registered successfully.")


# Model Loading & predicting on sample data

# In[273]:


new_data = pd.DataFrame([{
    'Sector': 'Technology',
    'Country': 'USA',
    'Market_Cap_Billions': 120,
    'Carbon_Emissions': 30,
    'Renewable_Energy_Usage': 50,
    'Employee_Satisfaction': 85,
    'Diversity_Ratio': 0.4,
    'Board_Independence': 0.8,
    'CEO_Compensation_Ratio': 90,
    'ROE': 15
}])


# In[275]:


print(new_data)


# In[283]:


#model loading for predictions on sample
import mlflow.pyfunc
model = mlflow.sklearn.load_model (model_uri="models:/GradientBoostingPipelineModel/1")


# In[287]:


new_data.dtypes


# In[295]:


new_data['Sector'] = new_data['Sector'].astype('category')
new_data['Country'] = new_data['Country'].astype('category')


# In[299]:


new_data.dtypes


# In[305]:


new_data.columns


# In[307]:


X_train.columns


# In[311]:


#Column names are not matching with Caps, so decode it now.
new_data.columns = new_data.columns.str.strip().str.lower().str.replace(' ','_')


# In[313]:


new_data.columns


# In[315]:


#Prediction on sample new data
predict = model.predict(new_data)


# In[317]:


print(predict)


# Streamlit GUI creation

# In[329]:


model = mlflow.sklearn.load_model (model_uri="models:/GradientBoostingPipelineModel/1")
import streamlit as st
import mlflow.sklearn
st.title("ESG Score Prediction App")
#choose mode as file uploading
st.write("Upload a CSV file containing required features to predict ESG scores.")
uploaded_file = st.file_uploader("Upload input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", input_df.head())
    if st.button("Predict ESG Scores"):
        try:
            predictions = model.predict(input_df)
            input_df['Predicted_ESG_Score']= predictions
            st.success("Predictions generated successfully!")
            st.write(input_df)
            csv=input_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv,"esg_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("Please upload a csv file to begin.")


# In[ ]:





# In[ ]:




