#!/usr/bin/env python
# coding: utf-8

# # ***CROP SEEKERS***

# Dataset link: https://www.kaggle.com/datasets/suraj520/agricultural-data-for-rajasthan-india-2018-2019/data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


crop_production_df = pd.read_csv('crop_production_data.csv')
water_usage_df = pd.read_csv('water_usage_data.csv')
soil_analysis_df = pd.read_csv('soil_analysis_data.csv')
crop_price_df = pd.read_csv('crop_price_data.csv')


# In[ ]:


crop_production_df.head()


# In[ ]:


water_usage_df.head()


# In[ ]:


soil_analysis_df.head()


# In[ ]:


crop_price_df.head()


# In[ ]:


# Merging crop_production with water_usage
merged_df_1 = pd.merge(crop_production_df, water_usage_df, on=["District", "Crop"], how="inner")

# Merging the result of above with soil_analysis
merged_df_2 = pd.merge(merged_df_1, soil_analysis_df, on="District", how="inner")

# Merging the immediate above with crop_price
merged_final = pd.merge(merged_df_2, crop_price_df, on=["District", "Crop"], how="inner")


# In[ ]:


merged_df = merged_final.sample(frac=0.005, random_state=42)


# In[ ]:


merged_df.shape


# In[ ]:


merged_df.head()


# In[ ]:


merged_df.dtypes


# In[ ]:


merged_df.info()


# In[ ]:


merged_df.describe()


# In[ ]:


merged_df.duplicated().sum()


# In[ ]:


# Check for missing values in each column
missing_values = merged_df.isnull().sum()
print(missing_values)


# In[ ]:


# Converting an object type representing a date to a datetime type so that we can perform arthematic operations on days,weeks,years and extract info about months,dates etc
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
print(merged_df['Date'])


# In[ ]:


# verifying if Date column is of type DateTime
merged_df.dtypes


# In[ ]:


merged_df['Date'] = pd.to_datetime(merged_df['Date'])

# Extract features from Date column format yyyy-mm-dd
# merged_df['Year'] = merged_df['Date'].dt.year
# merged_df['Month'] = merged_df['Date'].dt.month
# merged_df['Day'] = merged_df['Date'].dt.day
merged_df['Day_of_Year'] = merged_df['Date'].dt.dayofyear

merged_df = merged_df.drop(columns=['Date'])


# In[ ]:


print(merged_df.columns)


# In[ ]:


# standardising numerical values

from sklearn.preprocessing import MinMaxScaler

# Selecting the numerical columns
numerical_columns = [
    'Area (hectares)',
    'Yield (quintals)',
    'Production (metric tons)',
    'Water Consumption (liters/hectare)',
    'Water Availability (liters/hectare)',
    'pH Level',
    'Organic Matter (%)',
    'Nitrogen Content (kg/ha)',
    'Phosphorus Content (kg/ha)',
    'Potassium Content (kg/ha)',
    'Price (INR/quintal)',
    'Day_of_Year'
]

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Apply the scaler to the numerical columns
merged_df[numerical_columns] = scaler.fit_transform(merged_df[numerical_columns])


# In[ ]:


merged_df.head()


# In[ ]:


print(merged_df.columns)


# **The pandas-profiling library, named "ydata-profiling", is a Python library which is designed to automate the process of Exploratory Data Analysis (EDA). It generates a comprehensive and interactive report that provides a detailed overview of a dataset, making it easier to understand the data's structure, contents, and potential issues without writing extensive code.**

# In[ ]:


get_ipython().system('pip install ydata-profiling')


# In[ ]:


from ydata_profiling import ProfileReport


# In[ ]:


profile = ProfileReport(merged_df, title="Merged DataFrame Profiling Report", explorative=True)
profile.to_notebook_iframe()


# In[ ]:


print(merged_df['District'].unique())


# In[ ]:


print(merged_df['Crop'].unique())


# In[ ]:


print(merged_df['Season'].unique())


# In[ ]:


print(merged_df['Irrigation Method'].unique())


# In[ ]:


print(merged_df['Soil Type'].unique())


# In[ ]:


print(merged_df['Market'].unique())


# In[ ]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

categorical_columns = ['District', 'Crop', 'Season', 'Irrigation Method', 'Soil Type', 'Market']

label_encoders = {}
mappings = {}

for col in categorical_columns:
    le = LabelEncoder()
    merged_df[col] = le.fit_transform(merged_df[col])
    label_encoders[col] = le
    mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

for col, mapping in mappings.items():
    print(f"Mapping for {col}:")
    for category, encoded_value in mapping.items():
        print(f"  {category}: {encoded_value}")
    print()


# In[ ]:


# check whether all categorical data is label encoded
print(merged_df['Market'].unique())
print(merged_df['Soil Type'].unique())
print(merged_df['Irrigation Method'].unique())
print(merged_df['Season'].unique())
print(merged_df['Crop'].unique())
print(merged_df['District'].unique())


# In[ ]:


merged_df.shape


# In[ ]:


print(merged_df.columns)


# # Spliting the Data into Training and Testing Sets (80:20)

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# x is input
x = merged_df.drop(columns=['Price (INR/quintal)'])
# y is output
y = merged_df['Price (INR/quintal)']


# In[ ]:


merged_df.shape


# In[ ]:


# x_train and y_train are the training data
# x_test and y_test are the testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[ ]:


print("x_train shape is: ", x_train.shape)
print("y_train shape is: ",y_train.shape)
print("x_test shape is: ",x_test.shape)
print("y_test shape is: ",y_test.shape)


# 
# 
# ---
# 
# 
# # Feature Selection with Random Forest
# 
# 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


# Initialize the RandomForestRegressor model
model = RandomForestRegressor(random_state=42)


# In[ ]:


# Fit the model to the training data
model.fit(x_train, y_train)


# In[ ]:


# Get feature importances from the trained model
importances = model.feature_importances_


# In[ ]:


# Create a DataFrame to display the features and their importance scores
feature_importances = pd.DataFrame({'Feature': x_train.columns, 'Importance': importances})


# In[ ]:


# Sort the features by their importance score in descending order
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)


# In[ ]:


# Set a threshold for feature importance
threshold = 0.05  # Example threshold, you can adjust this value


# In[ ]:


# Select features with importance above the threshold
selected_features = feature_importances[feature_importances['Importance'] > threshold]['Feature']


# In[ ]:


print(f"Selected features with importance above {threshold}:")
print(selected_features)


# In[ ]:


# Select only the features that meet the threshold from the training and testing datasets
x_train_selected = x_train[selected_features]
x_test_selected = x_test[selected_features]


# # ***Applying Models***

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# # 1.  XGBoost Regressor

# In[ ]:


from xgboost import XGBRegressor


# In[ ]:


param_grid = {
    'n_estimators': [1500, 1700],  # Number of trees
    'learning_rate': [0.01, 0.02],  # Lower learning rate for more precise boosting
    'max_depth': [12,13]  # Deeper trees to capture more complex patterns
}


# In[ ]:


# Initialize the XGBRegressor
model = XGBRegressor(random_state=42)


# In[ ]:


# Initialize GridSearchCV with the model and parameter grid
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='r2')


# In[ ]:


# Fit the GridSearchCV to the training data
grid_search.fit(x_train_selected, y_train)


# In[ ]:


# Using the best model to make predictions on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test_selected)

# Evaluating the model performance on the test set
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test Set Mean Squared Error: {mse}")
print(f"Test Set R-squared: {r2}")


# # 2. Light Gradient Boosting Machine Regressor

# In[ ]:


import lightgbm as lgb


# In[ ]:


# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [1000, 1500],  # Number of boosting iterations
    'learning_rate': [0.01, 0.05],  # Learning rate
    'max_depth': [10, 15],  # Maximum depth of trees
    'num_leaves': [50, 100],  # Number of leaves in one tree
}


# In[ ]:


# Initialize the LightGBM Regressor
model = lgb.LGBMRegressor(random_state=42)


# In[ ]:


# Initialize GridSearchCV with the model and parameter grid
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='r2')


# In[ ]:


# fitting model
grid_search.fit(x_train_selected, y_train)


# In[ ]:


# Using the best model to make predictions on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test_selected)

# Evaluating the model performance on the test set
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test Set Mean Squared Error: {mse}")
print(f"Test Set R-squared: {r2}")


# # 3. Random Forest Regressor
# 
# 
# 
# 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


param_grid = {
    'n_estimators': [800,1200],  # Number of trees in the forest
    'max_depth': [None],  # Allow trees to grow deep enough to capture patterns
    'max_features': ['sqrt', 'log2']  # Consider a subset of features at each split
}


# In[ ]:


# Initializing RandomForestRegressor
model = RandomForestRegressor(random_state=42, n_jobs=-1)


# In[ ]:


# Initializing GridSearchCV with the parameter grid
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='r2')


# In[ ]:


# Fitting GridSearchCV to the training data
grid_search.fit(x_train_selected, y_train)


# In[ ]:


# print("Best Hyperparameters:", grid_search.best_params_)
# print("Best Cross-Validation R2 Score:", grid_search.best_score_)


# In[ ]:


# Using the best model to make predictions on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test_selected)

# Evaluating the model performance on the test set
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test Set Mean Squared Error: {mse}")
print(f"Test Set R-squared: {r2}")


# # 4. Gradient Boosting Regressor
# 

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


# In[ ]:


# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [900, 1000],  # Number of trees
    'learning_rate': [0.01, 0.02],  # Learning rate
    'max_depth': [15, 20]  # Depth of each tree
}


# In[ ]:


# Initialize the GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=42)


# In[ ]:


# Initialize GridSearchCV with the model and parameter grid
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=3, scoring='r2')


# In[ ]:


# Fit the GridSearchCV to the training data
grid_search.fit(x_train_selected, y_train)


# In[ ]:


# Using the best model to make predictions on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test_selected)

# Evaluating the model performance on the test set
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test Set Mean Squared Error: {mse}")
print(f"Test Set R-squared: {r2}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


models = ['GradientBoostingRegressor', 'XGBoostRegressor', 'LightGradientBoostingMachineRegressor', 'RandomForestRegressor']
r2_scores = [0.9844, 0.9799, 0.9476, 0.9802]  # R² scores
mse_values = [0.028, 0.036, 0.094, 0.035]  # MSE values

# Plotting R² Scores
plt.figure(figsize=(12, 6))
bars = plt.bar(models, r2_scores, color='skyblue')
plt.title('R² Scores for Different Models', fontsize=16)
plt.xlabel('Models', fontsize=14)
plt.ylabel('R² Score', fontsize=14)
plt.ylim(0.94, 1)  # Adjusted y-axis limit to focus on high accuracy values

# Keep the x-tick labels horizontal
plt.xticks(fontsize=8, rotation=0, ha='center')
plt.yticks(fontsize=12)

# Adding the value on top of the bars with four decimal places
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f"{yval:.4f}", ha='center', fontsize=12)

plt.show()


# In[ ]:


models = ['GradientBoostingRegressor', 'XGBoostRegressor', 'LightGradientBoostingMachineRegressor', 'RandomForestRegressor']
mse_values = [0.000281, 0.000362, 0.000945, 0.000356]  # MSE values

# Plotting Mean Squared Error (MSE)
plt.figure(figsize=(10, 6))
bars = plt.bar(models, mse_values, color='salmon')
plt.title('Mean Squared Error (MSE) for Different Models', fontsize=16)
plt.xlabel('Models', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.xticks(fontsize=8)
plt.yticks(fontsize=12)

# Extend the y-axis limit to provide more space above the bars
plt.ylim(0, max(mse_values) + max(mse_values) * 0.2)

# Adding the value on top of the bars with proper positioning and precision
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + max(mse_values) * 0.05, f"{yval:.6f}", ha='center', fontsize=12)

plt.show()


# In[ ]:




