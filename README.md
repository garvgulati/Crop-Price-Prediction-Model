This project is a comprehensive machine learning pipeline designed to predict crop prices based on various influencing factors such as crop production, water usage, soil properties, and market conditions. 
It integrates data preprocessing, feature selection, and hyperparameter tuning to deliver accurate predictions using multiple regression models. 
This project provides a scalable, extensible framework for agricultural price prediction and paves the way for innovative solutions in the agri-tech domain. 
This model was trained to make a better model than an already existing model that the Government of India already uses to predict the crop prices i.e. the ARIMA model. 
This was a model that me and my team trained as our submission for the Smart India Hackathon. 

1. Data Integration:
Combines datasets on crop production, water usage, soil analysis, and crop prices.
Handles categorical and numerical variables efficiently through encoding and normalization.

2. Data Preprocessing:
Detects and handles missing values, duplicates, and inconsistent data.
Standardizes numerical features using MinMaxScaler for uniform input scaling.

3. Feature Engineering:
Extracts features such as day of the year from date columns.
Selects relevant features using feature importance scores from a Random Forest Regressor.

4. Regression Models:
Implements and evaluates multiple machine learning algorithms:
XGBoost Regressor
Light Gradient Boosting Machine (LightGBM)
Random Forest Regressor
Gradient Boosting Regressor

5. Hyperparameter Tuning:
Utilizes GridSearchCV for optimizing model parameters.
Compares models based on R² scores and Mean Squared Error (MSE).

6. Visualization:
Uses Matplotlib and Seaborn for data visualization and performance analysis.
Generates R² score and MSE comparison charts for different models.

Tech Stack:
Languages: Python

Libraries:
Data Handling: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-learn, XGBoost, LightGBM
Feature Selection and Scaling: Random Forest, MinMaxScaler
Utilities: GridSearchCV for hyperparameter optimization
