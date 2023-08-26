# House_Price_Prediction
This repository contains a complete machine learning pipeline for predicting housing prices using a dataset of housing features. 
The pipeline includes data preprocessing, feature engineering, model selection, training, evaluation, hyperparameter tuning, interpretation, and deployment.


## Introduction

Predicting housing prices is a common machine learning task. 
This repository demonstrates a step-by-step pipeline to build, train, evaluate, and deploy a housing price prediction model using Python and popular machine learning libraries.

## Installation

   Clone this repository
   git clone https://github.com/Nour11111/House_Price_Prediction.git
   
   cd House_Price_Prediction
   
   Install the required Python libraries
   
   Place  housing dataset in the project directory as 'housing.csv'
   
   Run the main script to execute the complete pipeline
   
   python housing_price_model.py
   
   
Follow the console outputs to observe each step of the pipeline.

## Pipeline Steps:
Data Preprocessing: Load the dataset and handle missing values using mean imputation. Convert categorical variables to binary using one-hot encoding.

Feature Engineering: Create a new feature 'totalrooms' by summing 'bedrooms' and 'bathrooms'.

Handling Multicollinearity: Calculate the correlation matrix and identify highly correlated features. Handle multicollinearity by removing one feature from each correlated pair.

Data Splitting: Split the dataset into training and testing sets for model training and evaluation.

Model Selection and Training: Standardize the features and train a Linear Regression model. Evaluate the model using Root Mean Squared Error (RMSE).

Hyperparameter Tuning: Use Grid Search with cross-validation to tune hyperparameters for a RandomForestRegressor model.

Interpretation: Obtain feature importances from the best RandomForest model to understand feature contributions.

Deployment: Save the best model using joblib. Load the model to make predictions on new data.

Deployment
The best-trained model can be saved and deployed to make predictions on new data. The saved model can be loaded and used .

