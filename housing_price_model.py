# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 2: Load and preprocess the dataset
data = pd.read_csv('housing.csv')

# Handle missing values 
data.fillna(data.mean(), inplace=True)

# Convert categorical variables to binary using one-hot encoding
data = pd.get_dummies(data, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning'])

# Step 3: Feature Engineering 
# For example, create a 'totalrooms' feature by combining 'bedrooms' and 'bathrooms'
data['totalrooms'] = data['bedrooms'] + data['bathrooms']

# Step 4: Handling Multicollinearity 
# Calculate the correlation matrix and deal with highly correlated features
X = data.drop('price', axis=1)
correlation_matrix = X.corr()

# Visualize the correlation matrix 
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Identify highly correlated features
threshold = 0.7
highly_correlated_pairs = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            highly_correlated_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

print("Highly Correlated Feature Pairs:")
for feature1, feature2 in highly_correlated_pairs:
    print(f"{feature1} - {feature2}")

# Handle multicollinearity 
# 1. Remove one feature from each highly correlated pair
# For example:
# X = X.drop(['feature_to_remove'], axis=1)

# Step 5: Data Splitting
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Selection and Training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 7: Model Evaluation
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

# Step 8: Fine-Tuning 


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

best_rf_model = grid_search.best_estimator_

# Step 9: Interpretation and Insights
feature_importances = best_rf_model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

print("Feature Importances:")
for i in sorted_indices:
    print(f"{X.columns[i]}: {feature_importances[i]}")

# Step 10: Deployment and Monitoring
# Deploy the model for predictions on new data
# You can save the trained model and use it for making predictions on new data

import joblib

# Save the best model
joblib.dump(best_rf_model, 'housing_price_model.pkl')

# Load the model for predictions
loaded_model = joblib.load('housing_price_model.pkl')

# Make predictions on new data
new_data = pd.DataFrame({
    'area': [1500],
    'bedrooms': [3],
    'bathrooms': [2],
    'stories': [2],
    'mainroad_1': [1],
    'guestroom_1': [0],
    'basement_1': [1],
    'hotwaterheating_1': [0],
    'airconditioning_1': [1],
    'totalrooms': [5]
})

# Scale the new data using the same scaler
new_data_scaled = scaler.transform(new_data)

# Predict the price using the loaded model
predicted_price = loaded_model.predict(new_data_scaled)
print(f"Predicted Price: {predicted_price[0]}")
