import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# GitHub raw file link for the 1030 data set
github_link = "https://github.com/hoangnguyence/hpconcrete/raw/master/data/hpc_compressive_strength_1030.xlsx"

# Load the Excel file into a DataFrame
current_directory = os.getcwd()
excel_file_name = "hpc_compressive_strength.xlsx"
excel_file_path = os.path.join(current_directory, excel_file_name)
df = pd.read_excel(excel_file_path)

# Display the first few rows of the DataFrame
print(df.head())

# Step 1: Explore the Data

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values)

# Check data types
data_types = df.dtypes
print("\nData Types:\n", data_types)

# Summary statistics
summary_stats = df.describe()
print("\nSummary Statistics:\n", summary_stats)

# Distribution of the target variable (Compressive Strength)
plt.figure(figsize=(8, 6))
sns.histplot(df['Concrete compressive strength (MPa, megapascals) '], bins=30, kde=True)
plt.title('Distribution of Compressive Strength')
plt.xlabel('Compressive Strength')
plt.ylabel('Frequency')
plt.show()

# Step 2: Prepare the Data

# Separate features and target variable
X = df.drop('Concrete compressive strength (MPa, megapascals) ', axis=1)
y = df['Concrete compressive strength (MPa, megapascals) ']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Feature Scaling

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing data
X_test_scaled = scaler.transform(X_test)

# Step 4: Train the GEP Regressor Model

# Initialize the GEP Regressor
gep_model = SymbolicRegressor()

# Train the model
gep_model.fit(X_train_scaled, y_train)

# Step 5: Make Predictions on the Test Set

# Transform the test features using the trained scaler
X_test_scaled = scaler.transform(X_test)

# Make predictions on the test set
y_pred = gep_model.predict(X_test_scaled)

# Visualize actual vs. predicted compressive strength
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.title('Actual vs. Predicted Compressive Strength')
plt.xlabel('Actual Compressive Strength')
plt.ylabel('Predicted Compressive Strength')
plt.show()

# Step 6: Evaluate the Model

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared (R2): {r2}')

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')

# Step 7: User Input and Prediction

# Extract feature names from the DataFrame
feature_names = X.columns.tolist()

# Take user input for all features from a real-time user
user_input = {}
for feature in feature_names:
    user_input[feature] = float(input(f'Enter value for {feature}: '))

# Convert user input to DataFrame
user_df = pd.DataFrame([user_input])

# Transform the user input using the trained scaler
user_input_scaled = scaler.transform(user_df)

# Make prediction for user input
user_pred_strength = gep_model.predict(user_input_scaled)

# Print the predicted compressive strength
print(f'Predicted Compressive Strength: {user_pred_strength[0]}')
