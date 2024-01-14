import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# GitHub raw file link
github_link = "https://github.com/hoangnguyence/hpconcrete/raw/master/data/hpc_compressive_strength.xlsx"

# Load the Excel file into a DataFrame
df = pd.read_excel(github_link)

# Display the first few rows of the DataFrame
print(df.head())

#predicting concrete strength we'll work with existing data from the MDPI repo

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

# Step 4: Train the Gradient Boosting Regressor Model

# Initialize the Gradient Boosting Regressor
gbr_model = GradientBoostingRegressor()

# Train the model
gbr_model.fit(X_train_scaled, y_train)

# Step 5: Make Predictions on the Test Set

# Transform the test features using the trained scaler
X_test_scaled = scaler.transform(X_test)

# Make predictions on the test set
y_pred = gbr_model.predict(X_test_scaled)

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

# Step 7: User Input and Prediction

# Take user input for curing age
# user_curing_age = float(input('Enter the curing age (in days): '))

# Extract feature names from the DataFrame
feature_names = X.columns.tolist()

# Take user input for all features from a real-time user
user_input = {}
for feature in feature_names:
    user_input[feature] = float(input(f'Enter value for {feature}: '))

# Convert user input to DataFrame
user_df = pd.DataFrame([user_input])

# Make prediction for user input
user_pred_strength = gbr_model.predict(user_df)

# Print the predicted compressive strength
print(f'Predicted Compressive Strength at {feature} days: {user_pred_strength[0]}')

# git clone <repository_url>
# cd <project_directory>
# python -m venv venv
# .\venv\Scripts\activate
# pip freeze > requirements.txt
# pip install -r requirements.txt
# pip install scikit-learn