import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import norm
from scipy.stats import linregress


# GitHub raw file link
github_link = "https://github.com/hoangnguyence/hpconcrete/raw/master/data/hpc_compressive_strength.xlsx"

# Load the Excel file into a DataFrame
# df = pd.read_excel(github_link)

# Get the current working directory
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)

# Construct the relative path to the Excel file
excel_file_name = "hpc_compressive_strength.xlsx"
excel_file_path = os.path.join(current_directory, excel_file_name)
print("Excel File Path:", excel_file_path)

df = pd.read_excel(excel_file_path)

# Display the first few rows of the DataFrame
print(df.head())

# predicting concrete strength we'll work with existing data from the MDPI repo

# Step 1: Explore the Data

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values)

# Handle missing values
df = df.dropna()  # Drop rows with missing values

# Visualize outliers using boxplots
plt.figure(figsize=(10, 7))

# Remove units from parameter names
params_without_units = [param.split(' ')[0] for param in df.drop('Concrete compressive strength (MPa, megapascals) ', axis=1).columns]

# Create boxplot without units in parameter names
sns.boxplot(data=df.drop('Concrete compressive strength (MPa, megapascals) ', axis=1))
plt.xticks(range(len(params_without_units)), params_without_units)  # Set x-axis ticks with modified parameter names
plt.title('Boxplot of Features')
plt.show()

# Remove outliers using IQR method
# def remove_outliers_iqr(dataframe, column):
#     Q1 = dataframe[column].quantile(0.25)
#     Q3 = dataframe[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]

# # Apply IQR method to each feature
# for feature in df.columns:
#     if feature != 'Concrete compressive strength (MPa, megapascals) ':
#         df = remove_outliers_iqr(df, feature)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=22)

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


# Step 5: Boxplots for Range of Values

# plt.figure(figsize=(12, 10))

# Iterate through each feature for plotting boxplots
# for i, feature in enumerate(X.columns):
#     plt.subplot(3, 3, i+1)
#     plt.subplots_adjust(hspace=0.5, wspace=0.5)

#     # Plot boxplot
#     sns.boxplot(X[feature], color='lightblue')

#     # Add title
#     plt.title(f'Boxplot of {feature}')
#     plt.ylabel('Values')

#     # Calculate statistics for the current parameter
#     min_val = X[feature].min()
#     max_val = X[feature].max()

#     # Add vertical lines for min and max
#     plt.axvline(min_val, color='green', linestyle='dashed', linewidth=2, label='Min')
#     plt.axvline(max_val, color='red', linestyle='dashed', linewidth=2, label='Max')

#     plt.legend()

# plt.tight_layout()
# plt.show()

# Step 5.5: Normal Distribution Curve and Bar Charts

plt.figure(figsize=(12, 10))

# Iterate through each feature for plotting
for i, feature in enumerate(X.columns):
    plt.subplot(3, 3, i+1)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    # Plot histogram
    sns.histplot(X[feature], kde=True, color='skyblue', stat='density')

    # Calculate statistics for the current parameter
    mean_val = X[feature].mean()
    std_val = X[feature].std()
    min_val = X[feature].min()
    max_val = X[feature].max()

    # Add vertical lines for mean, min, and max
    plt.axvline(mean_val, color='orange', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(min_val, color='green', linestyle='dashed', linewidth=2, label='Min')
    plt.axvline(max_val, color='red', linestyle='dashed', linewidth=2, label='Max')

    # Add normal distribution curve
    x_axis = np.linspace(min_val, max_val, 100)
    plt.plot(x_axis, norm.pdf(x_axis, mean_val, std_val), color='purple', label='Normal Distribution')

    plt.ylabel('Relative Frequency')
    plt.legend()


plt.tight_layout()
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
user_pred_strength = gbr_model.predict(user_input_scaled)

# Print the predicted compressive strength
print(f'Predicted Compressive Strength at {feature} days: {user_pred_strength[0]}')


# Scatter plot for Validation Set with Fitted Line
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Validation Set')
plt.title('Actual vs. Predicted Compressive Strength in Validation Set')
plt.xlabel('Actual Compressive Strength')
plt.ylabel('Predicted Compressive Strength')

# Fit a linear regression line
slope, intercept, _, _, _ = linregress(y_test, y_pred)
fit_line = slope * y_test + intercept
plt.plot(y_test, fit_line, '--', color='red', linewidth=2, label='Fitted Line')

# Add the equation of the fitted line to the chart
equation_text = f'Fitted Line: y = {slope:.2f}x + {intercept:.2f}'
plt.text(0.5, 0.92, equation_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

r2_text = f'R-squared (R2): {r2:.3f}'
plt.text(0.5, 0.85, r2_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

plt.legend()
plt.show()

# Scatter plot for Test Set with Fitted Line
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='green', label='Test Set', alpha=0.7)
# plt.scatter(user_df[user_df.columns[0]], user_pred_strength, color='red', marker='X', s=200, label='User Input')
plt.title('Actual vs. Predicted Compressive Strength')
plt.xlabel('Actual Compressive Strength')
plt.ylabel('Predicted Compressive Strength')

# Fit a linear regression line
slope, intercept, _, _, _ = linregress(y_test, y_pred)
fit_line = slope * y_test + intercept
plt.plot(y_test, fit_line, '--', color='red', linewidth=2, label='Fitted Line')

# Add the equation of the fitted line to the chart
equation_text = f'Fitted Line: y = {slope:.2f}x + {intercept:.2f}'
plt.text(0.5, 0.92, equation_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

plt.text(0.5, 0.85, r2_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

plt.legend()
plt.show()