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
import random


# Load the Excel file into a DataFrame

# Get the current working directory
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)

# Construct the relative path to the Excel file
excel_file_name = "compressive-strength-data.xlsx"
excel_file_path = os.path.join(current_directory, excel_file_name)
print("Excel File Path:", excel_file_path)

df = pd.read_excel(excel_file_path)

# Display the first few rows of the DataFrame
print(df.head())

# Load the Excel file into a DataFrame
df = pd.read_excel(excel_file_path)

# Extract input variables
input_variables = df.iloc[:, :9]  # Assuming the first 8 columns are input variables

# Remove units from column names and "Concrete" from the column name "Concrete compressive strength", and make them lowercase
input_variables.columns = [col.split(' (')[0].replace('Concrete ', '').lower() for col in input_variables.columns]

# Calculate correlation matrix
corr_matrix = input_variables.corr()
# print(corr_matrix)

# Plot correlation heatmap with larger figure size and rotated labels
plt.figure(figsize=(15, 12))  # Increase the figure size for better spacing
heatmap = sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f", annot_kws={"size": 12})

# Rotate the x and y-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotate x-axis labels
plt.yticks(rotation=0, fontsize=12)  # Rotate y-axis labels

plt.show()



# predicting concrete strength we'll work with existing data from the MDPI repo

# Step 1: Explore the Data

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values)

# Handle missing values
df = df.dropna()  # Drop rows with missing values

# Visualize outliers using boxplots
plt.figure(figsize=(14, 8))

# Remove units from parameter names
params_without_units = [param.split(' ')[0] for param in df.drop('Concrete compressive strength (MPa, megapascals) ', axis=1).columns]

# Create boxplot without units in parameter names
sns.boxplot(data=df.drop('Concrete compressive strength (MPa, megapascals) ', axis=1))
plt.xticks(range(len(params_without_units)), params_without_units, fontsize=15)  # Set x-axis ticks with modified parameter names
plt.yticks(fontsize=15)
plt.xlabel('Input Parameters', fontsize=15)
plt.ylabel('Values', fontsize=15)
plt.title('Boxplot of Features', fontsize=18)
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


# Set pandas to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Calculate and print summary statistics for all parameters
summary_stats = df.describe()
print("\nSummary Statistics:\n", summary_stats)

# Summary statistics
# summary_stats = df.describe()
# print("\nSummary Statistics:\n", summary_stats)

# Distribution of the target variable (Compressive Strength)
plt.figure(figsize=(12, 8))
sns.histplot(df['Concrete compressive strength (MPa, megapascals) '], bins=30, kde=True)

# Calculate statistics for compressive strength
mean_val = df['Concrete compressive strength (MPa, megapascals) '].mean()
std_val = df['Concrete compressive strength (MPa, megapascals) '].std()
min_val = df['Concrete compressive strength (MPa, megapascals) '].min()
max_val = df['Concrete compressive strength (MPa, megapascals) '].max()

# Add vertical lines for mean, min, and max
plt.axvline(mean_val, color='orange', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(min_val, color='green', linestyle='dashed', linewidth=2, label='Min')
plt.axvline(max_val, color='red', linestyle='dashed', linewidth=2, label='Max')

# Add normal distribution curve
x_axis = np.linspace(min_val, max_val, 100)
plt.plot(x_axis, norm.pdf(x_axis, mean_val, std_val), color='purple', label='Normal Distribution')

plt.xlabel('Compressive Strength', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.legend(fontsize=12)
plt.show()

# Step 2: Prepare the Data

# Separate features and target variable
X = df.drop('Concrete compressive strength (MPa, megapascals) ', axis=1)
y = df['Concrete compressive strength (MPa, megapascals) ']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

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

    plt.ylabel('Relative Frequency', fontsize=15)
    plt.xlabel(feature, fontsize=15)
    plt.legend(fontsize=12)

plt.tight_layout()
plt.show()

# Step 6: Evaluate the Model

# Remove units from input parameter names
input_variables.columns = [col.split(' (')[0] for col in input_variables.columns]

# Get feature importances
feature_importance = gbr_model.feature_importances_

# Normalize the feature importances to percentage
relative_importance = 100.0 * (feature_importance / feature_importance.max())

# Sort the relative importances and corresponding feature names
sorted_idx = np.argsort(relative_importance)
sorted_features = input_variables.columns[sorted_idx]
sorted_relative_importance = relative_importance[sorted_idx]

# Plot the relative importance of input parameters
plt.figure(figsize=(10, 6))
bars = plt.barh(sorted_features, sorted_relative_importance, color='skyblue')

# Add importance values beside the bars
for bar, importance in zip(bars, sorted_relative_importance):
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f'{importance:.2f}%', 
             va='center', ha='left', fontsize=12, color='black')

plt.xlabel('Relative Importance (%)', fontsize=15)
# plt.ylabel('Input Parameters')
# plt.title('Relative Importance of Input Parameters')
plt.show()

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


# Fit a linear regression line
slope, intercept, _, _, _ = linregress(y_test, y_pred)
fit_line = slope * y_test + intercept

r2_text = f'R-squared (R²): {r2:.3f}'


# Scatter plot for Test Set with Fitted Line
plt.figure(figsize=(14, 10))
plt.scatter(y_test, y_pred, color='green', label='Test Set', alpha=0.7)
# plt.scatter(user_df[user_df.columns[0]], user_pred_strength, color='red', marker='X', s=200, label='User Input')
# plt.title('Actual vs. Predicted Compressive Strength')
plt.xlabel('Actual Compressive Strength', fontsize=15)
plt.ylabel('Predicted Compressive Strength', fontsize=15)

# Fit a linear regression line
slope, intercept, _, _, _ = linregress(y_test, y_pred)
fit_line = slope * y_test + intercept
plt.plot(y_test, fit_line, '--', color='red', linewidth=2, label='Fitted Line')

# Add the equation of the fitted line to the chart
equation_text = f'Fitted Equation: y = {slope:.2f}x + {intercept:.2f}'
plt.text(0.5, 0.92, equation_text, transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')

plt.text(0.5, 0.85, r2_text, transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')

plt.legend(fontsize=12)
plt.show()

# Create an array of sample indices for plotting
sample_indices = np.arange(len(y_test))

# Sort the sample indices
sorted_indices = np.argsort(sample_indices)
sorted_sample_indices = sample_indices[sorted_indices]

# Sort the actual and predicted values accordingly
sorted_y_test = y_test.values[sorted_indices]
sorted_y_pred = y_pred[sorted_indices]

# Plot the scatter plot diagram
plt.figure(figsize=(14, 10))
plt.scatter(sorted_sample_indices, sorted_y_test, color='red', label='Actual', alpha=0.7)
plt.scatter(sorted_sample_indices, sorted_y_pred, color='blue', label='Predicted', alpha=0.7)

# Connect actual and predicted values with lines
plt.plot(sorted_sample_indices, sorted_y_test, color='red', linestyle='-', linewidth=1)
plt.plot(sorted_sample_indices, sorted_y_pred, color='blue', linestyle='-', linewidth=1)

plt.xlabel('Sample Number', fontsize=15)
plt.ylabel('Compressive Strength', fontsize=15)
# plt.title('Actual vs. Predicted Compressive Strength by Sample Number')
plt.legend(fontsize=12)
plt.show()

# Ensure that the sample size does not exceed the available data
sample_size = min(10, len(y_test))
random.seed(100)
sample_indices = random.sample(range(len(y_test)), sample_size)

# Obtain actual compressive strength for the selected samples
actual_strength = y_test.values[sample_indices]

# Predict compressive strength for the selected samples
predicted_strength = y_pred[sample_indices]

# Define the width of each bar
bar_width = 0.35

# Plot a bar chart comparing actual and predicted compressive strength
plt.figure(figsize=(14, 8))

# Bar chart for actual compressive strength
plt.bar(range(sample_size), actual_strength, color='blue', width=bar_width, label='Actual')

# Bar chart for predicted compressive strength
plt.bar([i + bar_width for i in range(sample_size)], predicted_strength, color='orange', width=bar_width, label='Predicted')

plt.xlabel('Sample Number', fontsize=15)
plt.ylabel('Compressive Strength', fontsize=15)

# Set x-axis ticks and labels
plt.xticks([i + bar_width / 2 for i in range(sample_size)], sample_indices)

plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

