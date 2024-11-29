from statistics import linear_regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import linregress
from scipy.stats import norm
import random

# Print package versions
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)

# Load the Excel file into a DataFrame
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)

excel_file_name = "hpc_compressive_strength.xlsx"
excel_file_path = os.path.join(current_directory, excel_file_name)
print("Excel File Path:", excel_file_path)

df = pd.read_excel(excel_file_path)
print(df.head())

# Remove units from column names and "Concrete" from the column name "Concrete compressive strength", and make them lowercase
input_variables = df.iloc[:, :9]
input_variables.columns = [col.split(' (')[0].replace('Concrete ', '').lower() for col in input_variables.columns]

# Plot correlation heatmap
corr_matrix = input_variables.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f", annot_kws={"size": 12})
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.show()

# Handle missing values
df = df.dropna()

# Remove units from parameter names
params_without_units = [param.split(' ')[0] for param in df.drop('Concrete compressive strength (MPa, megapascals) ', axis=1).columns]

# Boxplot visualization of features
plt.figure(figsize=(14, 8))
params_without_units = [param.split(' ')[0] for param in df.drop('Concrete compressive strength (MPa, megapascals) ', axis=1).columns]
sns.boxplot(data=df.drop('Concrete compressive strength (MPa, megapascals) ', axis=1))
plt.xticks(range(len(params_without_units)), params_without_units, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Input Parameters', fontsize=15)
plt.ylabel('Values', fontsize=15)
plt.title('Boxplot of Features', fontsize=18)
plt.show()

# Summary statistics
print("\nSummary Statistics:\n", df.describe())

# Distribution of the target variable (Compressive Strength)
plt.figure(figsize=(12, 8))
sns.histplot(df['Concrete compressive strength (MPa, megapascals) '], bins=30, kde=True)
mean_val = df['Concrete compressive strength (MPa, megapascals) '].mean()
std_val = df['Concrete compressive strength (MPa, megapascals) '].std()
min_val = df['Concrete compressive strength (MPa, megapascals) '].min()
max_val = df['Concrete compressive strength (MPa, megapascals) '].max()
plt.axvline(mean_val, color='orange', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(min_val, color='green', linestyle='dashed', linewidth=2, label='Min')
plt.axvline(max_val, color='red', linestyle='dashed', linewidth=2, label='Max')
x_axis = np.linspace(min_val, max_val, 100)
plt.plot(x_axis, norm.pdf(x_axis, mean_val, std_val), color='purple', label='Normal Distribution')
plt.xlabel('Compressive Strength', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.legend(fontsize=12)
plt.show()

# Prepare the data
X = df.drop('Concrete compressive strength (MPa, megapascals) ', axis=1)
y = df['Concrete compressive strength (MPa, megapascals) ']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3]
}
gbr_model = GradientBoostingRegressor(random_state=0)
grid_search = GridSearchCV(estimator=gbr_model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Best model and parameters
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best R² Score on Training Set:", grid_search.best_score_)

# Evaluate the model on the test set
y_pred = best_model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Test Set R²: {r2}")
print(f"Test Set MSE: {mse}")
print(f"Test Set MAE: {mae}")

# Cross-validation results
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, scoring='r2', cv=kf)
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean R² Score: {cv_scores.mean()}")
print(f"Standard Deviation of R² Scores: {cv_scores.std()}")
# Visualize cross-validation results
plt.figure(figsize=(8, 6))
plt.plot(range(1, n_splits + 1), cv_scores, marker='o', label='R² Score per Fold', color='blue')
plt.axhline(cv_scores.mean(), color='red', linestyle='--', label='Mean R² Score')
plt.fill_between(
    range(1, n_splits + 1),
    cv_scores.mean() - cv_scores.std(),
    cv_scores.mean() + cv_scores.std(),
    color='red',
    alpha=0.2,
    label='Standard Deviation Range'
)
plt.title('Cross-Validation R² Scores', fontsize=15)
plt.xlabel('Fold', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.legend(fontsize=10)
plt.grid()

# Feature importance
feature_importance = best_model.feature_importances_
relative_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(relative_importance)
sorted_features = X.columns[sorted_idx]
sorted_relative_importance = relative_importance[sorted_idx]
plt.figure(figsize=(10, 6))
bars = plt.barh(sorted_features, sorted_relative_importance, color='skyblue')
for bar, importance in zip(bars, sorted_relative_importance):
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f'{importance:.2f}%', va='center', ha='left', fontsize=12)
plt.xlabel('Relative Importance (%)', fontsize=15)
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

# Make prediction for user input using the best_model
user_pred_strength = best_model.predict(user_input_scaled)

# Print the predicted compressive strength
print(f'Predicted Compressive Strength at {feature} days: {user_pred_strength[0]}')


# Fit a linear regression line
slope, intercept, _, _, _ = linregress(y_test, y_pred)
fit_line = slope * y_test + intercept

r2_text = f'R-squared (R²): {r2:.3f}'


# Scatter plot for Test Set with Fitted Line
plt.figure(figsize=(14, 10))
plt.scatter(y_test, y_pred, color='green', label='Test Set', alpha=0.7)

# Fit a linear regression line
slope, intercept, _, _, _ = linregress(y_test, y_pred)
fit_line = slope * y_test + intercept
plt.plot(y_test, fit_line, '--', color='red', linewidth=2, label='Fitted Line')

# Add the equation of the fitted line to the chart
equation_text = f'Fitted Equation: y = {slope:.2f}x + {intercept:.2f}'
plt.text(0.5, 0.92, equation_text, transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')

r2_text = f'R-squared (R²): {r2:.3f}'
plt.text(0.5, 0.85, r2_text, transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')

plt.xlabel('Actual Compressive Strength', fontsize=15)
plt.ylabel('Predicted Compressive Strength', fontsize=15)
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

