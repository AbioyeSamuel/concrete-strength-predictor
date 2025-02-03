import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import kendalltau
from scipy.stats import norm
from scipy.stats import linregress
import random
import time
from mpl_toolkits.mplot3d import Axes3D
import shap


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

# Print the correlation matrix values in the terminal
print(corr_matrix)




# Compute Kendall's tau nonlinear correlation for each pair of features
kendall_corr_matrix = input_variables.corr(method='kendall')

# Plot the Kendall's tau correlation heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(
    kendall_corr_matrix, 
    annot=True, 
    cmap='cividis', 
    fmt=".2f", 
    annot_kws={"size": 16},  # Increase font size of the values
    cbar_kws={"shrink": 0.8}  # Adjust color bar size if needed
)
plt.xticks(rotation=90, ha='right', fontsize=14)  # Increase font size of x-axis labels
plt.yticks(rotation=0, fontsize=14)  # Increase font size of y-axis labels
plt.show()

# # Adjust pandas display settings to show all columns
# pd.set_option('display.max_columns', None)  # Show all columns
# pd.set_option('display.width', 1000)    

# Print the Kendall's tau correlation matrix values
print("Kendall's Tau Correlation Matrix:\n", kendall_corr_matrix)








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
# plt.title('Boxplot of Features')
plt.show()

# Check data types
data_types = df.dtypes
print("\nData Types:\n", data_types)

# Summary statistics
summary_stats = df.describe()
print("\nSummary Statistics:\n", summary_stats)

# # Adjust pandas display settings to show all columns
# pd.set_option('display.max_columns', None)

# # Generate and print summary statistics
# summary_stats = df.describe()
# print("\nSummary Statistics:\n", summary_stats)

# # Optionally, reset the display settings after printing
# pd.reset_option('display.max_columns')

# Distribution of the target variable (Compressive Strength)
plt.figure(figsize=(8, 6))
sns.histplot(df['Concrete compressive strength (MPa, megapascals) '], bins=30, kde=True)
# plt.title('Distribution of Compressive Strength')
plt.xlabel('Compressive Strength')
plt.ylabel('Frequency')
plt.show()

# Step 2: Prepare the Data

# Separate features and target variable
X = df.drop('Concrete compressive strength (MPa, megapascals) ', axis=1)
y = df['Concrete compressive strength (MPa, megapascals) ']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Step 3: Feature Scaling

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing data
X_test_scaled = scaler.transform(X_test)

# Step 4: Hyperparameter Tuning using GridSearchCV

# Define the parameter grid for XGBoost
param_grid = {

    'n_estimators': [100, 150, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
}

# Initialize the XGBRegressor
xgb_model = XGBRegressor()

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Best model and parameters
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best R² Score on Training Set:", grid_search.best_score_)

# Step 5: Retrain with Best Parameters

# Get the best estimator from the grid search
best_model = grid_search.best_estimator_

# Step 5.5: Make Predictions on the Test Set
# Evaluate the Model
y_pred = best_model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)


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
user_pred_strength = best_model.predict(user_input_scaled)

# Print the predicted compressive strength
print(f'Predicted Compressive Strength at {feature} days: {user_pred_strength[0]}')


# Measure training time
start_time = time.time()
best_model.fit(X_train_scaled, y_train)  # Train the model
training_time = time.time() - start_time

# Output training time
print(f"Training Time: {training_time:.2f} seconds")



# Scatter plot for Test Set with Fitted Line
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='green', label='Test Set', alpha=0.7)
# plt.title('Actual vs. Predicted Compressive Strength')
plt.xlabel('Actual Compressive Strength')
plt.ylabel('Predicted Compressive Strength')

# Fit a linear regression line
slope, intercept, _, _, _ = linregress(y_test, y_pred)
fit_line = slope * y_test + intercept
plt.plot(y_test, fit_line, '--', color='red', linewidth=2, label='Fitted Line')

# Add the equation of the fitted line to the chart
equation_text = f'Fitted Line: y = {slope:.2f}x + {intercept:.2f}'
plt.text(0.5, 0.92, equation_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

r2_text = f'R-squared (R²): {r2:.3f}'
plt.text(0.5, 0.85, r2_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

plt.legend()
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
plt.figure(figsize=(10, 6))
plt.scatter(sorted_sample_indices, sorted_y_test, color='red', label='Actual', alpha=0.7)
plt.scatter(sorted_sample_indices, sorted_y_pred, color='blue', label='Predicted', alpha=0.7)

# Connect actual and predicted values with lines
plt.plot(sorted_sample_indices, sorted_y_test, color='red', linestyle='-', linewidth=1)
plt.plot(sorted_sample_indices, sorted_y_pred, color='blue', linestyle='-', linewidth=1)

plt.xlabel('Sample Number')
plt.ylabel('Compressive Strength')
# plt.title('Actual vs. Predicted Compressive Strength by Sample Number')
plt.legend()
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


# SHAP Code
model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=0)
model.fit(X_train_scaled, y_train)
# Initialize SHAP explainer
explainer = shap.Explainer(model, X_train_scaled)

# Compute SHAP values
shap_values = explainer(X_test_scaled)


# Assign the actual feature names to the SHAP values object
shap_values.feature_names = input_variables.columns

# Summary plot (bar chart of feature importance)
shap.summary_plot(shap_values, X_test, feature_names=input_variables.columns, plot_type="bar")

# Detailed summary plot
shap.summary_plot(shap_values, X_test, feature_names=input_variables.columns)

# Bar plot of feature importance
shap.plots.bar(shap_values)


# # Visualize cross-validation results
# plt.figure(figsize=(8, 6))
# plt.grid()
# # Fit the model (assuming `best_model` is already trained)
# explainer = shap.TreeExplainer(best_model)
# shap_values = explainer.shap_values(X)

# # SHAP summary plot for feature importance
# shap.summary_plot(shap_values, X, feature_names=input_variables.columns)

# # Relative Importance of Features using SHAP
# shap_importance = np.abs(shap_values).mean(axis=0)
# relative_importance = 100.0 * (shap_importance / shap_importance.max())
# sorted_idx = np.argsort(relative_importance)

# # Short forms mapping for feature names
# feature_short_forms = {
#     "Cement": "cem",
#     "Blast": "bfs",  # Handles "Blast furnace slag"
#     "Fly": "fa",     # Handles "Fly ash"
#     "Water": "wtr",
#     "Superplasticizer": "sp",
#     "Coarse": "cag",   # Handles "Coarse aggregate"
#     "Fine": "fag",     # Handles "Fine aggregate"
#     "Age": "age",
# }

# # Apply short forms and sort
# sorted_features = [feature_short_forms[feature.split(' ')[0]] for feature in X.columns[sorted_idx]]
# sorted_relative_importance = relative_importance[sorted_idx]

# # Step 1: Create a 3D plot
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Step 2: Position bars in 3D
# x_pos = np.arange(len(sorted_features)) * 1.5  # Increased spacing between bars
# y_pos = np.zeros(len(sorted_features))  # Y positions (set to zero)
# z_pos = np.zeros(len(sorted_features))  # Z positions (base height of bars)

# # Bar dimensions
# bar_width = 0.4
# bar_depth = 0.3
# bar_height = sorted_relative_importance  # Heights correspond to the importance values

# # Step 3: Plot 3D bars
# colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))  # Color map based on importance
# ax.bar3d(x_pos, y_pos, z_pos, bar_width, bar_depth, bar_height, color=colors, alpha=0.8)

# # Step 4: Add labels and customize the chart
# ax.set_xticks(x_pos)
# ax.set_xticklabels(sorted_features, rotation=45, ha='right', fontsize=10)
# ax.set_yticks([])  # Remove Y-ticks for cleaner look
# ax.set_xlabel('Features', fontsize=12, labelpad=30)
# ax.set_zlabel('Relative Importance (%)', fontsize=12, labelpad=10)

# # Annotate each bar with its importance value
# for i in range(len(sorted_relative_importance)):
#     ax.text(x_pos[i], y_pos[i], bar_height[i] + 2, f"{bar_height[i]:.2f}%", color='skyblue', ha='center', fontsize=12)

# # Display the plot
# plt.tight_layout()
# plt.show()



