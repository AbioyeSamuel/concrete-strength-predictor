import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import linregress
import os
import random

# Load the Excel file into a DataFrame
current_directory = os.getcwd()
excel_file_name = "hpc_compressive_strength.xlsx"
excel_file_path = os.path.join(current_directory, excel_file_name)
df = pd.read_excel(excel_file_path)

# Step 1: Data Exploration (unchanged)
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values)

# Visualize outliers using boxplots
plt.figure(figsize=(10, 7))
params_without_units = [param.split(' ')[0] for param in df.drop('Concrete compressive strength (MPa, megapascals) ', axis=1).columns]
sns.boxplot(data=df.drop('Concrete compressive strength (MPa, megapascals) ', axis=1))
plt.xticks(range(len(params_without_units)), params_without_units)  # Set x-axis ticks with modified parameter names
plt.title('Boxplot of Features')
plt.show()

# Summary statistics
summary_stats = df.describe()
print("\nSummary Statistics:\n", summary_stats)

# Step 2: Data Preparation
X = df.drop('Concrete compressive strength (MPa, megapascals) ', axis=1)
y = df['Concrete compressive strength (MPa, megapascals) ']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Step 3: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Hyperparameter Tuning with GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(100,), (150,)],  # Fewer choices to speed up search
    'activation': ['relu', 'tanh'],  # Keep these as is
    'solver': ['adam'],  # 'lbfgs' is slower, so using 'adam' only
    'alpha': [0.0001, 0.001],  # Reduce the number of values to test
    'learning_rate': ['constant', 'adaptive']  
}

mlp = MLPRegressor(max_iter=8000, solver='adam', activation='relu', hidden_layer_sizes=(100,), alpha=0.0001)

grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"\nBest Hyperparameters: {best_params}")

# Step 5: Make Predictions using the Best Model
y_pred = best_model.predict(X_test_scaled)

# Step 6: Evaluate the Model
r2 = r2_score(y_test, y_pred)
print(f'R-squared (R²): {r2}')

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')

rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE): {rmse}')

mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')


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
plt.show()

# Step 7: User Input and Prediction
feature_names = X.columns.tolist()
user_input = {}

for feature in feature_names:
    user_input[feature] = float(input(f'Enter value for {feature}: '))

user_df = pd.DataFrame([user_input])
user_pred_strength = best_model.predict(user_df)
print(f'Predicted Compressive Strength: {user_pred_strength[0]}')

# Visualize the Performance

# Fit a linear regression line
slope, intercept, _, _, _ = linregress(y_test, y_pred)
fit_line = slope * y_test + intercept
plt.plot(y_test, fit_line, '--', color='red', linewidth=2, label='Fitted Line')

r2_text = f'R-squared (R²): {r2:.3f}'


# Scatter plot for Test Set with Fitted Line
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='green', label='Test Set', alpha=0.7)
slope, intercept, _, _, _ = linregress(y_test, y_pred)
fit_line = slope * y_test + intercept
plt.plot(y_test, fit_line, '--', color='red', linewidth=2, label='Fitted Line')

equation_text = f'Fitted Equation: y = {slope:.2f}x + {intercept:.2f}'
plt.text(0.5, 0.92, equation_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

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
