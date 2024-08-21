import pandas as pd
import matplotlib.pyplot as plt

# Correct file path
file_path = 'C:/Users/olaol/Downloads/heat-of-hydration-data.xlsx'

# Read data from XLSX file
data = pd.read_excel(file_path)

# Function to clean data, remove inconsistencies, and plot the graph
def clean_and_plot(data_subset, title):
    # Convert 'Time' to datetime format
    data_subset['Time'] = pd.to_datetime(data_subset['Time'], format='%H:%M:%S', errors='coerce')

    # Drop rows with invalid times
    data_subset.dropna(subset=['Time'], inplace=True)

    # Calculate cumulative hours
    data_subset['Cumulative Hours'] = (data_subset['Time'] - data_subset['Time'].iloc[0]).dt.total_seconds() / 3600

    # Plot the data
    plt.plot(data_subset['Cumulative Hours'], data_subset['Temp A'], marker='o', linestyle='-', color='b', label='Temp A')
    plt.plot(data_subset['Cumulative Hours'], data_subset['Temp B'], marker='o', linestyle='-', color='r', label='Temp B')
    plt.plot(data_subset['Cumulative Hours'], data_subset['Temp C'], marker='o', linestyle='-', color='g', label='Temp C')
    plt.plot(data_subset['Cumulative Hours'], data_subset['Temp D'], marker='o', linestyle='-', color='y', label='Temp D')

    # Add grid, title, and labels
    plt.title(f'Temperature trend for {title}', fontsize=14)
    plt.xlabel('Time (Hours)', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)

    # Set x-axis limits and intervals
    max_hours = int(data_subset['Cumulative Hours'].max()) + 1
    plt.xlim(0, max_hours)  # Set the limit to include the entire range of data
    plt.xticks(range(max_hours))  # Set x-axis ticks to be every hour

    # Set y-axis to have equal intervals (e.g., 0 to 30 with 5-degree increments)
    plt.yticks(range(0, 31, 5))

    # Explicitly set the y-axis limit to include all temperatures
    plt.ylim(bottom=0)  # Adjust this as needed to include your data

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

# Plot all graphs on a single page
plt.figure(figsize=(20, 24))

# Define slices for each graph
slices = [
    (2, 1778, 'Day 1 of M15 (1)'),
    (1779, 3459, 'Day 1 of M20 (2)'),
    (3460, 3645, 'Day 1 of M25 (3)'),
    (3646, 7327, 'Day 1 of M30 (4)'),
    (7328, 11510, 'Day 3 of M15 (5)'),
    (11511, 17494, 'Day 3 of M20 (6)'),
    (17495, 19496, 'Day 3 of M25 (7)'),
    (19497, 26226, 'Day 3 of M30 (8)')
]

# Plot each slice in a subplot
for i, (start, end, title) in enumerate(slices, 1):
    plt.subplot(4, 2, i)
    data_subset = data.iloc[start:end].copy()
    clean_and_plot(data_subset, title)

plt.show()

# range of temp
# Calculate the minimum and maximum for each temperature column
min_temps = data[['Temp A', 'Temp B', 'Temp C', 'Temp D']].min()
max_temps = data[['Temp A', 'Temp B', 'Temp C', 'Temp D']].max()

# Create a DataFrame for easy plotting
temp_range_df = pd.DataFrame({'Min Temperature': min_temps, 'Max Temperature': max_temps})

# Plot the range as a bar chart
ax = temp_range_df.plot(kind='bar', figsize=(12, 8), color=['skyblue', 'orange'])

# Annotate the bars with the actual values
for i in ax.containers:
    ax.bar_label(i, label_type='edge', fontsize=12)  # Increased label font size

# Set the title and labels with increased font size
plt.title('Temperature Range for Each Column', fontsize=16)
plt.xlabel('Temperature Columns', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)

# Increase the size of the tick labels on both axes
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

# Set the y-axis limits and ticks to match the range of temperatures
plt.ylim(bottom=0)  # Adjust the y-axis to start from 0
plt.yticks(list(range(0, int(max(max_temps)) + 5, 5)))  # Setting y-axis ticks with a step of 5

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust the layout and show the plot
plt.tight_layout()
plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt

# # Correct file path
# file_path = 'C:/Users/olaol/Downloads/heat-of-hydration-data.xlsx'

# # Read data from XLSX file
# data = pd.read_excel(file_path)

# # Print column names to debug
# print("Columns in the XLSX file:", data.columns)

# # Print first few rows to inspect data
# print(data.head())

# # Function to clean data, remove inconsistencies, and plot the graph
# def clean_and_plot(data_subset, title):
#     # Convert 'Time' to datetime format
#     data_subset['Time'] = pd.to_datetime(data_subset['Time'], format='%H:%M:%S', errors='coerce')

#     # Drop rows with invalid times
#     data_subset.dropna(subset=['Time'], inplace=True)

#     # Calculate cumulative hours
#     data_subset['Cumulative Hours'] = (data_subset['Time'] - data_subset['Time'].iloc[0]).dt.total_seconds() / 3600

#     # Plot the data
#     plt.figure(figsize=(12, 6))
#     plt.plot(data_subset['Cumulative Hours'], data_subset['Temp A'], marker='o', linestyle='-', color='b', label='Temp A')
#     plt.plot(data_subset['Cumulative Hours'], data_subset['Temp B'], marker='o', linestyle='-', color='r', label='Temp B')
#     plt.plot(data_subset['Cumulative Hours'], data_subset['Temp C'], marker='o', linestyle='-', color='g', label='Temp C')
#     plt.plot(data_subset['Cumulative Hours'], data_subset['Temp D'], marker='o', linestyle='-', color='y', label='Temp D')

#     # Add grid, title, and labels
#     plt.title(f'Temperature trend for {title}', fontsize=14)
#     plt.xlabel('Time (Hours)', fontsize=12)
#     plt.ylabel('Temperature (°C)', fontsize=12)

#     # Set x-axis limits and intervals
#     max_hours = int(data_subset['Cumulative Hours'].max()) + 1
#     plt.xlim(0, max_hours)  # Set the limit to include the entire range of data
#     plt.xticks(range(max_hours))  # Set x-axis ticks to be every hour

#     # Set y-axis to have equal intervals (e.g., 0 to 30 with 5-degree increments)
#     plt.yticks(range(0, 31, 5))

#     # Explicitly set the y-axis limit to include all temperatures
#     plt.ylim(bottom=0)  # Adjust this as needed to include your data

#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend()
#     plt.tight_layout()

# # Plot all graphs on a single page
# plt.figure(figsize=(20, 24))

# # Define slices for each graph
# slices = [
#     (2, 1778, 'Day 1 of M15 (1)'),
#     (1779, 3459, 'Day 1 of M20 (2)'),
#     (3460, 3645, 'Day 1 of M25 (3)'),
#     (3646, 7327, 'Day 1 of M30 (4)'),
#     (7328, 11510, 'Day 3 of M15 (5)'),
#     (11511, 17494, 'Day 3 of M20 (6)'),
#     (17495, 19496, 'Day 3 of M25 (7)'),
#     (19497, 26226, 'Day 3 of M30 (8)')
# ]

# # Plot each slice in a subplot
# for i, (start, end, title) in enumerate(slices, 1):
#     plt.subplot(4, 2, i)
#     data_subset = data.iloc[start:end].copy()
#     clean_and_plot(data_subset, title)

# plt.show()



# Code For Seperate figures

# import pandas as pd
# import matplotlib.pyplot as plt

# # Correct file path
# file_path = 'C:/Users/olaol/Downloads/heat-of-hydration-data.xlsx'

# # Read data from XLSX file
# data = pd.read_excel(file_path)

# # Print column names to debug
# print("Columns in the XLSX file:", data.columns)

# # Print first few rows to inspect data
# print(data.head())

# # Function to clean data, remove inconsistencies, and plot the graph
# def clean_and_plot(data_subset, title):
#     # Convert 'Time' to datetime format
#     data_subset['Time'] = pd.to_datetime(data_subset['Time'], format='%H:%M:%S', errors='coerce')

#     # Drop rows with invalid times
#     data_subset.dropna(subset=['Time'], inplace=True)

#     # Calculate cumulative hours
#     data_subset['Cumulative Hours'] = (data_subset['Time'] - data_subset['Time'].iloc[0]).dt.total_seconds() / 3600

#     # Plot the data
#     plt.figure(figsize=(12, 6))
#     plt.plot(data_subset['Cumulative Hours'], data_subset['Temp A'], marker='o', linestyle='-', color='b', label='Temp A')
#     plt.plot(data_subset['Cumulative Hours'], data_subset['Temp B'], marker='o', linestyle='-', color='r', label='Temp B')
#     plt.plot(data_subset['Cumulative Hours'], data_subset['Temp C'], marker='o', linestyle='-', color='g', label='Temp C')
#     plt.plot(data_subset['Cumulative Hours'], data_subset['Temp D'], marker='o', linestyle='-', color='y', label='Temp D')

#     # Add grid, title, and labels
#     plt.title(f'Temperature trend for {title}', fontsize=14)
#     plt.xlabel('Time (Hours)', fontsize=12)
#     plt.ylabel('Temperature (°C)', fontsize=12)

#     # Set x-axis limits and intervals
#     max_hours = int(data_subset['Cumulative Hours'].max()) + 1
#     plt.xlim(0, max_hours)  # Set the limit to include the entire range of data
#     plt.xticks(range(max_hours))  # Set x-axis ticks to be every hour

#     # Set y-axis to have equal intervals (e.g., 0 to 30 with 5-degree increments)
#     plt.yticks(range(0, 31, 5))

#     # Explicitly set the y-axis limit to include all temperatures
#     plt.ylim(bottom=0)  # Adjust this as needed to include your data

#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend()
#     plt.tight_layout()

# # Plot all graphs on a single page
# plt.figure(figsize=(20, 24))

# # Define slices for each graph
# slices = [
#     (2, 1778, 'Day 1 of M15'),
#     (1779, 3459, 'Day 1 of M20'),
#     (3460, 3645, 'Day 1 of M25'),
#     (3646, 7327, 'Day 1 of M30'),
#     (7328, 11510, 'Day 3 of M15'),
#     (11511, 17494, 'Day 3 of M20'),
#     (17495, 19496, 'Day 3 of M25'),
#     (19497, 26226, 'Day 3 of M30')
# ]

# # Plot each slice in a subplot
# for i, (start, end, title) in enumerate(slices, 1):
#     plt.subplot(4, 2, i)
#     data_subset = data.iloc[start:end].copy()
#     clean_and_plot(data_subset, title)

# plt.show()






# import pandas as pd
# import matplotlib.pyplot as plt

# # Correct file path
# file_path = 'C:/Users/olaol/Downloads/heat-of-hydration-data.xlsx'

# # Read data from XLXS file
# data = pd.read_csv(file_path)

# # Print column names to debug
# print("Columns in the XLSX file:", data.columns)

# # Print first few rows to inspect data
# print(data.head())

# #1. Slice the data from row 2 to row 1779, plot the chart name it Day 1 of M15,
# #2. Slice the data from row 1779 to row 3460, plot the chart name it Day 1 of M20,
# #3. Slice the data from row 3460 to row 3646, plot the chart name it Day 1 of M25,
# #4. Slice the data from row 3646 to row 7328, plot the chart name it Day 1 of M30,
# #5. Slice the data from row 7328 to row 11511, plot the chart name it Day 3 of M15,
# #6. Slice the data from row 11511 to row 17495, plot the chart name it Day 3 of M20,
# #7. Slice the data from row 17495 to row 19497, plot the chart name it Day 3 of M25,
# #8. Slice the data from row 19497 to row 26227, plot the chart name it Day 3 of M30,
# data_subset = data.iloc[3645:7327].copy()

# # Convert 'Time' to datetime format
# data_subset['Time'] = pd.to_datetime(data_subset['Time'], format='%H:%M:%S', errors='coerce')

# # Calculate cumulative hours
# data_subset['Cumulative Hours'] = (data_subset['Time'] - data_subset['Time'].iloc[0]).dt.total_seconds() / 3600

# # Plot the data with enhancements
# plt.figure(figsize=(12, 6))
# plt.plot(data_subset['Cumulative Hours'], data_subset['Temp B'], marker='o', linestyle='-', color='r', label='Temperature')

# # Add grid, title, and labels
# plt.title('Temperature trend for Day 1 of Grade M30 concrete', fontsize=14)
# plt.xlabel('Time (Hours)', fontsize=12)
# plt.ylabel('Temperature (°C)', fontsize=12)

# # Set x-axis limits and intervals
# max_hours = int(data_subset['Cumulative Hours'].max()) + 1
# plt.xlim(0, max_hours)  # Set the limit to include the entire range of data
# plt.xticks(range(max_hours))  # Set x-axis ticks to be every hour

# # Set y-axis to have equal intervals (e.g., 0 to 30 with 5-degree increments)
# plt.yticks(range(0, 31, 5))

# # Explicitly set the y-axis limit to include all temperatures
# plt.ylim(bottom=0)  # Adjust this as needed to include your data

# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend()
# plt.tight_layout()

# # Show the plot
# plt.show()


# # Provide the full path to the CSV file if it's not in the same directory
# file_path = 'C:/Users/olaol/Downloads/heat-of-hydration-data.csv'

# # Read data from CSV file
# data = pd.read_csv(file_path)

# # Print column names to debug
# print("Columns in the CSV file:", data.columns)

# import numpy as np

# # Define the values for each concrete grade
# M15_day1 = [2.6667, 2.4889, 2.5778, 2.5098]
# M15_day3 = [6.4000, 6.2222, 6.0444, 6.4000]
# M20_day1 = [3.3778, 3.5556, 3.5556, 3.4667]
# M20_day3 = [8.0889, 7.9111, 9.0667, 8.8889]
# M25_day1 = [4.2667, 4.2667, 4.6222, 4.1778]
# M25_day3 = [10.1333, 10.4889, 10.6667, 10.8444]
# M30_day1 = [4.4706, 5.3333, 4.9778, 5.1556]
# M30_day3 = [14.1333, 11.2941, 13.4222, 12.5333]

# # Calculate the standard deviation for each set
# std_M15_1 = np.std(M15_day1)
# std_M15_3 = np.std(M15_day3)
# std_M20_1 = np.std(M20_day1)
# std_M20_3 = np.std(M20_day3)
# std_M25_1 = np.std(M25_day1)
# std_M25_3 = np.std(M25_day3)
# std_M30_1 = np.std(M30_day1)
# std_M30_3 = np.std(M30_day3)

# # Print the results
# print(f"Standard Deviation for M15 day 1: {std_M15_1:.4f}")
# print(f"Standard Deviation for M15 day 3: {std_M15_3:.4f}")
# print(f"Standard Deviation for M20 day 1: {std_M20_1:.4f}")
# print(f"Standard Deviation for M20 day 3: {std_M20_3:.4f}")
# print(f"Standard Deviation for M25 day 1: {std_M25_1:.4f}")
# print(f"Standard Deviation for M25 day 3: {std_M25_3:.4f}")
# print(f"Standard Deviation for M30 day 1: {std_M30_1:.4f}")
# print(f"Standard Deviation for M30 day 3: {std_M30_3:.4f}")

# import matplotlib.pyplot as plt

# # Sand data
# sieve_sizes_sand = [8, 4, 2.36, 1, 0.50, 0.30, 0.25, 0.15, 0.075]
# percent_passing_sand = [92.4, 85.0, 40.8, 14.6, 4.3, 0, 0, 0, 0]

# # Coarse Aggregate data
# sieve_sizes_coarse = [37.5, 20, 10]
# percent_passing_coarse = [100, 68.7, 0 ]

# # Plotting both PSD curves
# plt.figure(figsize=(12, 8))

# # Plot for sand
# plt.plot(sieve_sizes_sand, percent_passing_sand, marker='o', linestyle='-', color='b', label='Sand')

# # Plot for coarse aggregate
# plt.plot(sieve_sizes_coarse, percent_passing_coarse, marker='o', linestyle='-', color='r', label='Granite')

# plt.xscale('log')  # Log scale for sieve sizes
# plt.xlabel('Sieve Size (mm)', fontsize=14)
# plt.ylabel('Cumulative Percent Passing (%)', fontsize=14)
# plt.title('Particle Size Distribution Curves', fontsize=16)
# plt.grid(True, which="both", ls="--")
# plt.gca().invert_xaxis()  # Invert x-axis for better visualization
# plt.legend()
# plt.show()
