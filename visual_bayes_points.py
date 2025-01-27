import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = 'predicted_driver_points.csv'
df = pd.read_csv(file_path)

# Summing predicted and actual points per driver
driver_points_summary = df.groupby('driver')[['predicted_points', 'actual_points']].sum().reset_index()

# Average predicted and actual points per driver
driver_points_average = df.groupby('driver')[['predicted_points', 'actual_points']].mean().reset_index()

# Plotting settings
sns.set_theme(style="whitegrid")

# 1. Total Predicted vs Actual Points per Driver
plt.figure(figsize=(12, 6))
sns.barplot(data=driver_points_summary.melt(id_vars='driver', value_vars=['predicted_points', 'actual_points']),
            x='driver', y='value', hue='variable')
plt.title('Total Predicted vs Actual Points per Driver')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Total Points')
plt.xlabel('Driver')
plt.tight_layout()
plt.savefig('visuals_bayes_points/total_points_comparison.png')
plt.show()

# 2. Average Predicted vs Actual Points per Driver
plt.figure(figsize=(12, 6))
sns.barplot(data=driver_points_average.melt(id_vars='driver', value_vars=['predicted_points', 'actual_points']),
            x='driver', y='value', hue='variable')
plt.title('Average Predicted vs Actual Points per Driver')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Average Points')
plt.xlabel('Driver')
plt.tight_layout()
plt.savefig('visuals_bayes_points/average_points_comparison.png')
plt.show()

# 3. Scatter Plot: Predicted vs Actual Points (Sum)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=driver_points_summary, x='predicted_points', y='actual_points', hue='driver', s=100)
plt.plot([0, driver_points_summary['actual_points'].max()],
         [0, driver_points_summary['actual_points'].max()], 'r--', label='Ideal Prediction')
plt.title('Scatter Plot of Total Predicted vs Actual Points')
plt.xlabel('Total Predicted Points')
plt.ylabel('Total Actual Points')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('visuals_bayes_points/scatter_sum_points.png')
plt.show()

# 4. Scatter Plot: Predicted vs Actual Points (Average)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=driver_points_average, x='predicted_points', y='actual_points', hue='driver', s=100)
plt.plot([0, driver_points_average['actual_points'].max()],
         [0, driver_points_average['actual_points'].max()], 'r--', label='Ideal Prediction')
plt.title('Scatter Plot of Average Predicted vs Actual Points')
plt.xlabel('Average Predicted Points')
plt.ylabel('Average Actual Points')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('visuals_bayes_points/scatter_avg_points.png')
plt.show()

# Save summarized data
driver_points_summary.to_csv('visuals_bayes_points/driver_points_summary.csv', index=False)
driver_points_average.to_csv('visuals_bayes_points/driver_points_average.csv', index=False)

print("Plots saved as images and summaries saved as CSV files.")
