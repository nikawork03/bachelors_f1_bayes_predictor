import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Load the data
csv_file = "driver_points_per_circuit_with_names.csv"
df = pd.read_csv(csv_file)

# Floor the predicted points
df["predicted_points"] = df["predicted_points"].apply(lambda x: int(x // 1))

# Convert to classification problem: Did the model correctly predict drivers scoring points?
df["actual_class"] = (df["actual_points"] > 0).astype(int)  # 1 if driver scored points, else 0
df["predicted_class"] = (df["predicted_points"] > 0).astype(int)  # 1 if predicted points > 0, else 0

# Classification Metrics
precision = precision_score(df["actual_class"], df["predicted_class"])
recall = recall_score(df["actual_class"], df["predicted_class"])
f1 = f1_score(df["actual_class"], df["predicted_class"])
accuracy = accuracy_score(df["actual_class"], df["predicted_class"])

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(df["actual_class"], df["predicted_class"])
ConfusionMatrixDisplay(cm, display_labels=["No Points", "Points"]).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("visuals_svm/confusion_matrix.png", dpi=300)
plt.show()

# Aggregate the data by driver
agg_data = df.groupby("driver").agg(
    avg_actual_points=("actual_points", "mean"),
    avg_predicted_points=("predicted_points", "mean"),
).reset_index()

# Calculate differences
agg_data["avg_difference"] = agg_data["avg_actual_points"] - agg_data["avg_predicted_points"]
agg_data["squared_avg_difference"] = agg_data["avg_difference"] ** 2

# Calculate MAE and MSE
mae = mean_absolute_error(df["actual_points"], df["predicted_points"])
mse = mean_squared_error(df["actual_points"], df["predicted_points"])

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Add MAE and MSE per driver
agg_data["mae_per_driver"] = abs(agg_data["avg_difference"])
agg_data["mse_per_driver"] = agg_data["avg_difference"] ** 2

# Plot 1: Average of actual vs predicted points per driver
plt.figure(figsize=(14, 8))
sns.barplot(
    data=agg_data.melt(
        id_vars="driver",
        value_vars=["avg_actual_points", "avg_predicted_points"],
        var_name="Point Type",
        value_name="Points",
    ),
    x="driver",
    y="Points",
    hue="Point Type",
)
plt.title("Average of Actual vs Predicted Points per Driver", fontsize=16)
plt.xlabel("Driver", fontsize=12)
plt.ylabel("Points", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.legend(title="Point Type")
plt.tight_layout()
plt.savefig("visuals_svm/avg_actual_vs_predicted_points.png", dpi=300)
plt.show()

# Plot 2: Average of difference between actual and predicted points per driver
plt.figure(figsize=(14, 8))
sns.barplot(data=agg_data, x="driver", y="avg_difference", color="lightcoral")
plt.title("Average of Difference Between Actual and Predicted Points per Driver", fontsize=16)
plt.xlabel("Driver", fontsize=12)
plt.ylabel("Difference", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.tight_layout()
plt.savefig("visuals_svm/avg_difference_points.png", dpi=300)
plt.show()

# Plot 3: Squared average difference between actual and predicted points per driver
plt.figure(figsize=(14, 8))
sns.barplot(data=agg_data, x="driver", y="squared_avg_difference", color="skyblue")
plt.title("Squared Average Difference Between Actual and Predicted Points per Driver", fontsize=16)
plt.xlabel("Driver", fontsize=12)
plt.ylabel("Squared Difference", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.tight_layout()
plt.savefig("visuals_svm/squared_avg_difference_points.png", dpi=300)
plt.show()

# Plot 4: MAE per driver
plt.figure(figsize=(14, 8))
sns.barplot(data=agg_data, x="driver", y="mae_per_driver", color="green")
plt.title("Mean Absolute Error (MAE) per Driver", fontsize=16)
plt.xlabel("Driver", fontsize=12)
plt.ylabel("MAE", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.tight_layout()
plt.savefig("visuals_svm/mae_per_driver.png", dpi=300)
plt.show()

# Plot 5: MSE per driver
plt.figure(figsize=(14, 8))
sns.barplot(data=agg_data, x="driver", y="mse_per_driver", color="purple")
plt.title("Mean Squared Error (MSE) per Driver", fontsize=16)
plt.xlabel("Driver", fontsize=12)
plt.ylabel("MSE", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.tight_layout()
plt.savefig("visuals_svm/mse_per_driver.png", dpi=300)
plt.show()
