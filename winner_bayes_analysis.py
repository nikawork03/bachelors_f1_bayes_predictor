import pandas as pd

# Load the comparison data
comparison_file = 'winning_predictions_naive_bayes.csv'
comparison = pd.read_csv(comparison_file)

# Add a column to indicate correctness of prediction
comparison['correct_prediction'] = comparison['driver_predicted'] == comparison['driver_actual']

# Combine detailed information for display
comparison['details'] = comparison.apply(
    lambda row: f"Driver: {row['driver_predicted']}, Circuit: {row['circuit']}, Year: {row['year']}, Prob: {row['winner_probability']:.2f}",
    axis=1
)

# Display Correct vs Incorrect Predictions with Details
correct_predictions = comparison[comparison['correct_prediction']]
incorrect_predictions = comparison[~comparison['correct_prediction']]

print("\nCorrect Predictions:")
print(correct_predictions[['details']])

print("\nIncorrect Predictions:")
print(incorrect_predictions[['details']])

# Calculate and display average probabilities
avg_correct_prob = correct_predictions['winner_probability'].mean()
avg_incorrect_prob = incorrect_predictions['winner_probability'].mean()

print(f"\nAverage Probability for Correct Predictions: {avg_correct_prob:.2f}")
print(f"Average Probability for Incorrect Predictions: {avg_incorrect_prob:.2f}")

print("Summary of Correct vs Incorrect Predictions Displayed.")
