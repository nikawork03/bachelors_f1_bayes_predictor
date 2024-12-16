import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from naiveBayes import NaiveBayes

# Database setup and data loading
db_name = 'f1_database.db'
conn = sqlite3.connect(db_name)

query = """
    SELECT *
    FROM final_table
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Ensure data is sorted by year and round for proper form calculation
df = df.sort_values(by=['year', 'round'])

# Target variable (winner or not)
df['winner'] = (df['finishing_position'] == 1).astype(int)
df = df.dropna()  # Remove rows with missing values

# Calculate performance metric (higher is better)
max_position = df['finishing_position'].max()
df['performance'] = max_position - df['finishing_position'] + 1

# Initialize driver form column
df['driver_form'] = 0.0

# Combine year and round to ensure continuity across years
df['unique_round'] = df.groupby(['year', 'round']).ngroup()

# Calculate driver form using a rolling window over unique rounds
df = df.sort_values(by=['driver', 'unique_round'])
df['driver_form'] = (
    df.groupby('driver')['performance']
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(drop=True)
)

# Drop the helper column
df = df.drop(columns=['unique_round'])

# Verify the updated DataFrame
print(df[['year', 'round', 'driver', 'winner', 'driver_form']].head(10))
df.to_csv('test_driverform', index=False)
# Encode categorical features
label_encoder_driver = LabelEncoder()
label_encoder_constructor = LabelEncoder()
label_encoder_circuit = LabelEncoder()

df['driver_encoded'] = label_encoder_driver.fit_transform(df['driver'])
df['constructor_encoded'] = label_encoder_constructor.fit_transform(df['constructor'])
df['circuit_encoded'] = label_encoder_circuit.fit_transform(df['circuit'])

# Features and target
X = df[['starting_position', 'driver_encoded', 'constructor_encoded', 'year', 'circuit_encoded', 'driver_form']].values
y = df['winner'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = NaiveBayes()
model.fit(X_train, y_train)

# Predictions and probabilities
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Preparing the test DataFrame for analysis
df_test = pd.DataFrame(X_test, columns=['starting_position', 'driver_encoded', 'constructor_encoded', 'year', 'circuit_encoded', 'driver_form'])
df_test['winner_probability'] = y_pred_proba[:, 1]  # Probability of being a winner
df_test['predicted_winner'] = y_pred

df_test['driver'] = label_encoder_driver.inverse_transform(df_test['driver_encoded'].astype(int))
df_test['constructor'] = label_encoder_constructor.inverse_transform(df_test['constructor_encoded'].astype(int))
df_test['circuit'] = label_encoder_circuit.inverse_transform(df_test['circuit_encoded'].astype(int))
df_test['actual_winner'] = y_test
df_test['actual_winner_driver'] = df_test.apply(lambda row: row['driver'] if row['actual_winner'] == 1 else None, axis=1)

# Compare predictions and actual winners
predicted_winners = df_test[df_test['predicted_winner'] == 1]
actual_winners = df_test[df_test['actual_winner'] == 1]

# Save the comparison to a CSV file
comparison = pd.merge(
    predicted_winners[['driver', 'circuit', 'year', 'winner_probability', 'driver_form']],
    actual_winners[['driver', 'circuit', 'year']],
    on=['circuit', 'year'],
    suffixes=('_predicted', '_actual'),
    how='inner'
)

# Save all predictions and actual results
comparison.to_csv('winning_predictions.csv', index=False)

print("\nComparison of Predicted vs Actual Winners with Probabilities saved to 'winning_predictions.csv':")
print(comparison[['driver_predicted', 'driver_actual', 'circuit', 'year', 'winner_probability']])
