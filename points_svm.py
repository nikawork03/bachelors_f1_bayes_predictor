import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR

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

# Assign points based on finishing position
points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
df['points'] = df['finishing_position'].map(points_system).fillna(0)

# Calculate performance metric (higher is better)
max_position = df['finishing_position'].max()
df['performance'] = max_position - df['finishing_position'] + 1

# Initialize driver form column
df['driver_form'] = 1.0

# Combine year and round to ensure continuity across years
df['unique_round'] = df.groupby(['year', 'round']).ngroup()

# Calculate driver form using a rolling window over unique rounds
df = df.sort_values(by=['driverId', 'unique_round'])
df['driver_form'] = (
    df.groupby('driverId')['performance']
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(drop=True)
)

# Drop the helper column
df = df.drop(columns=['unique_round'])

# Features and target
X = df[['starting_position', 'driverId', 'constructorId', 'year', 'circuitId', 'driver_form']].values
y = df['points'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear SVR model
linear_svr_model = LinearSVR(C=1.0, epsilon=0.5, random_state=42)
linear_svr_model.fit(X_train, y_train)

# Predictions
y_pred = linear_svr_model.predict(X_train)  # Predict on the training data

# Ensure predicted points are non-negative
y_pred = [max(0, pred) for pred in y_pred]  # Replace negative values with 0

# Preparing the training DataFrame for analysis
df_train = pd.DataFrame(X_train, columns=['starting_position', 'driverId', 'constructorId', 'year', 'circuitId', 'driver_form'])
df_train['predicted_points'] = y_pred
df_train['actual_points'] = y_train

# Deduplicate the original DataFrame for merging
df_unique = df[['driverId', 'constructorId', 'circuitId', 'driver', 'constructor', 'circuit']].drop_duplicates()

# Merge with deduplicated data to get names
df_train = df_train.merge(df_unique,
                          on=['driverId', 'constructorId', 'circuitId'],
                          how='left')

# Save predicted and actual points per circuit
df_train_circuits = df_train[['driver', 'constructor', 'circuit', 'year', 'predicted_points', 'actual_points']]

# Sort and drop duplicates
df_train_circuits = df_train_circuits.sort_values(by=['year', 'circuit', 'driver']).drop_duplicates()

# Save to CSV
df_train_circuits.to_csv('driver_points_per_circuit_with_names.csv', index=False)

print("\nPredicted and Actual Points per Circuit with Names saved to 'driver_points_per_circuit_with_names.csv':")
print(df_train_circuits.head(10))
