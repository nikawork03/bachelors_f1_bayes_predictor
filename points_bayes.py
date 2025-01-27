import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from naiveBayesClass import NaiveBayes

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

# Features and target
X = df[['starting_position', 'driverId', 'constructorId', 'year', 'circuitId', 'driver_form']].values
y = df['points'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train Naive Bayes model
model = NaiveBayes()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Preparing the test DataFrame for analysis
df_test = pd.DataFrame(X_test, columns=['starting_position', 'driverId', 'constructorId', 'year', 'circuitId', 'driver_form'])
df_test['predicted_points'] = y_pred
df_test['actual_points'] = y_test

# Map IDs back to names
label_encoder_driver = df[['driver', 'driverId']].drop_duplicates().set_index('driverId').to_dict()['driver']
label_encoder_constructor = df[['constructor', 'constructorId']].drop_duplicates().set_index('constructorId').to_dict()['constructor']
label_encoder_circuit = df[['circuit', 'circuitId']].drop_duplicates().set_index('circuitId').to_dict()['circuit']

df_test['driver'] = df_test['driverId'].map(label_encoder_driver)
df_test['constructor'] = df_test['constructorId'].map(label_encoder_constructor)
df_test['circuit'] = df_test['circuitId'].map(label_encoder_circuit)

# Save predictions and actual points to CSV
df_test = df_test[['driver', 'constructor', 'circuit', 'year', 'predicted_points', 'actual_points']]
df_test.to_csv('predicted_driver_points.csv', index=False)

print("\nPredicted and Actual Points saved to 'predicted_driver_points.csv':")
print(df_test.head(10))
