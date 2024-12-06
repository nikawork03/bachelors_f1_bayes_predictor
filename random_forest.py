import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

db_name = 'f1_database.db'
conn = sqlite3.connect(db_name)

query = """
    SELECT *
    FROM final_table 
"""

df = pd.read_sql_query(query, conn)
conn.close()

df['winner'] = (df['finishing_position'] == 1).astype(int)  # 1 if winner, 0 otherwise
df = df.dropna()  # Remove rows with missing values

label_encoder_driver = LabelEncoder()
label_encoder_constructor = LabelEncoder()
label_encoder_circuit = LabelEncoder()

df['driver_encoded'] = label_encoder_driver.fit_transform(df['driver'])
df['constructor_encoded'] = label_encoder_constructor.fit_transform(df['constructor'])
df['circuit_encoded'] = label_encoder_circuit.fit_transform(df['circuit'])

X = df[['starting_position', 'driver_encoded', 'constructor_encoded', 'year', 'circuit_encoded']]
y = df['winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]

df_test = X_test.copy()
df_test['prob_win'] = y_prob  # Probability of winning
df_test['driver'] = label_encoder_driver.inverse_transform(df_test['driver_encoded'])
df_test['constructor'] = label_encoder_constructor.inverse_transform(df_test['constructor_encoded'])
df_test['circuit'] = label_encoder_circuit.inverse_transform(df_test['circuit_encoded'])

df_test['year_circuit'] = df_test['year'].astype(str) + "_" + df_test['circuit']
predicted_winner_per_race = df_test.loc[df_test.groupby('year_circuit')['prob_win'].idxmax()]

predicted_winner_per_race['actual_winner'] = y_test.loc[predicted_winner_per_race.index].values

predicted_winner_per_race['is_correct'] = predicted_winner_per_race['actual_winner'] == 1

total_races = len(predicted_winner_per_race)
correct_predictions = predicted_winner_per_race['is_correct'].sum()
incorrect_predictions = total_races - correct_predictions
accuracy = correct_predictions / total_races

print(f"\nTotal Races: {total_races}")
print(f"Correct Predictions (Highest Probability Driver Wins): {correct_predictions}")
print(f"Incorrect Predictions: {incorrect_predictions}")
print(f"Accuracy: {accuracy:.2f}")

print("\nPredicted Winners (One per Race):")
print(predicted_winner_per_race[['driver', 'circuit', 'year', 'prob_win', 'is_correct']])