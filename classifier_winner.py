import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

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

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

df_test = X_test.copy()
df_test['predicted_winner'] = y_pred
df_test['winner_probability'] = y_pred_proba[:, 1]  # Probability of being a winner
df_test['driver'] = label_encoder_driver.inverse_transform(df_test['driver_encoded'])
df_test['constructor'] = label_encoder_constructor.inverse_transform(df_test['constructor_encoded'])
df_test['circuit'] = label_encoder_circuit.inverse_transform(df_test['circuit_encoded'])

df_test['actual_winner'] = y_test.values
#
df_test['actual_winner_driver'] = df_test.apply(
    lambda row: row['driver'] if row['actual_winner'] == 1 else None, axis=1
)

predicted_winners = df_test[df_test['predicted_winner'] == 1]
actual_winners = df_test[df_test['actual_winner'] == 1]

print("\nPredicted Winners with Probabilities:")
print(predicted_winners[['driver', 'circuit', 'year', 'winner_probability']])

print("\nActual Winners:")
print(actual_winners[['driver', 'circuit', 'year']])

comparison = pd.merge(
    predicted_winners,
    actual_winners,
    on=['circuit', 'year'],
    suffixes=('_predicted', '_actual'),
    how='inner'
)
# Merge predicted and actual winners with probabilities using an inner join
comparison = pd.merge(
    predicted_winners[['driver', 'circuit', 'year', 'winner_probability']],
    actual_winners[['driver', 'circuit', 'year']],
    on=['circuit', 'year'],
    suffixes=('_predicted', '_actual'),
    how='inner'
)

print("\nComparison of Predicted vs Actual Winners with Probabilities:")
print(comparison[['driver_predicted', 'driver_actual', 'circuit', 'year', 'winner_probability']])
