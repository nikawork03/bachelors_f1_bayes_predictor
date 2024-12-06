import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

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

y_pred_proba = model.predict_proba(X_test)

df_test = X_test.copy()
df_test['winner_probability'] = y_pred_proba[:, 1]  # Probability of being a winner
df_test['driver'] = label_encoder_driver.inverse_transform(df_test['driver_encoded'])
df_test['constructor'] = label_encoder_constructor.inverse_transform(df_test['constructor_encoded'])
df_test['circuit'] = label_encoder_circuit.inverse_transform(df_test['circuit_encoded'])
df_test['predicted_winner'] = (df_test['winner_probability'] > 0.5).astype(int)

df_test['actual_winner'] = y_test.values
df_test['actual_winner_driver'] = df_test.apply(lambda row: row['driver'] if row['actual_winner'] == 1 else None, axis=1)

top3_predicted = (
    df_test.sort_values(by=['circuit', 'year', 'winner_probability'], ascending=[True, True, False])
    .groupby(['circuit', 'year'])
    .head(3)
)

actual_winners = df_test[df_test['actual_winner'] == 1]

comparison_top3 = pd.merge(
    top3_predicted[['driver', 'circuit', 'year', 'winner_probability']],
    actual_winners[['driver', 'circuit', 'year']],
    on=['circuit', 'year'],
    suffixes=('_predicted', '_actual'),
    how='inner'
)

comparison_top3['is_actual_winner_in_top3'] = comparison_top3['driver_actual'] == comparison_top3['driver_predicted']

comparison_summary = comparison_top3.groupby(['circuit', 'year']).agg(
    actual_winner=('driver_actual', 'first'),
    predicted_top3=('driver_predicted', list),
    probabilities=('winner_probability', list),
    is_actual_winner_in_top3=('is_actual_winner_in_top3', 'any')
).reset_index()

comparison_summary.to_csv('top3_comparison_summary.csv', index=False)
print("\nTop 3 Predicted vs Actual Winners Comparison saved to 'top3_comparison_summary.csv'.")

not_in_top3 = comparison_summary[comparison_summary['is_actual_winner_in_top3'] == False]
not_in_top3.to_csv('not_in_top3.csv', index=False)
print("\nRaces Where the Actual Winner Was Not in the Top 3 saved to 'not_in_top3.csv'.")
