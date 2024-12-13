import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Load and preprocess dataset
# Replace with your dataset path or loading method
data = pd.read_csv('final_table.csv')

# Encode categorical variables
label_encoder_driver = LabelEncoder()
label_encoder_constructor = LabelEncoder()
label_encoder_circuit = LabelEncoder()

data['driver_encoded'] = label_encoder_driver.fit_transform(data['driver'])
data['constructor_encoded'] = label_encoder_constructor.fit_transform(data['constructor'])
data['circuit_encoded'] = label_encoder_circuit.fit_transform(data['circuit'])

# Features and target variable
X = data[['starting_position', 'driver_encoded', 'constructor_encoded', 'year', 'circuit_encoded', 'driver_form']]
y = data['winner']  # Target: winner (0 or 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Standardize the features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Train SVM with a radial basis function (RBF) kernel
svm_model = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced', C=1)
svm_model.fit(X_train_resampled, y_train_resampled)

# Predictions and probabilities
y_pred = svm_model.predict(X_test)
y_pred_proba = svm_model.predict_proba(X_test)[:, 1]

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Prepare predictions DataFrame
df_test = X_test.copy()
df_test = pd.DataFrame(df_test, columns=['starting_position', 'driver_encoded', 'constructor_encoded', 'year', 'circuit_encoded', 'driver_form'])
df_test['winner_probability'] = y_pred_proba
df_test['predicted_winner'] = y_pred

# Decode categorical labels for clarity
valid_labels = set(range(len(label_encoder_driver.classes_)))
df_test.loc[~df_test['driver_encoded'].isin(valid_labels), 'driver_encoded'] = -1
df_test['driver'] = df_test['driver_encoded'].apply(lambda x: 'Unknown' if x == -1 else label_encoder_driver.inverse_transform([x])[0])
df_test['constructor'] = label_encoder_constructor.inverse_transform(df_test['constructor_encoded'].astype(int))
df_test['circuit'] = label_encoder_circuit.inverse_transform(df_test['circuit_encoded'].astype(int))

# Actual winners for comparison
df_test['actual_winner'] = y_test.reset_index(drop=True)

# Display results
predicted_winners = df_test[df_test['predicted_winner'] == 1]
actual_winners = df_test[df_test['actual_winner'] == 1]

print("\nPredicted Winners with Probabilities:")
print(predicted_winners[['driver', 'circuit', 'year', 'winner_probability', 'driver_form']])

print("\nActual Winners:")
print(actual_winners[['driver', 'circuit', 'year']])

# Compare predicted and actual winners
comparison = pd.merge(
    predicted_winners[['driver', 'circuit', 'year', 'winner_probability']],
    actual_winners[['driver', 'circuit', 'year']],
    on=['circuit', 'year'],
    suffixes=('_predicted', '_actual'),
    how='inner'
)

print("\nComparison of Predicted vs Actual Winners with Probabilities:")
print(comparison[['driver_predicted', 'driver_actual', 'circuit', 'year', 'winner_probability']])
