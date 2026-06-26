from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'Dataset' / 'loan_prediction.csv'
MODEL_PATH = ROOT / 'Flask' / 'model.pkl'
SCALER_PATH = ROOT / 'Flask' / 'scaler.pkl'

if not DATA_PATH.exists():
    raise FileNotFoundError(f'Dataset file not found: {DATA_PATH}')

print(f'Loading data from {DATA_PATH}')
df = pd.read_csv(DATA_PATH)

# Map categories to numeric values used by the Flask app.
map_values = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Education': {'Graduate': 0, 'Not Graduate': 1},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Loan_Status': {'Y': 1, 'N': 0},
}

# Normalize Dependents and fill missing values.
df['Dependents'] = df['Dependents'].replace({'3+': 3})
for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode().iloc[0])

for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

for col, mapping in map_values.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

feature_columns = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Property_Area'
]

if df['Loan_Status'].isna().any():
    df = df.dropna(subset=['Loan_Status'])

X = df[feature_columns].astype(float)
y = df['Loan_Status'].astype(int)

print('Training samples:', len(X))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42, n_estimators=150)
model.fit(X_train_scaled, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
print(f'Train accuracy: {train_acc:.4f}')
print(f'Test accuracy: {test_acc:.4f}')

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print(f'Saved model to {MODEL_PATH}')
print(f'Saved scaler to {SCALER_PATH}')
