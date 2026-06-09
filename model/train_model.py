import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os

# ── 1. LOAD DATA ──────────────────────────────────────────────
df = pd.read_csv('data/wifi_fingerprints.csv')
print(f"Dataset shape: {df.shape}")
print(f"Locations: {df['location'].unique()}")

# ── 2. CLEAN DATA ─────────────────────────────────────────────
# Drop timestamp column - not useful for ML
df = df.drop(columns=['timestamp'])

# Fill missing RSSI values with -100 (means network not visible)
df = df.fillna(-100)

print(f"\nAfter cleaning: {df.shape}")

# ── 3. SPLIT FEATURES AND LABEL ───────────────────────────────
# X = WiFi signal columns (what the model learns from)
# y = location column (what the model predicts)
X = df.drop(columns=['location'])
y = df['location']

print(f"\nFeatures (X): {X.shape}")
print(f"Labels (y): {y.unique()}")

# ── 4. ENCODE LABELS ──────────────────────────────────────────
# ML models need numbers not text
# my_desk → 0, main_door → 1, rahul_desk → 2
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nEncoded labels: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── 5. TRAIN TEST SPLIT ───────────────────────────────────────
# 80% training, 20% testing
# Since we have small data, we use test_size=0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42 ,stratify=y_encoded
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ── 6. TRAIN 3 MODELS ─────────────────────────────────────────
models = {
    'KNN':           KNeighborsClassifier(n_neighbors=2),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':           SVC(kernel='rbf', random_state=42)
}

results = {}

print("\n" + "="*45)
print("       MODEL TRAINING RESULTS")
print("="*45)

best_model = None
best_accuracy = 0
best_name = ""

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Score
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy * 100:.1f}%")
    print(f"  Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
    
    # Track best
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

# ── 7. SUMMARY ────────────────────────────────────────────────
print("="*45)
print("       SUMMARY")
print("="*45)
for name, acc in results.items():
    bar = "█" * int(acc * 20)
    print(f"  {name:<15} {acc*100:.1f}%  {bar}")

print(f"\n  Best Model: {best_name} ({best_accuracy*100:.1f}%)")

# ── 8. SAVE BEST MODEL ────────────────────────────────────────
os.makedirs('model', exist_ok=True)

with open('model/pathfi_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('model/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

with open('model/feature_columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print(f"\n  Model saved to model/pathfi_model.pkl")
print(f"  Label encoder saved to model/label_encoder.pkl")
print(f"  Feature columns saved to model/feature_columns.pkl")
print("\nPathFi model is ready!")