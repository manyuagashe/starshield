# ==============================================================================
# ASTEROID RISK MODEL - TRAINING SCRIPT (JSON VERSION)
#
# Instructions:
# 1. Prepare your data in a file named `real_asteroid_data.json`.
#    It MUST contain an array of objects with the required feature columns and the `risk_level` field.
# 2. Run this script from your terminal: `python train_model.py`
#
# This script will load your real data from JSON, train a model, and save the output
# to `asteroid_risk_model.joblib` and `risk_level_labels.joblib`.
# ==============================================================================

# ### 1. Setup and Imports ###
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json
import os

print("--- Section 1: Libraries imported successfully! ---")


# ### 2. Load Real Data from JSON ###
print("\n--- Section 2: Loading real asteroid data from JSON... ---")
try:
    with open('real_asteroid_data.json', 'r') as f:
        data = json.load(f)
    
    # Convert JSON data to DataFrame
    df = pd.DataFrame(data)
    print("Successfully loaded 'real_asteroid_data.json'.")
    print(f"Loaded {len(df)} records.")
    print("First 5 rows of your data:")
    print(df.head())
    
    # Validate required columns
    required_columns = ['distance_au', 'velocity_kms', 'diameter_km', 'v_infinity_kms', 
                       'is_pha', 'orbit_class', 'risk_level']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"\nERROR: Missing required columns: {missing_columns}")
        print(f"Required columns: {required_columns}")
        print(f"Found columns: {list(df.columns)}")
        exit()
        
except FileNotFoundError:
    print("\nERROR: `real_asteroid_data.json` not found.")
    print("Please create this file with your labeled training data in JSON format.")
    print("\nExpected JSON format:")
    print("""[
    {
        "distance_au": 0.018,
        "velocity_kms": 22.3,
        "diameter_km": 0.45,
        "v_infinity_kms": 18.1,
        "is_pha": true,
        "orbit_class": "APO",
        "risk_level": "High"
    },
    {
        "distance_au": 0.025,
        "velocity_kms": 15.2,
        "diameter_km": 0.32,
        "v_infinity_kms": 12.8,
        "is_pha": false,
        "orbit_class": "ATE",
        "risk_level": "Medium"
    }
]""")
    exit()
except json.JSONDecodeError as e:
    print(f"\nERROR: Invalid JSON format in 'real_asteroid_data.json': {e}")
    print("Please check your JSON file for syntax errors.")
    exit()

# ### 3. Data Preprocessing (Handling Categorical Data) ###
print("\n--- Section 3: Preprocessing data for the model... ---")

# Convert the text-based risk_level to a numerical format for the model
risk_mapping = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
df['risk_encoded'] = df['risk_level'].map(risk_mapping)

# Check for unmapped risk levels
unmapped_risks = df[df['risk_encoded'].isna()]['risk_level'].unique()
if len(unmapped_risks) > 0:
    print(f"\nWARNING: Found unmapped risk levels: {unmapped_risks}")
    print(f"Valid risk levels are: {list(risk_mapping.keys())}")
    # Remove rows with unmapped risk levels
    df = df.dropna(subset=['risk_encoded'])
    print(f"Removed rows with unmapped risk levels. Remaining records: {len(df)}")

# Convert the 'orbit_class' column into numerical format using one-hot encoding
df_processed = pd.get_dummies(df, columns=['orbit_class'], prefix='class')

print(f"\nData after one-hot encoding 'orbit_class'. Shape: {df_processed.shape}")
print("First 5 rows:")
print(df_processed.head())

# Display class distribution
print(f"\nRisk level distribution:")
for risk_level, count in df['risk_level'].value_counts().items():
    print(f"  {risk_level}: {count} ({count/len(df)*100:.1f}%)")


# ### 4. Model Training (Using All Features from Your Data) ###
print("\n--- Section 4: Training the RandomForest model on your feature set... ---")
# Define the final feature set for the model
features = [
    'distance_au', 'velocity_kms', 'diameter_km', 'v_infinity_kms', 'is_pha',
    'class_AMO', 'class_APO', 'class_ATE', 'class_IEO'
]

# Ensure all feature columns exist, even if one orbit class isn't in the data
for col in features:
    if col not in df_processed.columns:
        df_processed[col] = 0
        print(f"Added missing feature column '{col}' with default value 0")

X = df_processed[features]
y = df_processed['risk_encoded']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Features used: {features}")

# Check if we have enough data for train/test split
if len(df) < 10:
    print(f"\nWARNING: Very small dataset ({len(df)} records). Consider adding more data for better model performance.")

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
except ValueError as e:
    print(f"\nERROR during train/test split: {e}")
    print("This might be due to insufficient data for some risk levels.")
    print("Using non-stratified split instead...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Model training complete!")


# ### 5. Model Evaluation ###
print("\n--- Section 5: Evaluating the model on your data... ---")
y_pred = model.predict(X_test)

print("\nClassification Report:")
target_names = [name for name, code in sorted(risk_mapping.items(), key=lambda x: x[1]) if code in y_test.unique()]
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
for _, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Your Data')
plt.tight_layout()
plt.show()


# ### 6. Saving the Model for API Use ###
print("\n--- Section 6: Saving model and label files... ---")
model_filename = 'asteroid_risk_model.joblib'
joblib.dump(model, model_filename)

label_map = {v: k for k, v in risk_mapping.items()}
label_filename = 'risk_level_labels.joblib'
joblib.dump(label_map, label_filename)

print(f"Model saved to: {os.path.abspath(model_filename)}")
print(f"Label map saved to: {os.path.abspath(label_filename)}")

# Save training metadata
metadata = {
    'training_records': len(df),
    'features_used': features,
    'risk_mapping': risk_mapping,
    'model_type': 'RandomForestClassifier',
    'model_params': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Training metadata saved to: {os.path.abspath('model_metadata.json')}")
print("\nSCRIPT FINISHED! Your model is trained and ready to be served.")