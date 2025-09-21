# ==============================================================================
# ASTEROID RISK MODEL - TRAINING SCRIPT (DUAL JSON VERSION)
#
# Instructions:
# 1. Prepare your data in two files:
#    - `final_data_false.json` (contains asteroids with is_pha=false)
#    - `final_data_true.json` (contains asteroids with is_pha=true)
# 2. Run this script from your terminal: `python train_model.py`
#
# This script will load both files, mix them, split into train/test sets,
# and save the model to `asteroid_risk_model.joblib` and `risk_level_labels.joblib`.
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


# ### 2. Load and Combine Data from Both JSON Files ###
print("\n--- Section 2: Loading and combining asteroid data from JSON files... ---")
try:
    # Load data with is_pha=false
    with open('final_data_false.json', 'r') as f:
        data_false = json.load(f)
    print(f"Loaded {len(data_false)} records with is_pha=false")
    
    # Load data with is_pha=true
    with open('final_data_true.json', 'r') as f:
        data_true = json.load(f)
    print(f"Loaded {len(data_true)} records with is_pha=true")
    
    # Combine both datasets
    all_data = data_false + data_true
    print(f"\nTotal combined records: {len(all_data)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Shuffle the data to ensure good mixing
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("Data shuffled successfully")
    
    # Display PHA distribution
    print(f"\nPHA distribution:")
    pha_counts = df['is_pha'].value_counts()
    for pha_status, count in pha_counts.items():
        print(f"  is_pha={pha_status}: {count} ({count/len(df)*100:.1f}%)")
    
    # Validate required columns
    required_columns = ['distance_au', 'velocity_kms', 'diameter_km', 'v_infinity_kms', 
                       'is_pha', 'orbit_class', 'risk_level']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"\nERROR: Missing required columns: {missing_columns}")
        print(f"Required columns: {required_columns}")
        print(f"Found columns: {list(df.columns)}")
        exit()
    
    print("\nFirst 5 rows of combined data:")
    print(df.head())
        
except FileNotFoundError as e:
    print(f"\nERROR: Required file not found: {e}")
    print("Please ensure you have both files:")
    print("  - final_data_false.json")
    print("  - final_data_true.json")
    exit()
except json.JSONDecodeError as e:
    print(f"\nERROR: Invalid JSON format: {e}")
    print("Please check your JSON files for syntax errors.")
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

# Display class distribution
print(f"\nRisk level distribution:")
for risk_level, count in df['risk_level'].value_counts().items():
    print(f"  {risk_level}: {count} ({count/len(df)*100:.1f}%)")


# ### 4. Train/Test Split and Model Training ###
print("\n--- Section 4: Splitting data and training the RandomForest model... ---")

# First split the data into train and test sets
# Using stratify to maintain the distribution of risk levels and is_pha
stratify_column = df['risk_level'].astype(str) + '_' + df['is_pha'].astype(str)
train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42,
    stratify=stratify_column
)

print(f"\nData split:")
print(f"  Training set: {len(train_df)} records ({len(train_df)/len(df)*100:.1f}%)")
print(f"  Test set: {len(test_df)} records ({len(test_df)/len(df)*100:.1f}%)")

# Verify distribution is maintained
print(f"\nPHA distribution in splits:")
print("  Training set:")
for pha_status, count in train_df['is_pha'].value_counts().items():
    print(f"    is_pha={pha_status}: {count} ({count/len(train_df)*100:.1f}%)")
print("  Test set:")
for pha_status, count in test_df['is_pha'].value_counts().items():
    print(f"    is_pha={pha_status}: {count} ({count/len(test_df)*100:.1f}%)")

# Save the split data for future use (optional)
with open('real_asteroid_data_train.json', 'w') as f:
    json.dump(train_df.drop('risk_encoded', axis=1).to_dict('records'), f, indent=2)
with open('real_asteroid_data_test.json', 'w') as f:
    json.dump(test_df.drop('risk_encoded', axis=1).to_dict('records'), f, indent=2)
print("\nSaved split data to real_asteroid_data_train.json and real_asteroid_data_test.json")

# Process both training and test data with same preprocessing
def preprocess_data(data_df):
    """Apply preprocessing to training or test data"""
    # Convert the text-based risk_level to a numerical format
    data_df = data_df.copy()
    if 'risk_encoded' not in data_df.columns:
        data_df['risk_encoded'] = data_df['risk_level'].map(risk_mapping)
    
    # One-hot encode orbit_class
    data_processed = pd.get_dummies(data_df, columns=['orbit_class'], prefix='class')
    
    return data_processed

# Preprocess both datasets
train_processed = preprocess_data(train_df)
test_processed = preprocess_data(test_df)

# Define the final feature set for the model
features = [
    'distance_au', 'velocity_kms', 'diameter_km', 'v_infinity_kms', 'is_pha',
    'class_AMO', 'class_APO', 'class_ATE', 'class_IEO'
]

# Ensure all feature columns exist in both datasets
for dataset in [train_processed, test_processed]:
    for col in features:
        if col not in dataset.columns:
            dataset[col] = 0
            print(f"Added missing feature column '{col}' with default value 0")

# Prepare training data
X_train = train_processed[features]
y_train = train_processed['risk_encoded']

# Prepare test data
X_test = test_processed[features]
y_test = test_processed['risk_encoded']

print(f"\nTraining set - Features: {X_train.shape}, Labels: {y_train.shape}")
print(f"Test set - Features: {X_test.shape}, Labels: {y_test.shape}")
print(f"Features used: {features}")

# Train the model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Model training complete!")

# ### 4.5. PHA Binary Classification Model ###
print("\n--- Section 4.5: Training PHA Binary Classification Model... ---")

# Define features for PHA prediction (excluding is_pha itself)
pha_features = [
    'distance_au', 'velocity_kms', 'diameter_km', 'v_infinity_kms',
    'class_AMO', 'class_APO', 'class_ATE', 'class_IEO'
]

# Ensure all PHA feature columns exist in both datasets
for dataset in [train_processed, test_processed]:
    for col in pha_features:
        if col not in dataset.columns:
            dataset[col] = 0
            print(f"Added missing PHA feature column '{col}' with default value 0")

# Prepare PHA training data
X_train_pha = train_processed[pha_features]
y_train_pha = train_processed['is_pha']

# Prepare PHA test data
X_test_pha = test_processed[pha_features]
y_test_pha = test_processed['is_pha']

print(f"\nPHA Training set - Features: {X_train_pha.shape}, Labels: {y_train_pha.shape}")
print(f"PHA Test set - Features: {X_test_pha.shape}, Labels: {y_test_pha.shape}")
print(f"PHA Features used: {pha_features}")

# Train the PHA binary classifier
pha_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
pha_model.fit(X_train_pha, y_train_pha)

print("PHA binary classification model training complete!")

# Evaluate PHA model
y_pred_pha = pha_model.predict(X_test_pha)
pha_accuracy = (y_test_pha == y_pred_pha).mean()
print(f"PHA Model Accuracy: {pha_accuracy:.2%}")

# PHA model classification report
print("\nPHA Model Classification Report:")
print(classification_report(y_test_pha, y_pred_pha, target_names=['Non-PHA', 'PHA'], zero_division=0))

# PHA Feature importance
pha_feature_importance = pd.DataFrame({
    'feature': pha_features,
    'importance': pha_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nPHA Model Feature Importance:")
for _, row in pha_feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")


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

# Analyze predictions by is_pha status
print("\nModel Performance by PHA Status:")
test_results = test_df.copy()
test_results['predicted_risk'] = [risk_mapping.get(pred, pred) for pred in y_pred]
test_results['actual_risk_encoded'] = y_test.values

# Performance for is_pha=True
pha_true_mask = test_results['is_pha'] == True
pha_true_accuracy = (test_results[pha_true_mask]['predicted_risk'] == test_results[pha_true_mask]['actual_risk_encoded']).mean()
print(f"  Accuracy for PHA asteroids (is_pha=True): {pha_true_accuracy:.2%}")

# Performance for is_pha=False
pha_false_mask = test_results['is_pha'] == False
pha_false_accuracy = (test_results[pha_false_mask]['predicted_risk'] == test_results[pha_false_mask]['actual_risk_encoded']).mean()
print(f"  Accuracy for non-PHA asteroids (is_pha=False): {pha_false_accuracy:.2%}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Combined Data')
plt.tight_layout()
plt.show()


# ### 6. Saving the Model for API Use ###
print("\n--- Section 6: Saving model and label files... ---")
model_filename = 'asteroid_risk_model.joblib'
joblib.dump(model, model_filename)

label_map = {v: k for k, v in risk_mapping.items()}
label_filename = 'risk_level_labels.joblib'
joblib.dump(label_map, label_filename)

# Save PHA binary classification model
pha_model_filename = 'asteroid_pha_model.joblib'
joblib.dump(pha_model, pha_model_filename)

print(f"Model saved to: {os.path.abspath(model_filename)}")
print(f"Label map saved to: {os.path.abspath(label_filename)}")
print(f"PHA model saved to: {os.path.abspath(pha_model_filename)}")

# Save training metadata
metadata = {
    'training_records': len(train_df),
    'test_records': len(test_df),
    'total_records': len(df),
    'pha_true_count': int((df['is_pha'] == True).sum()),
    'pha_false_count': int((df['is_pha'] == False).sum()),
    'features_used': features,
    'risk_mapping': risk_mapping,
    'model_type': 'RandomForestClassifier',
    'model_params': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    },
    'test_accuracy': float((y_test == y_pred).mean()),
    'pha_true_accuracy': float(pha_true_accuracy),
    'pha_false_accuracy': float(pha_false_accuracy),
    'pha_model': {
        'features_used': pha_features,
        'test_accuracy': float(pha_accuracy),
        'model_type': 'RandomForestClassifier',
        'model_params': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
    }
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Training metadata saved to: {os.path.abspath('model_metadata.json')}")
print("\nSCRIPT FINISHED! Your model is trained and ready to be served.")