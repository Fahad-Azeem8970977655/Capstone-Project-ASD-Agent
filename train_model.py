"""
Train a RandomForest on Final.csv and save model + meta to model/
Updated with consistent question formatting
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from data_prep import load_dataset, basic_cleaning, infer_features_and_target, train_test_split_df
import os

# FIXED: Changed from absolute Linux path to relative path
OUT_DIR = Path("model")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT_DIR / "model.joblib"
META_PATH = OUT_DIR / "model.meta.joblib"

def encode_df_numeric(df: pd.DataFrame, feature_cols):
    """Enhanced encoding to handle various answer formats"""
    df_enc = df.copy()
    
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(df_enc[c]):
            # Convert to string and clean
            s = df_enc[c].astype(str).str.strip().str.lower()
            
            # Handle categorical responses
            mapping_dict = {
                'yes': 1, 'y': 1, 'true': 1, '1': 1, 'often': 2, 'very often': 3,
                'no': 0, 'n': 0, 'false': 0, '0': 0, 'never': 0,
                'sometimes': 1, 'maybe': 1, 'some-time': 1,
                'before 9 months': 0, '9-12 months': 1, '1-2 years': 2, 'after 2 years': 3
            }
            
            # Apply mapping
            s_mapped = s.replace(mapping_dict)
            
            # For any remaining non-numeric values, try to extract numbers
            def extract_numeric(val):
                if isinstance(val, (int, float)) and not pd.isna(val):
                    return float(val)
                try:
                    return float(val)
                except:
                    # Extract numbers from strings like "2 hour", "30 words"
                    if isinstance(val, str):
                        digits = ''.join(ch for ch in val if ch.isdigit() or ch == '.')
                        return float(digits) if digits else 0.0
                    return 0.0
            
            # Convert to numeric
            df_enc[c] = pd.to_numeric(s_mapped, errors='coerce')
            
            # Fill remaining NaN values
            df_enc[c] = df_enc[c].fillna(df_enc[c].apply(extract_numeric))
    
    return df_enc

def train_and_save(csv_path: str = None):
    """Main training function with consistent question formatting"""
    print("Loading dataset...")
    df = load_dataset(csv_path)
    print(f"Original dataset shape: {df.shape}")
    
    print("Cleaning data...")
    df = basic_cleaning(df)
    print(f"After cleaning shape: {df.shape}")
    
    print("Identifying features and target...")
    feature_cols, target_col = infer_features_and_target(df)
    print(f"Features: {len(feature_cols)}, Target: {target_col}")
    print(f"Feature columns: {feature_cols}")
    
    print("Encoding data to numeric...")
    df_enc = encode_df_numeric(df, feature_cols + [target_col])
    
    # Prepare features and target
    X = df_enc[feature_cols].values
    y = pd.to_numeric(df_enc[target_col], errors='coerce').fillna(0).astype(int).values
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split_df(X, y)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Train model
    print("Training Random Forest model...")
    clf = RandomForestClassifier(
        n_estimators=200, 
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred)
    
    report = classification_report(y_test, y_pred, zero_division=0)
    
    auc = None
    try:
        y_proba = clf.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception as e:
        print(f"ROC AUC calculation failed: {e}")
    
    # Feature importance
    feature_importance = dict(zip(feature_cols, clf.feature_importances_))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Save model and metadata
    joblib.dump(clf, MODEL_PATH)
    
    # Save metadata
    metadata = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "feature_importance": feature_importance,
        "training_info": {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "roc_auc": auc,
            "n_samples": len(X),
            "n_features": len(feature_cols),
            "class_distribution": pd.Series(y).value_counts().to_dict()
        }
    }
    
    joblib.dump(metadata, META_PATH)
    
    # Print results
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved meta to: {META_PATH}")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"ROC AUC: {auc:.4f}" if auc else "ROC AUC: Not available")
    print(f"\nClassification Report:\n{report}")
    
    print(f"\nTop 5 Most Important Features:")
    for feature, importance in sorted_importance[:5]:
        print(f"  {feature}: {importance:.4f}")
    
    print(f"\nModel Info:")
    print(f"  Estimators: {clf.n_estimators}")
    print(f"  Classes: {clf.classes_}")
    print(f"  Features: {clf.n_features_in_}")
    
    return clf, metadata

if __name__ == "__main__":
    train_and_save()