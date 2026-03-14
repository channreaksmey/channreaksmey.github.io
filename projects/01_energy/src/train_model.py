"""
Train energy consumption prediction model.
Uses Random Forest regression and exports predictions for frontend.
"""

import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def load_processed_data(data_dir: str) -> pd.DataFrame:
    """Load preprocessed data."""
    df = pd.read_csv(Path(data_dir) / 'processed.csv')
    return df


def train_model(df: pd.DataFrame, feature_cols: list, target_col: str):
    """Train Random Forest model."""
    X = df[feature_cols]
    y = df[target_col]
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,  # Number of trees
        max_depth=10,      # Prevent overfitting
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"  RMSE: {rmse:,.0f} kBtu")
    print(f"  R²: {r2:.3f}")
    
    return model, X.columns.tolist()


def generate_predictions(model, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Add prediction columns to dataframe."""
    df = df.copy()
    df['predicted_consumption'] = model.predict(df[feature_cols])
    df['prediction_error'] = df['predicted_consumption'] - df['energy_consumption_kbtu']
    df['error_percent'] = (df['prediction_error'] / df['energy_consumption_kbtu'] * 100).round(2)
    return df


def save_results(model, df_with_preds: pd.DataFrame, feature_cols: list, output_dir: str):
    """Save model and predictions."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model for Python use
    joblib.dump(model, output_path / 'model.joblib')
    
    # Save predictions as JSON for React
    # Convert to records format for easy frontend consumption
    predictions = df_with_preds[[
        'building_id', 'square_feet', 'building_age',
        'energy_consumption_kbtu', 'predicted_consumption',
        'error_percent'
    ]].copy()
    
    # Round numbers for cleaner JSON
    predictions = predictions.round(2)
    
    predictions.to_json(
        output_path / 'predictions.json', 
        orient='records', 
        indent=2
    )
    
    # Save model metrics and feature importance
    importance = dict(zip(feature_cols, model.feature_importances_.round(4)))
    
    metrics = {
        'model_type': 'RandomForestRegressor',
        'n_estimators': model.n_estimators,
        'features': feature_cols,
        'feature_importance': importance,
        'metrics': {
            'rmse': float(np.sqrt(mean_squared_error(
                df_with_preds['energy_consumption_kbtu'],
                df_with_preds['predicted_consumption']
            ))),
            'r2': float(r2_score(
                df_with_preds['energy_consumption_kbtu'],
                df_with_preds['predicted_consumption']
            ))
        }
    }
    
    with open(output_path / 'model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nSaved results to {output_path}")


if __name__ == '__main__':
    data_dir = 'public/data/energy'
    output_dir = 'public/data/energy'
    
    # Load data
    df = load_processed_data(data_dir)
    
    # Get features (all columns except target and ID)
    feature_cols = [c for c in df.columns if c not in ['building_id', 'energy_consumption_kbtu']]
    
    # Train
    model, used_features = train_model(df, feature_cols, 'energy_consumption_kbtu')
    
    # Generate predictions for all buildings
    df_with_preds = generate_predictions(model, df, used_features)
    
    # Save everything
    save_results(model, df_with_preds, used_features, output_dir)
    
    print("\nTraining complete!")
    print(f"Feature importance: {dict(zip(used_features, model.feature_importances_.round(2)))}")