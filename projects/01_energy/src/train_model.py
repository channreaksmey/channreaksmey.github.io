"""
Train energy consumption model on real Seattle data.
"""

import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


def load_processed_data(data_dir: str) -> pd.DataFrame:
    """Load preprocessed data."""
    df = pd.read_csv(Path(data_dir) / 'processed.csv')
    print(f"Training on {len(df):,} buildings")
    return df


def train_model(df: pd.DataFrame, feature_cols: list, target_col: str):
    """Train Random Forest with better validation."""
    X = df[feature_cols]
    y = df[target_col]
    
    # Convert target to log scale (handles heavy-tailed energy distribution)
    # y_log = np.log1p(y)  # log1p = log(1 + x), safe for small values
    
    # Group split using log-transformed target
    groups = df['source_building_id']
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    # y_train_log, y_test_log = y_log.iloc[train_idx], y_log.iloc[test_idx]  # Note: _log suffix
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train with more trees for stability
    model = RandomForestRegressor(
        n_estimators=200,      # More trees for stability
        max_depth=15,          # Allow deeper trees for complex patterns
        min_samples_split=5,   # Prevent overfitting on small groups
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1              # Use all CPU cores
    )
    
    # model.fit(X_train, y_train_log)
    model.fit(X_train, y_train)

        # Evaluate on test set (predict in log scale, then invert)
        # Evaluate on test set (predict in log scale, then invert)
    # y_pred_log = model.predict(X_test)
    # y_pred = np.expm1(y_pred_log)           # Invert: expm1 = exp(x) - 1
    # y_test_actual = np.expm1(y_test_log)    # Invert test labels too
    y_pred = model.predict(X_test)
    y_test_actual = y_test

    # Compute metrics on ACTUAL scale (not log)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    mae = mean_absolute_error(y_test_actual, y_pred)
    r2 = r2_score(y_test_actual, y_pred)
    
    # Calculate MAPE on actual scale
    mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
    
    # Calculate WMAPE (Weighted Mean Absolute Percentage Error)
    wmape = 100 * np.sum(np.abs(y_test_actual - y_pred)) / np.sum(np.abs(y_test_actual))
    
    # Calculate SMAPE (Symmetric Mean Absolute Percentage Error)
    epsilon = 1e-9
    smape = 100 * np.mean(2 * np.abs(y_test_actual - y_pred) / (np.abs(y_test_actual) + np.abs(y_pred) + epsilon))
    
    print(f"\nModel Performance (Test Set):")
    print(f"  RMSE: {rmse:,.0f} kBtu")
    print(f"  MAE:  {mae:,.0f} kBtu")
    print(f"  MAPE: {mape:.1f}%")
    print(f"  WMAPE: {wmape:.1f}%")
    print(f"  SMAPE: {smape:.1f}%")
    print(f"  R²:   {r2:.3f}")
    
    # Business interpretation
    print(f"\nInterpretation:")
    print(f"  - Predictions are off by {wmape:.0f}% on average (WMAPE)")
    print(f"  - Model explains {r2*100:.0f}% of energy variation")
    
    return model, X.columns.tolist(), {'rmse': rmse, 'mae': mae, 'mape': mape, 'wmape': wmape, 'smape': smape, 'r2': r2}


def generate_predictions(model, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Add predictions to dataframe."""
    df = df.copy()
    y_pred_log = model.predict(df[feature_cols])
    df['predicted_consumption'] = model.predict(df[feature_cols])
    df['prediction_error'] = df['predicted_consumption'] - df['energy_consumption_kbtu']
    df['error_percent'] = (df['prediction_error'] / df['energy_consumption_kbtu'] * 100)
    df['abs_error_percent'] = df['error_percent'].abs()
    return df


def analyze_by_type(df: pd.DataFrame) -> dict:
    """Analyze performance by building type."""
    if 'building_type' not in df.columns:
        return {}
    
    analysis = {}
    for btype in df['building_type'].unique():
        subset = df[df['building_type'] == btype]
        
        # WMAPE for this type
        wmape = 100 * np.sum(np.abs(subset['predicted_consumption'] - subset['energy_consumption_kbtu'])) / np.sum(np.abs(subset['energy_consumption_kbtu']))
        
        analysis[btype] = {
            'count': len(subset),
            'avg_actual': subset['energy_consumption_kbtu'].mean(),
            'avg_predicted': subset['predicted_consumption'].mean(),
            'mape': subset['abs_error_percent'].mean(),
            'wmape': wmape
        }
    return analysis


def save_results(model, df_with_preds: pd.DataFrame, feature_cols: list, 
                 metrics: dict, output_dir: str):
    """Save comprehensive results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, output_path / 'model.joblib')
    
    # Prepare predictions for frontend (sample of 100 diverse buildings)
    display_df = df_with_preds.nsmallest(50, 'abs_error_percent').copy()  # 50 best
    display_df = pd.concat([display_df, 
                           df_with_preds.nlargest(50, 'abs_error_percent').copy()])  # 50 worst
    
    # Add building_id
    display_df['building_id'] = range(1, len(display_df) + 1)
    
    # Select display columns
    display_cols = ['building_id', 'building_name', 'building_type', 
                   'square_feet', 'building_age', 
                   'energy_consumption_kbtu', 'predicted_consumption', 
                   'error_percent']
    display_cols = [c for c in display_cols if c in display_df.columns]
    
    display_df = display_df[display_cols].round(2)
    display_df.to_json(output_path / 'predictions.json', orient='records', indent=2)
    
    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_.round(4)))
    
    # Type analysis
    type_analysis = analyze_by_type(df_with_preds)
    
    # Comprehensive metrics
    full_metrics = {
        'model_type': 'RandomForestRegressor',
        'dataset': 'Seattle Building Energy Benchmarking 2015',
        'n_buildings': len(df_with_preds),
        'n_train': model.n_estimators,
        'features': feature_cols,
        'feature_importance': importance,
        'performance': metrics,
        'performance_by_type': type_analysis,
        'sample_predictions': display_df.head(5).to_dict('records')
    }
    
    with open(output_path / 'model_metrics.json', 'w') as f:
        json.dump(full_metrics, f, indent=2, default=str)
    
    print(f"\nSaved to {output_path}")
    print(f"  - predictions.json ({len(display_df)} examples)")
    print(f"  - model_metrics.json")
    print(f"  - model.joblib")


if __name__ == '__main__':
    data_dir = 'public/data/energy'
    output_dir = 'public/data/energy'
    
    # Load raw selected features (keeps building_type for display/analysis)
    df = load_processed_data(data_dir)
    
    # Create model-only dataframe with one-hot encoded building type
    df_model = pd.get_dummies(df, columns=['building_type'], prefix='type')
    
    # Features (exclude target and non-feature text fields)
    exclude = ['building_name', 'address', 'energy_consumption_kbtu', 'source_building_id', 'data_year']
    feature_cols = [c for c in df_model.columns if c not in exclude]
    print(f"Features: {feature_cols}")
    
    # Train on encoded model dataframe
    model, used_features, metrics = train_model(
        df_model, feature_cols, 'energy_consumption_kbtu'
    )
    
    # Predict on encoded dataframe
    df_model_with_preds = generate_predictions(model, df_model, used_features)
    
    # Attach prediction outputs back to raw dataframe for frontend/type analysis
    df_with_preds = df.copy()
    for col in ['predicted_consumption', 'prediction_error', 'error_percent', 'abs_error_percent']:
        df_with_preds[col] = df_model_with_preds[col]
    
    # Save
    save_results(model, df_with_preds, used_features, metrics, output_dir)
    
    print("\nTraining complete!")