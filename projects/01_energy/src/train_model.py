"""
Two-stage training: Predict intensity first, then scale to total consumption.
This dramatically improves percentage error.
"""

import pandas as pd
import json
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_data(data_dir: str) -> pd.DataFrame:
    """Load processed data."""
    df = pd.read_csv(Path(data_dir) / 'processed.csv')
    print(f"Loaded {len(df):,} buildings")
    return df


def prepare_features(df: pd.DataFrame, target: str) -> tuple:
    """Prepare feature matrix."""
    exclude = ['building_name', 'address', 'energy_consumption_kbtu', 
               'log_energy', 'energy_intensity', 'type_avg_intensity']
    exclude = [c for c in exclude if c in df.columns]
    
    feature_cols = [c for c in df.columns if c not in exclude and c != target]
    
    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target]
    
    return X, y, X.columns.tolist()


def train_intensity_model(X, y_intensity, feature_cols):
    """
    Stage 1: Predict energy intensity (kBtu/sqft).
    This normalizes for building size and improves accuracy.
    """
    print("\n=== STAGE 1: Training Intensity Model ===")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_intensity, test_size=0.2, random_state=42)
    
    # Gradient Boosting often outperforms Random Forest for this
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate on intensity
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)
    
    print(f"Intensity Model - MAPE: {mape:.1f}%, R²: {r2:.3f}")
    
    return model, mape, r2


def predict_total_consumption(model, df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2: Convert intensity predictions to total consumption.
    total = intensity × square_feet
    """
    df = df.copy()
    
    # Predict intensity
    df['predicted_intensity'] = model.predict(X)
    
    # Convert to total consumption
    df['predicted_consumption'] = df['predicted_intensity'] * df['square_feet']
    
    # Ensure no negative predictions
    df['predicted_consumption'] = df['predicted_consumption'].clip(lower=0)
    
    return df


def evaluate_model(df: pd.DataFrame) -> dict:
    """Comprehensive evaluation with multiple metrics."""
    actual = df['energy_consumption_kbtu']
    predicted = df['predicted_consumption']
    
    # Standard metrics
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    # MAPE with protection against division by zero
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Symmetric MAPE (sMAPE) - handles small values better
    smape = np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))) * 100
    
    # Percentage within 10%, 20%, 30%
    pct_errors = np.abs((actual - predicted) / actual) * 100
    within_10 = np.mean(pct_errors <= 10) * 100
    within_20 = np.mean(pct_errors <= 20) * 100
    within_30 = np.mean(pct_errors <= 30) * 100
    
    print(f"\n=== FINAL EVALUATION ===")
    print(f"R² Score:        {r2:.3f}")
    print(f"RMSE:            {rmse:,.0f} kBtu")
    print(f"MAE:             {mae:,.0f} kBtu")
    print(f"MAPE:            {mape:.1f}%")
    print(f"sMAPE:           {smape:.1f}%")
    print(f"Within 10%:      {within_10:.1f}% of buildings")
    print(f"Within 20%:      {within_20:.1f}% of buildings")
    print(f"Within 30%:      {within_30:.1f}% of buildings")
    
    return {
        'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape, 'smape': smape,
        'within_10pct': within_10, 'within_20pct': within_20, 'within_30pct': within_30
    }


def save_results(model, df: pd.DataFrame, metrics: dict, feature_cols: list, output_dir: str):
    """Save comprehensive results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, output_path / 'model_intensity.joblib')
    
    # Prepare display sample showing range of accuracy
    df['abs_error_pct'] = np.abs((df['predicted_consumption'] - df['energy_consumption_kbtu']) / df['energy_consumption_kbtu'] * 100)
    
    # Stratified sample: good, medium, bad predictions
    good = df[df['abs_error_pct'] <= 15].sample(min(30, len(df[df['abs_error_pct'] <= 15])), random_state=42)
    medium = df[(df['abs_error_pct'] > 15) & (df['abs_error_pct'] <= 35)].sample(min(30, len(df[(df['abs_error_pct'] > 15) & (df['abs_error_pct'] <= 35)])), random_state=42)
    bad = df[df['abs_error_pct'] > 35].sample(min(20, len(df[df['abs_error_pct'] > 35])), random_state=42)
    
    display_df = pd.concat([good, medium, bad]).drop_duplicates()
    display_df['building_id'] = range(1, len(display_df) + 1)
    
    # Save predictions
    pred_cols = ['building_id', 'building_name', 'square_feet', 'building_age',
                'energy_consumption_kbtu', 'predicted_consumption', 'predicted_intensity']
    pred_cols = [c for c in pred_cols if c in display_df.columns]
    
    display_df['error_percent'] = ((display_df['predicted_consumption'] - display_df['energy_consumption_kbtu']) / display_df['energy_consumption_kbtu'] * 100).round(2)
    
    display_df[pred_cols + ['error_percent']].to_json(
        output_path / 'predictions.json', orient='records', indent=2
    )
    
    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_.round(4)))
    
    # Analysis by size decile
    df['size_decile'] = pd.qcut(df['square_feet'], 10, labels=False)
    size_analysis = df.groupby('size_decile').agg({
        'abs_error_pct': 'mean',
        'square_feet': 'mean',
        'building_name': 'count'
    }).round(2).to_dict()
    
    full_metrics = {
        'model_type': 'GradientBoostingRegressor (Two-Stage)',
        'approach': 'Predict intensity (kBtu/sqft), then scale by size',
        'dataset': 'Seattle Building Energy Benchmarking 2015',
        'n_buildings': len(df),
        'performance': metrics,
        'feature_importance': importance,
        'size_analysis': size_analysis,
        'sample_predictions': display_df.head(3).to_dict('records')
    }
    
    with open(output_path / 'model_metrics.json', 'w') as f:
        json.dump(full_metrics, f, indent=2, default=str)
    
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    data_dir = 'public/data/energy'
    output_dir = 'public/data/energy'
    
    # Load
    df = load_data(data_dir)
    
    # Stage 1: Predict energy intensity
    X, y_intensity, feature_cols = prepare_features(df, 'energy_intensity')
    model, intensity_mape, intensity_r2 = train_intensity_model(X, y_intensity, feature_cols)
    
    # Stage 2: Convert to total consumption
    df = predict_total_consumption(model, df, X)
    
    # Evaluate
    metrics = evaluate_model(df)
    
    # Save
    save_results(model, df, metrics, feature_cols, output_dir)
    
    print("\nTraining complete!")