"""
Data preprocessing for Energy Consumption Predictor.
Loads raw data, cleans it, encodes categories, and saves processed version.
"""

import pandas as pd
import json
from pathlib import Path


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate data."""
    # Remove any rows with missing values
    df = df.dropna()
    
    # Remove outliers (buildings with unrealistic energy consumption)
    q99 = df['energy_consumption_kbtu'].quantile(0.99)
    df = df[df['energy_consumption_kbtu'] <= q99]
    
    print(f"After cleaning: {len(df)} rows")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical variables to numeric."""
    # One-hot encode building_type (creates separate columns for each type)
    df_encoded = pd.get_dummies(df, columns=['building_type'], prefix='type')
    
    return df_encoded


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (excluding target and ID)."""
    exclude = ['building_id', 'energy_consumption_kbtu']
    return [col for col in df.columns if col not in exclude]


def save_processed_data(df: pd.DataFrame, output_dir: str):
    """Save processed data and metadata for frontend."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV for Python
    df.to_csv(output_path / 'processed.csv', index=False)
    
    # Save as JSON for React frontend
    df.to_json(output_path / 'buildings.json', orient='records', indent=2)
    
    # Save metadata about columns
    metadata = {
        'features': get_feature_columns(df),
        'target': 'energy_consumption_kbtu',
        'n_samples': len(df),
        'building_types': ['Office', 'Retail', 'Warehouse']
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved processed data to {output_path}")


if __name__ == '__main__':
    # Define paths
    raw_path = 'projects/01_energy/data/sample_buildings.csv'
    output_dir = 'public/data/energy'
    
    # Run pipeline
    df = load_data(raw_path)
    df = clean_data(df)
    df = encode_features(df)
    save_processed_data(df, output_dir)
    
    print("\nPreprocessing complete!")
    print(f"Features: {get_feature_columns(df)}")