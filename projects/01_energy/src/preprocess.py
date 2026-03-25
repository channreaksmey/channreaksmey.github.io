"""
Data preprocessing for Seattle Energy Benchmarking.
Handles real dataset with 3,000+ buildings.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw Seattle energy data."""
    df = pd.read_csv(filepath, low_memory=False, thousands=",", na_values=["NA", ""])
    print(f"Loaded {len(df):,} rows from {filepath}")
    print(f"Columns: {len(df.columns)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and filter to useful records."""
    numeric_cols = ["SiteEnergyUse(kBtu)", "PropertyGFATotal", "YearBuilt"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Only keep buildings with energy data
    df = df[df['SiteEnergyUse(kBtu)'].notna()]
    df = df[df['SiteEnergyUse(kBtu)'] > 0]
    
    # Only keep buildings with size data
    df = df[df['PropertyGFATotal'].notna()]
    df = df[df['PropertyGFATotal'] > 1000]  # Min 1,000 sq ft
    
    # Calculate building age
    current_year = datetime.now().year
    df['BuildingAge'] = current_year - df['YearBuilt']
    
    # Filter reasonable ages (0-200 years)
    df = df[df['BuildingAge'] >= 0]
    df = df[df['BuildingAge'] <= 200]
    
    # Remove extreme outliers (top 1% energy users - likely data errors)
    energy_99th = df['SiteEnergyUse(kBtu)'].quantile(0.99)
    df = df[df['SiteEnergyUse(kBtu)'] <= energy_99th]
    
    print(f"After cleaning: {len(df):,} buildings")
    print(f"Energy range: {df['SiteEnergyUse(kBtu)'].min():,.0f} - {df['SiteEnergyUse(kBtu)'].max():,.0f} kBtu")
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename columns for modeling."""
    # Map to simpler column names
    column_map = {
        'PropertyGFATotal': 'square_feet',
        'BuildingAge': 'building_age',
        'EPAPropertyType': 'building_type',
        'SiteEnergyUse(kBtu)': 'energy_consumption_kbtu',
        'BuildingName': 'building_name',
        'Address': 'address'
    }
    
    df = df.rename(columns=column_map)
    
    # Keep only columns we need
    keep_cols = ['building_name', 'address', 'square_feet', 'building_age', 
                 'building_type', 'energy_consumption_kbtu']
    
    df = df[keep_cols].copy()
    
    # Clean building types (group rare types)
    type_mapping = {
        'Office': 'Office',
        'Large Office': 'Office',
        'Small- and Mid-Sized Office': 'Office',
        'Retail Store': 'Retail',
        'Strip Mall': 'Retail',
        'Supermarket/Grocery Store': 'Retail',
        'Warehouse': 'Warehouse',
        'Distribution Center': 'Warehouse',
        'Self-Storage Facility': 'Warehouse',
        'Hotel': 'Hotel',
        'Hospital': 'Hospital',
        'Medical Office': 'Office',
        'K-12 School': 'School',
        'University': 'School',
        'Worship Facility': 'Other',
        'Restaurant': 'Restaurant',
        'Multifamily Housing': 'Residential'
    }
    
    df['building_type'] = df['building_type'].map(type_mapping)
    df['building_type'] = df['building_type'].fillna('Other')
    
    # Remove types with too few examples
    type_counts = df['building_type'].value_counts()
    valid_types = type_counts[type_counts >= 20].index
    df = df[df['building_type'].isin(valid_types)]
    
    print(f"Building types: {df['building_type'].value_counts().to_dict()}")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical variables to numeric."""
    # One-hot encode building type
    df_encoded = pd.get_dummies(df, columns=['building_type'], prefix='type')
    return df_encoded


def save_processed_data(df: pd.DataFrame, output_dir: str):
    """Save processed data and metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full dataset
    df.to_csv(output_path / 'processed.csv', index=False)
    
    # Save sample for frontend (first 100 buildings for display)
    sample_df = df.head(100).copy()
    sample_df['building_id'] = range(1, len(sample_df) + 1)
    
    # Reorder columns for frontend
    display_cols = ['building_id', 'building_name', 'building_type', 
                   'square_feet', 'building_age', 'energy_consumption_kbtu']
    sample_df = sample_df[[c for c in display_cols if c in sample_df.columns]]
    
    sample_df.to_json(output_path / 'buildings.json', orient='records', indent=2)
    
    # Save metadata
    metadata = {
        'dataset': 'Seattle Building Energy Benchmarking 2015',
        'source': 'https://data.seattle.gov/Built-Environment/Building-Energy-Benchmarking-Data-2015-Present/teqw-tu6e/about_data',
        'total_buildings': len(df),
        'display_sample': len(sample_df),
        'features': [c for c in df.columns if not c.startswith('type_') and c not in ['energy_consumption_kbtu', 'building_name', 'address']],
        'building_types': df['building_type'].unique().tolist() if 'building_type' in df.columns else [],
        'date_processed': pd.Timestamp.now().isoformat()
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved to {output_path}:")
    print(f"  - processed.csv ({len(df):,} buildings)")
    print(f"  - buildings.json ({len(sample_df)} sample for display)")


if __name__ == '__main__':
    raw_path = 'projects/01_energy/data/seattle_energy.csv'
    output_dir = 'public/data/energy'
    
    df = load_data(raw_path)
    df = clean_data(df)
    df = select_features(df)
    save_processed_data(df, output_dir)
    
    print("\nPreprocessing complete!")