"""
Enhanced preprocessing with feature engineering for better accuracy.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw Seattle energy data."""
    df = pd.read_csv(filepath, low_memory=False, thousands=",", na_values=["NA", ""])
    print(f"Loaded {len(df):,} rows")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced cleaning with outlier handling."""
    numeric_cols = ["SiteEnergyUse(kBtu)", "PropertyGFATotal", "YearBuilt"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Require essential fields
    df = df[df['SiteEnergyUse(kBtu)'].notna()]
    df = df[df['SiteEnergyUse(kBtu)'] > 0]
    df = df[df['PropertyGFATotal'].notna()]
    df = df[df['PropertyGFATotal'] >= 5000]  # Min 5,000 sq ft (focus on commercial)
    
    # Calculate age
    current_year = datetime.now().year
    df['BuildingAge'] = current_year - df['YearBuilt']
    df = df[df['BuildingAge'] >= 0]
    df = df[df['BuildingAge'] <= 150]  # Cap at 150 years
    
    # Remove extreme outliers using IQR method (more robust than percentile)
    Q1 = df['SiteEnergyUse(kBtu)'].quantile(0.25)
    Q3 = df['SiteEnergyUse(kBtu)'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.5 * IQR  # More permissive lower bound
    upper_bound = Q3 + 2.5 * IQR  # Stricter upper bound
    
    before = len(df)
    df = df[(df['SiteEnergyUse(kBtu)'] >= lower_bound) & 
            (df['SiteEnergyUse(kBtu)'] <= upper_bound)]
    print(f"Removed {before - len(df)} extreme outliers")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features that capture energy efficiency, not just consumption.
    This is the key to improving percentage error.
    """
    
    # 1. Energy Intensity (kBtu per sq ft) - the key efficiency metric
    df['energy_intensity'] = df['SiteEnergyUse(kBtu)'] / df['PropertyGFATotal']
    
    # 2. Size categories (non-linear effects)
    df['size_category'] = pd.cut(df['PropertyGFATotal'],
                                  bins=[0, 25000, 75000, 150000, 500000, float('inf')],
                                  labels=['Tiny', 'Small', 'Medium', 'Large', 'Huge'])
    
    # 3. Age categories (different eras have different efficiency standards)
    df['era_built'] = pd.cut(df['BuildingAge'],
                            bins=[-1, 10, 30, 50, 100, float('inf')],
                            labels=['New', 'Modern', 'Older', 'Old', 'Historic'])
    
    # 4. Log transforms (energy and size often have log-normal distributions)
    df['log_square_feet'] = np.log1p(df['PropertyGFATotal'])
    df['log_energy'] = np.log1p(df['SiteEnergyUse(kBtu)'])
    
    # 5. Polynomial features (interaction effects)
    df['size_x_age'] = df['PropertyGFATotal'] * df['BuildingAge']
    df['size_squared'] = df['PropertyGFATotal'] ** 2
    
    # 6. Efficiency proxies (if available)
    if 'ENERGYSTARScore' in df.columns:
        df['energy_star'] = df['ENERGYSTARScore'].fillna(df['ENERGYSTARScore'].median())
    else:
        df['energy_star'] = 50  # Neutral default
    
    # 7. Building type density (some types are more efficient)
    type_avg_intensity = df.groupby('EPAPropertyType')['energy_intensity'].transform('median')
    df['type_avg_intensity'] = type_avg_intensity
    
    print(f"Engineered features: {df.shape[1]} total columns")
    print(f"Key: energy_intensity ranges {df['energy_intensity'].min():.1f} to {df['energy_intensity'].max():.1f} kBtu/sqft")
    
    return df


def select_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Select final features and encode categoricals."""
    
    # Simplify building types
    type_mapping = {
        'Office': 'Office',
        'Large Office': 'Office',
        'Small- and Mid-Sized Office': 'Office',
        'Medical Office': 'Office',
        'Retail Store': 'Retail',
        'Strip Mall': 'Retail',
        'Supermarket/Grocery Store': 'Retail',
        'Restaurant': 'Restaurant',
        'Warehouse': 'Warehouse',
        'Distribution Center': 'Warehouse',
        'Self-Storage Facility': 'Warehouse',
        'Hotel': 'Hotel',
        'Hospital': 'Hospital',
        'K-12 School': 'School',
        'University': 'School',
        'Worship Facility': 'Other',
        'Multifamily Housing': 'Residential',
        'Mixed Use Property': 'Mixed'
    }
    
    df['building_type'] = df['EPAPropertyType'].map(type_mapping)
    df['building_type'] = df['building_type'].fillna('Other')
    
    # Keep only types with sufficient samples
    type_counts = df['building_type'].value_counts()
    valid_types = type_counts[type_counts >= 30].index
    df = df[df['building_type'].isin(valid_types)]
    
    # Rename core columns
    df = df.rename(columns={
        'PropertyGFATotal': 'square_feet',
        'BuildingAge': 'building_age',
        'SiteEnergyUse(kBtu)': 'energy_consumption_kbtu',
        'BuildingName': 'building_name',
        'Address': 'address'
    })
    
    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=['building_type', 'size_category', 'era_built'], 
                       prefix=['type', 'size', 'era'])
    
    return df


def save_processed_data(df: pd.DataFrame, output_dir: str):
    """Save with stratified sampling for display."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full dataset
    df.to_csv(output_path / 'processed.csv', index=False)
    
    # Smart sampling: representative of all building types and sizes
    # Sample evenly from size quartiles to show range
    df['temp_size_q'] = pd.qcut(df['square_feet'], 4, labels=False)
    
    samples = []
    for q in range(4):
        for t in df['building_type_Office'].unique() if 'building_type_Office' in df.columns else [0]:
            subset = df[df['temp_size_q'] == q]
            if len(subset) > 5:
                samples.append(subset.sample(min(5, len(subset)), random_state=42))
    
    if samples:
        display_df = pd.concat(samples).drop_duplicates().head(100)
    else:
        display_df = df.sample(min(100, len(df)), random_state=42)
    
    display_df = display_df.drop('temp_size_q', axis=1, errors='ignore')
    display_df['building_id'] = range(1, len(display_df) + 1)
    
    # Select display columns
    display_cols = ['building_id', 'building_name', 'square_feet', 'building_age',
                   'energy_consumption_kbtu', 'energy_intensity']
    display_cols = [c for c in display_cols if c in display_df.columns]
    display_df[display_cols].to_json(output_path / 'buildings.json', 
                                      orient='records', indent=2)
    
    # Metadata
    metadata = {
        'dataset': 'Seattle Building Energy Benchmarking 2015',
        'source': 'https://data.seattle.gov/dataset/2015-Building-Energy-Benchmarking/2bpz-gwpy',
        'total_buildings': len(df),
        'display_sample': len(display_df),
        'avg_intensity': df['energy_intensity'].mean(),
        'features': [c for c in df.columns if not any(x in c for x in ['building_name', 'address', 'energy_consumption'])],
        'building_types': [c for c in df.columns if c.startswith('type_')],
        'date_processed': pd.Timestamp.now().isoformat()
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved: {len(df):,} buildings, {len(display_df)} display sample")


if __name__ == '__main__':
    raw_path = 'projects/01_energy/data/seattle_energy.csv'
    output_dir = 'public/data/energy'
    
    df = load_data(raw_path)
    df = clean_data(df)
    df = engineer_features(df)  # NEW: Feature engineering
    df = select_and_encode(df)
    save_processed_data(df, output_dir)
    
    print("\nPreprocessing complete!")