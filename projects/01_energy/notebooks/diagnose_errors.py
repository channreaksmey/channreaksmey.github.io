import json
import joblib
import numpy as np
import pandas as pd

# Load processed data
df = pd.read_csv('public/data/energy/processed.csv')

# Load metrics JSON as dict
with open('public/data/energy/model_metrics.json', 'r') as f:
    metrics = json.load(f)

print('=== TRAINING METRICS ===')
print(metrics['performance'])

# Rebuild features exactly like training
df_model = pd.get_dummies(df, columns=['building_type'], prefix='type')

# Use exact trained feature list from metrics file
feature_cols = metrics['features']
X = df_model.reindex(columns=feature_cols, fill_value=0)
model = joblib.load('public/data/energy/model.joblib')
pred = model.predict(X)

# If model was trained on log1p(target), uncomment next line:
# pred = np.expm1(pred)

df['predicted'] = pred
df['error'] = df['predicted'] - df['energy_consumption_kbtu']
df['abs_error'] = df['error'].abs()
df['error_pct'] = (df['abs_error'] / df['energy_consumption_kbtu']) * 100

print('\n=== ERROR ANALYSIS ===')
print(f"\nOverall MAPE: {df['error_pct'].mean():.1f}%")
print(f"Median APE: {df['error_pct'].median():.1f}%")

print('\n--- By Building Size ---')
df['size_quartile'] = pd.qcut(df['square_feet'], 4, labels=['Q1 (Small)', 'Q2', 'Q3', 'Q4 (Large)'])
print(df.groupby('size_quartile', observed=False)['error_pct'].agg(['mean', 'median', 'count']))

print('\n--- Worst Predictions ---')
worst = df.nlargest(10, 'error_pct')[['building_name', 'square_feet', 'energy_consumption_kbtu', 'predicted', 'error_pct']]
print(worst)

print('\n--- Best Predictions ---')
best = df.nsmallest(10, 'error_pct')[['building_name', 'square_feet', 'energy_consumption_kbtu', 'predicted', 'error_pct']]
print(best)

print('\n--- By Building Type ---')
print(df.groupby('building_type')['error_pct'].agg(['mean', 'median', 'count']))

print('\n--- Correlation ---')
print(f"Error vs Size: {df['abs_error'].corr(df['square_feet']):.3f}")
print(f"Error % vs Size: {df['error_pct'].corr(df['square_feet']):.3f}")
