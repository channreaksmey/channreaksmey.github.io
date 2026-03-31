"""
Customer Segmentation using RFM Analysis + K-Means Clustering.
RFM = Recency, Frequency, Monetary value
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def load_data(data_dir: str) -> tuple:
    """Load Olist e-commerce data."""
    path = Path(data_dir)
    
    orders = pd.read_csv(path / 'orders.csv')
    customers = pd.read_csv(path / 'customers.csv')
    payments = pd.read_csv(path / 'payments.csv')
    
    print(f"Orders: {len(orders):,}")
    print(f"Customers: {len(customers):,}")
    print(f"Payments: {len(payments):,}")
    
    return orders, customers, payments


def create_rfm_features(orders, customers, payments) -> pd.DataFrame:
    """
    Create RFM features for each customer:
    - Recency: Days since last order
    - Frequency: Number of orders
    - Monetary: Total payment value
    """
    # Convert dates
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
    
    # Calculate reference date (day after last order)
    reference_date = orders['order_purchase_timestamp'].max() + timedelta(days=1)
    print(f"Reference date: {reference_date}")
    
    # Aggregate payments by order
    order_payments = payments.groupby('order_id')['payment_value'].sum().reset_index()
    
    # Merge orders with payments
    orders_with_payment = orders.merge(order_payments, on='order_id', how='left')
    
    # Calculate RFM for each customer
    rfm = orders_with_payment.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (reference_date - x.max()).days,  # Recency
        'order_id': 'count',  # Frequency
        'payment_value': 'sum'  # Monetary
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    # Remove customers with negative monetary values (returns)
    rfm = rfm[rfm['monetary'] > 0]
    
    # Merge with customer location info
    rfm = rfm.merge(customers[['customer_id', 'customer_state']], on='customer_id')
    
    print(f"\nRFM Features:")
    print(rfm[['recency', 'frequency', 'monetary']].describe())
    
    return rfm


def apply_kmeans(rfm: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """Apply K-Means clustering to RFM features."""
    # Select features for clustering
    features = ['recency', 'frequency', 'monetary']
    X = rfm[features].copy()
    
    # Log transform monetary (highly skewed)
    X['monetary'] = np.log1p(X['monetary'])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Calculate cluster statistics
    print(f"\nCluster Distribution:")
    print(rfm['cluster'].value_counts().sort_index())
    
    # Analyze cluster characteristics
    cluster_stats = rfm.groupby('cluster')[features].mean().round(2)
    print(f"\nCluster Centers (mean values):")
    print(cluster_stats)
    
    return rfm, kmeans, scaler, X_scaled


def label_clusters(rfm: pd.DataFrame) -> pd.DataFrame:
    """Assign business labels to clusters based on RFM characteristics."""
    # Calculate percentiles for interpretation
    rfm['r_score'] = pd.qcut(rfm['recency'], 4, labels=[4,3,2,1])  # Lower recency = higher score
    rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 4, labels=[1,2,3,4])
    rfm['m_score'] = pd.qcut(rfm['monetary'], 4, labels=[1,2,3,4])
    
    # Simple labeling based on frequency and monetary
    def get_segment_label(cluster_id, avg_freq, avg_monetary):
        if avg_freq >= 4 and avg_monetary >= 4:
            return "Champions"
        elif avg_freq >= 3 and avg_monetary >= 3:
            return "Loyal Customers"
        elif avg_freq <= 2 and avg_monetary >= 4:
            return "Big Spenders"
        elif avg_freq >= 3 and avg_monetary <= 2:
            return "Frequent Buyers"
        elif avg_freq <= 2 and avg_monetary <= 2:
            return "At Risk"
        else:
            return f"Segment {cluster_id}"
    
    # Calculate cluster averages and assign labels
    cluster_labels = {}
    for cid in rfm['cluster'].unique():
        subset = rfm[rfm['cluster'] == cid]
        avg_f = subset['f_score'].astype(int).mean()
        avg_m = subset['m_score'].astype(int).mean()
        label = get_segment_label(cid, avg_f, avg_m)
        cluster_labels[cid] = label
    
    rfm['segment_label'] = rfm['cluster'].map(cluster_labels)
    
    return rfm, cluster_labels


def save_results(rfm: pd.DataFrame, cluster_labels: dict, output_dir: str):
    """Save segmentation results for frontend."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save sample of customers (first 200 for display)
    display_cols = ['customer_id', 'recency', 'frequency', 'monetary', 
                   'customer_state', 'cluster', 'segment_label']
    sample = rfm[display_cols].head(200).copy()
    sample['customer_id'] = range(1, len(sample) + 1)  # Anonymize
    
    sample.to_json(output_path / 'customers.json', orient='records', indent=2)
    
    # Save cluster statistics
    stats = (
    rfm.groupby(['cluster', 'segment_label'])
    .agg(
        avg_recency=('recency', 'mean'),
        avg_frequency=('frequency', 'mean'),
        avg_monetary=('monetary', 'mean'),
        count=('monetary', 'count'),
    )
    .round(2)
    .reset_index())

    stats_dict = {}
    for _, row in stats.iterrows():
        cid = int(row['cluster'])
        stats_dict[cid] = {
            'label': row['segment_label'],
            'avg_recency': float(row['avg_recency']),
            'avg_frequency': float(row['avg_frequency']),
            'avg_monetary': float(row['avg_monetary']),
            'count': int(row['count']),
            'percentage': float(row['count']) / len(rfm) * 100
        }
    
    with open(output_path / 'clusters.json', 'w') as f:
        json.dump(stats_dict, f, indent=2)
    
    # Save 3D coordinates for visualization
    rfm_sample = rfm.sample(min(500, len(rfm)), random_state=42)
    coords = rfm_sample[['recency', 'frequency', 'monetary', 'cluster', 'segment_label']].copy()
    coords['monetary'] = coords['monetary'].round(2)
    coords.to_json(output_path / 'coordinates_3d.json', orient='records', indent=2)
    
    print(f"\nSaved to {output_path}")
    print(f"  - customers.json ({len(sample)} samples)")
    print(f"  - clusters.json ({len(stats_dict)} clusters)")
    print(f"  - coordinates_3d.json ({len(coords)} points for 3D viz)")


if __name__ == '__main__':
    data_dir = 'projects/02_segmentation/data'
    output_dir = 'public/data/segmentation'
    
    # Load
    orders, customers, payments = load_data(data_dir)
    
    # Create RFM features
    rfm = create_rfm_features(orders, customers, payments)
    
    # Cluster
    rfm, kmeans, scaler, X_scaled = apply_kmeans(rfm, n_clusters=5)
    
    # Label
    rfm, cluster_labels = label_clusters(rfm)
    
    # Save
    save_results(rfm, cluster_labels, output_dir)
    
    print("\nSegmentation complete!")