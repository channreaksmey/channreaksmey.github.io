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
    """Load data or generate synthetic if not available."""
    path = Path(data_dir)
    
    orders_file = path / 'orders.csv'
    customers_file = path / 'customers.csv'
    payments_file = path / 'payments.csv'
    
    # Check if all files exist and have content (> 1KB)
    if all(f.exists() and f.stat().st_size > 1000 for f in [orders_file, customers_file, payments_file]):
        print("Loading real Olist data...")
        orders = pd.read_csv(orders_file)
        customers = pd.read_csv(customers_file)
        payments = pd.read_csv(payments_file)
        return orders, customers, payments
    
    print("Real data not found, generating synthetic e-commerce data...")
    return generate_synthetic_data()


def generate_synthetic_data(n_customers: int = 5000) -> tuple:
    """Generate realistic synthetic Brazilian e-commerce data."""
    np.random.seed(42)
    
    # States with realistic distribution
    states = ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'DF', 'GO', 'PE']
    state_weights = [0.35, 0.15, 0.12, 0.08, 0.08, 0.05, 0.05, 0.04, 0.04, 0.04]
    
    customers = []
    for i in range(n_customers):
        state = np.random.choice(states, p=state_weights)
        customers.append({
            'customer_id': f'CUST_{i:06d}',
            'customer_unique_id': f'UNIQ_{i:06d}',
            'customer_zip_code_prefix': np.random.randint(1000, 99999),
            'customer_city': f'City_{i % 100}',
            'customer_state': state
        })
    customers_df = pd.DataFrame(customers)
    
    # Generate 1-5 orders per customer
    orders = []
    base_date = pd.Timestamp('2017-06-01')
    
    for _, cust in customers_df.iterrows():
        n_orders = np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05])
        for j in range(n_orders):
            days_offset = np.random.randint(0, 730)
            order_date = base_date + pd.Timedelta(days=days_offset)
            
            orders.append({
                'order_id': f'ORDER_{len(orders):07d}',
                'customer_id': cust['customer_id'],
                'order_status': 'delivered',
                'order_purchase_timestamp': order_date.isoformat(),
                'order_approved_at': (order_date + pd.Timedelta(hours=2)).isoformat(),
                'order_delivered_carrier_date': (order_date + pd.Timedelta(days=2)).isoformat(),
                'order_delivered_customer_date': (order_date + pd.Timedelta(days=5)).isoformat(),
                'order_estimated_delivery_date': (order_date + pd.Timedelta(days=7)).isoformat()
            })
    orders_df = pd.DataFrame(orders)
    
    # Generate payments
    payments = []
    for _, order in orders_df.iterrows():
        cust_idx = int(order['customer_id'].split('_')[1])
        base_value = 50 + (cust_idx % 10) * 25
        value = base_value * np.random.uniform(0.6, 2.5)
        
        payments.append({
            'order_id': order['order_id'],
            'payment_sequential': 1,
            'payment_type': np.random.choice(['credit_card', 'boleto', 'voucher'], p=[0.75, 0.20, 0.05]),
            'payment_installments': np.random.choice([1, 2, 3, 6, 10], p=[0.5, 0.2, 0.15, 0.1, 0.05]),
            'payment_value': round(value, 2)
        })
    payments_df = pd.DataFrame(payments)
    
    print(f"Generated {len(customers_df):,} customers, {len(orders_df):,} orders, {len(payments_df):,} payments")
    return orders_df, customers_df, payments_df

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