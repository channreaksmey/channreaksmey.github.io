import { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'

interface Customer {
  customer_id: number
  recency: number
  frequency: number
  monetary: number
  cluster: number
  segment_label: string
}

interface Cluster {
  label: string
  avg_recency: number
  avg_frequency: number
  avg_monetary: number
  count: number
  percentage: number
}

export default function SegmentationProject() {
  const [customers, setCustomers] = useState<Customer[]>([])
  const [clusters, setClusters] = useState<Record<string, Cluster>>({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      fetch('/data/segmentation/customers.json').then(r => r.json()),
      fetch('/data/segmentation/clusters.json').then(r => r.json())
    ])
      .then(([custData, clustData]) => {
        setCustomers(custData)
        setClusters(clustData)
        setLoading(false)
      })
      .catch(err => {
        console.error('Failed to load segmentation data:', err)
        setLoading(false)
      })
  }, [])

  if (loading) return <div className="p-8 text-center">Loading customer segmentation...</div>

  const clusterColors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-pink-600 text-white p-8 rounded-xl">
        <h2 className="text-3xl font-bold mb-2">Customer Segmentation Analysis</h2>
        <p className="text-purple-100 max-w-2xl">
          K-Means clustering on 95,000+ Brazilian e-commerce customers using RFM analysis 
          (Recency, Frequency, Monetary). Identifies 5 distinct customer segments for 
          targeted marketing strategies.
        </p>
      </div>

      {/* Cluster Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {Object.entries(clusters).map(([id, cluster]) => (
          <div key={id} className="bg-white p-6 rounded-lg shadow border-l-4" 
               style={{ borderLeftColor: clusterColors[parseInt(id) % clusterColors.length] }}>
            <h3 className="text-lg font-bold mb-2">{cluster.label}</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Customers:</span>
                <span className="font-medium">{cluster.count.toLocaleString()} ({cluster.percentage.toFixed(1)}%)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Avg Recency:</span>
                <span className="font-medium">{cluster.avg_recency.toFixed(0)} days</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Avg Frequency:</span>
                <span className="font-medium">{cluster.avg_frequency.toFixed(1)} orders</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Avg Monetary:</span>
                <span className="font-medium">R${cluster.avg_monetary.toFixed(2)}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* 3D Scatter Plot */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">3D Customer Segments (RFM Space)</h3>
        <Plot
          data={Object.keys(clusters).map((clusterId) => {
            const clusterCustomers = customers.filter(c => c.cluster === parseInt(clusterId))
            return {
              x: clusterCustomers.map(c => c.recency),
              y: clusterCustomers.map(c => c.frequency),
              z: clusterCustomers.map(c => c.monetary),
              mode: 'markers',
              type: 'scatter3d',
              name: clusters[clusterId]?.label || `Cluster ${clusterId}`,
              marker: {
                size: 4,
                color: clusterColors[parseInt(clusterId) % clusterColors.length],
                opacity: 0.7
              },
              text: clusterCustomers.map(c => 
                `${clusters[clusterId]?.label}<br>` +
                `Recency: ${c.recency} days<br>` +
                `Frequency: ${c.frequency}<br>` +
                `Monetary: R$${c.monetary.toFixed(2)}`
              ),
              hoverinfo: 'text'
            }
          })}
          layout={{
            title: { text: 'Customers in RFM Space' },
            scene: {
              xaxis: { title: { text: 'Recency (days)' } },
              yaxis: { title: { text: 'Frequency (orders)' } },
              zaxis: { title: { text: 'Monetary (R$)' } }
            },
            margin: { t: 30, r: 0, b: 0, l: 0 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent'
          }}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: '100%', height: '500px' }}
        />
      </div>

      {/* Methodology */}
      <div className="bg-gray-50 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-3">Methodology</h3>
        <div className="grid md:grid-cols-3 gap-4 text-sm">
          <div>
            <h4 className="font-medium mb-2">1. RFM Features</h4>
            <p className="text-gray-600">
              Calculated Recency (days since last order), Frequency (order count), 
              and Monetary (total spent) for each of 95,000 customers.
            </p>
          </div>
          <div>
            <h4 className="font-medium mb-2">2. K-Means Clustering</h4>
            <p className="text-gray-600">
              Applied K-Means with k=5 after standardizing features. 
              Used log-transform on Monetary to handle skewness.
            </p>
          </div>
          <div>
            <h4 className="font-medium mb-2">3. Segment Labeling</h4>
            <p className="text-gray-600">
              Assigned business labels based on cluster characteristics: 
              Champions, Loyal Customers, Big Spenders, etc.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}