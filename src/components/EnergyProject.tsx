import { useState, useEffect } from 'react'

interface Building {
  building_id: number
  square_feet: number
  building_age: number
  energy_consumption_kbtu: number
  predicted_consumption: number
  error_percent: number
}

interface ModelMetrics {
  model_type: string
  metrics: {
    rmse: number
    r2: number
  }
  feature_importance: Record<string, number>
}

export default function EnergyProject() {
  const [buildings, setBuildings] = useState<Building[]>([])
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Fetch predictions and metrics
    Promise.all([
      fetch('/data/energy/predictions.json').then(r => r.json()),
      fetch('/data/energy/model_metrics.json').then(r => r.json())
    ])
      .then(([predData, metricsData]) => {
        setBuildings(predData)
        setMetrics(metricsData)
        setLoading(false)
      })
      .catch(err => {
        console.error('Failed to load data:', err)
        setLoading(false)
      })
  }, [])

  if (loading) return <div className="p-8 text-center">Loading project data...</div>

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-bold mb-2">Energy Consumption Predictor</h2>
        <p className="text-gray-600">
          Machine learning model predicting building energy usage based on size, age, and type.
          Runs entirely in your browser using pre-computed predictions.
        </p>
      </div>

      {/* Metrics Cards */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <MetricCard 
            title="Model R² Score" 
            value={metrics.metrics.r2.toFixed(3)}
            subtitle="1.0 = perfect predictions"
          />
          <MetricCard 
            title="RMSE" 
            value={`${Math.round(metrics.metrics.rmse).toLocaleString()} kBtu`}
            subtitle="Average prediction error"
          />
          <MetricCard 
            title="Features Used" 
            value={Object.keys(metrics.feature_importance).length.toString()}
            subtitle="Input variables"
          />
        </div>
      )}

      {/* Predictions Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="px-6 py-4 border-b">
          <h3 className="text-lg font-semibold">Predictions vs Actual</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Building</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Size (sq ft)</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Age</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Actual</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Predicted</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">Error %</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {buildings.map((b) => (
                <tr key={b.building_id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">#{b.building_id}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">{b.square_feet.toLocaleString()}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">{b.building_age} years</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-right">{Math.round(b.energy_consumption_kbtu).toLocaleString()}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-right font-medium text-blue-600">{Math.round(b.predicted_consumption).toLocaleString()}</td>
                  <td className={`px-6 py-4 whitespace-nowrap text-sm text-right ${Math.abs(b.error_percent) > 10 ? 'text-red-600' : 'text-green-600'}`}>
                    {b.error_percent > 0 ? '+' : ''}{b.error_percent}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Feature Importance */}
      {metrics && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Feature Importance</h3>
          <div className="space-y-3">
            {Object.entries(metrics.feature_importance)
              .sort(([,a], [,b]) => b - a)
              .map(([feature, importance]) => (
                <div key={feature} className="flex items-center">
                  <span className="w-32 text-sm text-gray-600">{feature}</span>
                  <div className="flex-1 h-4 bg-gray-100 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-blue-500 transition-all duration-500"
                      style={{ width: `${importance * 100}%` }}
                    />
                  </div>
                  <span className="w-16 text-right text-sm font-medium">{(importance * 100).toFixed(1)}%</span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  )
}

function MetricCard({ title, value, subtitle }: { title: string, value: string, subtitle: string }) {
  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <p className="text-sm font-medium text-gray-600">{title}</p>
      <p className="text-3xl font-bold text-gray-900 mt-2">{value}</p>
      <p className="text-xs text-gray-500 mt-1">{subtitle}</p>
    </div>
  )
}