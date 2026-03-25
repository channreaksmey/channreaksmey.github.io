import { useState, useEffect } from 'react'

interface Building {
  building_id: number
  building_name?: string
  building_type?: string
  square_feet: number
  building_age: number
  energy_consumption_kbtu: number
  predicted_consumption: number
  error_percent: number
}

interface ModelMetrics {
  model_type: string
  dataset: string
  n_buildings: number
  performance: {
    rmse: number
    mae: number
    mape: number
    r2: number
  }
  feature_importance: Record<string, number>
  performance_by_type?: Record<string, {
    count: number
    avg_actual: number
    mape: number
  }>
}

export default function EnergyProject() {
  const [buildings, setBuildings] = useState<Building[]>([])
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null)
  const [loading, setLoading] = useState(true)
  const [sortBy, setSortBy] = useState<'error' | 'size' | 'actual'>('error')

  useEffect(() => {
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

  const sortedBuildings = [...buildings].sort((a, b) => {
    if (sortBy === 'error') return Math.abs(b.error_percent) - Math.abs(a.error_percent)
    if (sortBy === 'size') return b.square_feet - a.square_feet
    return b.energy_consumption_kbtu - a.energy_consumption_kbtu
  })

  if (loading) return <div className="p-8 text-center">Loading Seattle energy data...</div>

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-800 text-white p-8 rounded-xl">
        <h2 className="text-3xl font-bold mb-2">Seattle Building Energy Predictor</h2>
        <p className="text-blue-100 max-w-2xl">
          Machine learning model trained on {metrics?.n_buildings.toLocaleString() || '2,800+'} real buildings 
          from the City of Seattle's 2015 Energy Benchmarking dataset. Predicts annual energy consumption 
          based on building characteristics.
        </p>
        <div className="mt-4 text-sm text-blue-200">
          Data source: <a href="https://data.seattle.gov/dataset/2015-Building-Energy-Benchmarking/2bpz-gwpy" 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="underline hover:text-white">
            data.seattle.gov
          </a>
        </div>
      </div>

      {/* Metrics */}
      {metrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard 
            title="Buildings Analyzed" 
            value={metrics.n_buildings.toLocaleString()}
            subtitle="Real Seattle properties"
          />
          <MetricCard 
            title="R² Score" 
            value={metrics.performance.r2.toFixed(3)}
            subtitle="Model accuracy"
          />
          <MetricCard 
            title="Avg Error" 
            value={`${metrics.performance.mape.toFixed(1)}%`}
            subtitle="Mean absolute % error"
          />
          <MetricCard 
            title="Top Feature" 
            value={Object.entries(metrics.feature_importance)
              .sort(([,a], [,b]) => b - a)[0][0]
              .replace('type_', '')
              .replace('_', ' ')}
            subtitle="Most predictive factor"
          />
        </div>
      )}

      {/* Performance by Type */}
      {metrics?.performance_by_type && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Performance by Building Type</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(metrics.performance_by_type)
              .sort(([,a], [,b]) => b.count - a.count)
              .map(([type, data]) => (
                <div key={type} className="border rounded-lg p-4">
                  <div className="text-sm text-gray-600">{type}</div>
                  <div className="text-2xl font-bold">{data.count}</div>
                  <div className="text-xs text-gray-500">{data.mape.toFixed(1)}% avg error</div>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Predictions Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="px-6 py-4 border-b flex justify-between items-center">
          <h3 className="text-lg font-semibold">Sample Predictions</h3>
          <select 
            value={sortBy} 
            onChange={(e) => setSortBy(e.target.value as any)}
            className="text-sm border rounded px-3 py-1"
          >
            <option value="error">Sort by Error</option>
            <option value="size">Sort by Size</option>
            <option value="actual">Sort by Consumption</option>
          </select>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left font-medium text-gray-500">Building</th>
                <th className="px-4 py-3 text-left font-medium text-gray-500">Type</th>
                <th className="px-4 py-3 text-right font-medium text-gray-500">Size (sq ft)</th>
                <th className="px-4 py-3 text-right font-medium text-gray-500">Age</th>
                <th className="px-4 py-3 text-right font-medium text-gray-500">Actual (kBtu)</th>
                <th className="px-4 py-3 text-right font-medium text-gray-500">Predicted</th>
                <th className="px-4 py-3 text-right font-medium text-gray-500">Error</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {sortedBuildings.map((b) => (
                <tr key={b.building_id} className="hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <div className="font-medium text-gray-900 truncate max-w-xs">
                      {b.building_name || `Building #${b.building_id}`}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span className="px-2 py-1 bg-gray-100 rounded text-xs">
                      {b.building_type || 'Unknown'}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right">{b.square_feet.toLocaleString()}</td>
                  <td className="px-4 py-3 text-right">{b.building_age} yrs</td>
                  <td className="px-4 py-3 text-right">{(b.energy_consumption_kbtu / 1000).toFixed(0)}k</td>
                  <td className="px-4 py-3 text-right font-medium text-blue-600">
                    {(b.predicted_consumption / 1000).toFixed(0)}k
                  </td>
                  <td className={`px-4 py-3 text-right font-medium ${
                    Math.abs(b.error_percent) > 25 ? 'text-red-600' : 
                    Math.abs(b.error_percent) > 10 ? 'text-yellow-600' : 'text-green-600'
                  }`}>
                    {b.error_percent > 0 ? '+' : ''}{b.error_percent.toFixed(1)}%
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
          <h3 className="text-lg font-semibold mb-4">What Drives Energy Consumption?</h3>
          <div className="space-y-3">
            {Object.entries(metrics.feature_importance)
              .sort(([,a], [,b]) => b - a)
              .map(([feature, importance]) => (
                <div key={feature} className="flex items-center">
                  <span className="w-40 text-sm text-gray-600 capitalize">
                    {feature.replace('type_', '').replace('_', ' ')}
                  </span>
                  <div className="flex-1 h-4 bg-gray-100 rounded-full overflow-hidden mx-4">
                    <div 
                      className="h-full bg-blue-500 transition-all duration-500"
                      style={{ width: `${importance * 100}%` }}
                    />
                  </div>
                  <span className="w-16 text-right text-sm font-medium">
                    {(importance * 100).toFixed(1)}%
                  </span>
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
    <div className="bg-white p-6 rounded-lg shadow border">
      <p className="text-sm font-medium text-gray-600">{title}</p>
      <p className="text-3xl font-bold text-gray-900 mt-2">{value}</p>
      <p className="text-xs text-gray-500 mt-1">{subtitle}</p>
    </div>
  )
}