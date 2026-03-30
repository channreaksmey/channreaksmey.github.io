import Plot from 'react-plotly.js'

type Trace = {
    x: number[]
    y: number[]
    mode: 'markers' | 'lines'
    type: 'scatter'
    name: string
    marker?: {
        size: number
        color: string
        opacity: number
    }
    line?: {
        color: string
        dash: 'dash' | 'solid' | 'dot' | 'dashdot'
        width: number
    }
    text?: string[]
    hoverinfo?: 'text' | 'skip'
}

interface Building {
  square_feet: number
  energy_consumption_kbtu: number
  predicted_consumption: number
  building_type?: string
}

interface Props {
  buildings: Building[]
}

export default function EnergyScatterPlot({ buildings }: Props) {
  // Color by building type if available
  const types = [...new Set(buildings.map(b => b.building_type || 'Unknown'))]
  const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899']

  const traces: Trace[] = types.map((type, idx) => {
    const typeBuildings = buildings.filter(b => (b.building_type || 'Unknown') === type)
    return {
      x: typeBuildings.map(b => b.square_feet),
      y: typeBuildings.map(b => b.energy_consumption_kbtu / 1000000), // Convert to millions
      mode: 'markers' as const,
      type: 'scatter' as const,
      name: type,
      marker: {
        size: 8,
        color: colors[idx % colors.length],
        opacity: 0.7
      },
      text: typeBuildings.map(b => 
        `Size: ${b.square_feet.toLocaleString()} sq ft<br>` +
        `Actual: ${(b.energy_consumption_kbtu/1000000).toFixed(1)}M kBtu<br>` +
        `Predicted: ${(b.predicted_consumption/1000000).toFixed(1)}M kBtu`
      ),
      hoverinfo: 'text'
    }
  })

  // Add perfect prediction line
  const maxVal = Math.max(...buildings.map(b => b.energy_consumption_kbtu)) / 1000000
  
  traces.push({
    x: [0, maxVal * 1000000],
    y: [0, maxVal],
    mode: 'lines' as const,
    type: 'scatter' as const,
    name: 'Perfect Prediction',
    line: {
      color: 'gray',
      dash: 'dash',
      width: 2
    },
    hoverinfo: 'skip'
  })

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <Plot
        data={traces}
        layout={{
            title: { text: 'Actual vs Predicted Energy Consumption' },
            xaxis: {
                title: { text: 'Building Size (sq ft)' },
                type: 'log',
                autorange: true
            },
            yaxis: {
                title: { text: 'Annual Energy Use (Million kBtu)' },
                type: 'log',
                autorange: true
            },
            hovermode: 'closest',
            showlegend: true,
            legend: { x: 0, y: 1 },
            margin: { t: 40, r: 20, b: 60, l: 80 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent'
        }}
        config={{
          responsive: true,
          displayModeBar: false
        }}
        style={{ width: '100%', height: '400px' }}
      />
    </div>
  )
}