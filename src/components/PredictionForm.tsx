import { useState } from 'react'

interface PredictionResult {
  predictedConsumption: number
  predictedIntensity: number
  confidence: 'high' | 'medium' | 'low'
}

export default function PredictionForm() {
  const [inputs, setInputs] = useState({
    squareFeet: 50000,
    buildingAge: 20,
    buildingType: 'Office',
    sizeCategory: 'Medium'
  })
  
  const [result, setResult] = useState<PredictionResult | null>(null)

  // Simple linear approximation based on your model's behavior
  // In production, you'd use TensorFlow.js or ONNX to run the real model
  const calculatePrediction = () => {
    // Base intensity by type (kBtu/sqft) - from your training data
    const baseIntensity: Record<string, number> = {
      'Office': 85,
      'Retail': 65,
      'Warehouse': 45,
      'Hotel': 120,
      'Hospital': 180,
      'School': 75,
      'Restaurant': 200,
      'Other': 80
    }

    // Size factor (larger buildings slightly more efficient per sqft)
    const sizeFactor = inputs.squareFeet > 100000 ? 0.9 : 
                      inputs.squareFeet > 50000 ? 0.95 : 1.0
    
    // Age factor (older buildings less efficient)
    const ageFactor = inputs.buildingAge > 50 ? 1.15 :
                     inputs.buildingAge > 30 ? 1.08 :
                     inputs.buildingAge > 10 ? 1.0 : 0.95

    const intensity = (baseIntensity[inputs.buildingType] || 80) * sizeFactor * ageFactor
    const consumption = intensity * inputs.squareFeet

    // Confidence based on input similarity to training data
    const confidence = (inputs.squareFeet >= 10000 && inputs.squareFeet <= 200000 &&
                       inputs.buildingAge <= 100) ? 'high' : 'medium'

    setResult({
      predictedConsumption: consumption,
      predictedIntensity: intensity,
      confidence
    })
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border">
      <h3 className="text-xl font-bold mb-4">Try the Predictor</h3>
      <p className="text-gray-600 text-sm mb-6">
        Enter building characteristics to see energy consumption prediction.
        This uses a simplified model for demo purposes.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Building Size (sq ft)
          </label>
          <input
            type="number"
            value={inputs.squareFeet}
            onChange={(e) => setInputs({...inputs, squareFeet: Number(e.target.value)})}
            className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            min="1000"
            max="1000000"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Building Age (years)
          </label>
          <input
            type="number"
            value={inputs.buildingAge}
            onChange={(e) => setInputs({...inputs, buildingAge: Number(e.target.value)})}
            className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            min="0"
            max="200"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Building Type
          </label>
          <select
            value={inputs.buildingType}
            onChange={(e) => setInputs({...inputs, buildingType: e.target.value})}
            className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="Office">Office</option>
            <option value="Retail">Retail</option>
            <option value="Warehouse">Warehouse</option>
            <option value="Hotel">Hotel</option>
            <option value="Hospital">Hospital</option>
            <option value="School">School</option>
            <option value="Restaurant">Restaurant</option>
            <option value="Other">Other</option>
          </select>
        </div>

        <div className="flex items-end">
          <button
            onClick={calculatePrediction}
            className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium"
          >
            Predict Energy Use
          </button>
        </div>
      </div>

      {result && (
        <div className="bg-gray-50 rounded-lg p-4 border">
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <p className="text-sm text-gray-600">Predicted Annual Use</p>
              <p className="text-2xl font-bold text-blue-600">
                {(result.predictedConsumption / 1000000).toFixed(1)}M kBtu
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Energy Intensity</p>
              <p className="text-2xl font-bold text-gray-800">
                {result.predictedIntensity.toFixed(0)} kBtu/sqft
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2 text-sm">
            <span className="text-gray-600">Confidence:</span>
            <span className={`px-2 py-1 rounded text-xs font-medium ${
              result.confidence === 'high' ? 'bg-green-100 text-green-800' :
              result.confidence === 'medium' ? 'bg-yellow-100 text-yellow-800' :
              'bg-red-100 text-red-800'
            }`}>
              {result.confidence.toUpperCase()}
            </span>
            <span className="text-gray-400 text-xs ml-2">
              (Based on similarity to training data)
            </span>
          </div>

          <p className="text-xs text-gray-500 mt-3">
            Note: This demo uses a simplified heuristic. The full model achieves 5.8% MAPE 
            on real Seattle buildings.
          </p>
        </div>
      )}
    </div>
  )
}