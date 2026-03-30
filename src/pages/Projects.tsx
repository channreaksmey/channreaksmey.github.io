import { useState } from 'react'

// This will later come from our Python-generated JSON
const PROJECTS = [
  {
    id: 1,
    title: "Building Energy Consumption Predictor",
    description: "Machine learning model to predict energy usage in commercial buildings based on size, age, and type. Includes interactive demo where you can input building parameters.",
    status: "Complete",
    tags: ["Regression", "Python", "React", "Energy Data"],
    githubUrl: "https://github.com/channreaksmey/channreaksmey.github.io/tree/main/projects/01_energy",
    demoUrl: "#/projects/energy",
    metrics: { rmse: 146851, r2: 0.997 }
  },
  {
    id: 2,
    title: "E-Commerce Customer Segmentation",
    description: "K-Means clustering on 95,000+ Brazilian e-commerce orders. RFM analysis identifies 5 customer segments for targeted marketing.",
    status: "Complete",
    tags: ["Clustering", "K-Means", "RFM Analysis", "Python", "3D Visualization"],
    githubUrl: "https://github.com/yourusername/yourusername.github.io/tree/main/projects/02_segmentation",
    demoUrl: "#/projects/segmentation",
    metrics: { clusters: 5, customers: 95000 }
  },
  {
    id: 3,
    title: "NLP Text Classifier",
    description: "Browser-based text classification using TensorFlow.js. Classifies technical questions into categories without backend server.",
    status: "Planned",
    tags: ["NLP", "TensorFlow.js", "Client-side ML", "Classification"],
    githubUrl: "#",
    demoUrl: "#",
    metrics: { accuracy: 0.89 }
  }
]

export default function Projects() {
  const [filter, setFilter] = useState('All')
  
  const categories = ['All', 'Regression', 'Clustering', 'NLP', 'Classification']
  
  const filteredProjects = filter === 'All' 
    ? PROJECTS 
    : PROJECTS.filter(p => p.tags.includes(filter))

  return (
    <div>
      <h1 className="text-3xl font-bold mb-2">Projects</h1>
      <p className="text-gray-600 mb-8">
        End-to-end data science projects with interactive demos. 
        All models are either pre-computed or run directly in your browser.
      </p>

      {/* Filter Buttons */}
      <div className="flex flex-wrap gap-2 mb-8">
        {categories.map(cat => (
          <button
            key={cat}
            onClick={() => setFilter(cat)}
            className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
              filter === cat
                ? 'bg-primary text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Project Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredProjects.map(project => (
          <div key={project.id} className="bg-white rounded-xl shadow-sm border overflow-hidden flex flex-col">
            <div className="p-6 flex-grow">
              <div className="flex justify-between items-start mb-4">
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  project.status === 'Complete' ? 'bg-green-100 text-green-800' :
                  project.status === 'In Progress' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-gray-100 text-gray-800'
                }`}>
                  {project.status}
                </span>
              </div>
              
              <h3 className="text-xl font-bold mb-2">{project.title}</h3>
              <p className="text-gray-600 text-sm mb-4">{project.description}</p>
              
              <div className="flex flex-wrap gap-2 mb-4">
                {project.tags.map(tag => (
                  <span key={tag} className="px-2 py-1 bg-gray-100 text-gray-600 rounded text-xs">
                    {tag}
                  </span>
                ))}
              </div>

              {/* Metrics */}
              {Object.entries(project.metrics).map(([key, value]) => (
                <div key={key} className="flex justify-between text-sm border-t pt-2 mt-2">
                  <span className="text-gray-500 uppercase">{key}</span>
                  <span className="font-mono font-medium">{value}</span>
                </div>
              ))}
            </div>
            
            <div className="p-4 bg-gray-50 border-t flex gap-3">
              <a href={project.githubUrl} className="flex-1 text-center py-2 border rounded hover:bg-white transition-colors text-sm">
                GitHub Repo
              </a>
              <a href={project.demoUrl} className="flex-1 text-center py-2 bg-primary text-white rounded hover:bg-blue-600 transition-colors text-sm">
                Live Demo
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}