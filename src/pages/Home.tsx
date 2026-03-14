import { Link } from 'react-router-dom'

export default function Home() {
  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <section className="text-center py-16 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl">
        <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
          Data Science <span className="text-primary">Portfolio</span>
        </h1>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto mb-8">
          Full-stack data science projects built with React, Python, and modern ML techniques. 
          Deployed on GitHub Pages with automated CI/CD.
        </p>
        <div className="flex justify-center gap-4">
          <Link to="/projects" className="btn-primary text-lg">
            View Projects
          </Link>
          <a 
            href="https://github.com/channreaksmey" 
            target="_blank" 
            rel="noopener noreferrer"
            className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors text-lg"
          >
            GitHub Profile
          </a>
        </div>
      </section>

      {/* Featured Skills */}
      <section>
        <h2 className="text-2xl font-bold mb-6">Tech Stack</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            'React + TypeScript',
            'Python + Pandas',
            'Scikit-Learn',
            'TensorFlow.js',
            'Tailwind CSS',
            'GitHub Actions',
            'D3.js / Plotly',
            'GitHub Pages',
          ].map((skill) => (
            <div key={skill} className="bg-white p-4 rounded-lg shadow-sm border text-center font-medium">
              {skill}
            </div>
          ))}
        </div>
      </section>

      {/* Recent Projects Preview */}
      <section>
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold">Featured Projects</h2>
          <Link to="/projects" className="text-primary hover:underline">
            View all →
          </Link>
        </div>
        <div className="grid md:grid-cols-2 gap-6">
          <ProjectCard 
            title="Energy Consumption Predictor"
            description="Predict building energy usage using machine learning. Features interactive visualizations and real-time predictions."
            tags={['Python', 'Scikit-Learn', 'React']}
          />
          <ProjectCard 
            title="Customer Segmentation"
            description="Unsupervised learning analysis of e-commerce behavior with interactive clustering visualization."
            tags={['KMeans', 'Pandas', 'D3.js']}
          />
        </div>
      </section>
    </div>
  )
}

// Helper component for project cards
function ProjectCard({ title, description, tags }: { title: string, description: string, tags: string[] }) {
  return (
    <div className="bg-white rounded-xl shadow-sm border p-6 hover:shadow-md transition-shadow">
      <h3 className="text-xl font-bold mb-2">{title}</h3>
      <p className="text-gray-600 mb-4">{description}</p>
      <div className="flex flex-wrap gap-2">
        {tags.map(tag => (
          <span key={tag} className="px-3 py-1 bg-blue-50 text-blue-700 rounded-full text-sm font-medium">
            {tag}
          </span>
        ))}
      </div>
    </div>
  )
}