export default function About() {
  return (
    <div className="max-w-2xl">
      <h1 className="text-3xl font-bold mb-6">About This Portfolio</h1>
      
      <div className="prose prose-lg text-gray-600 space-y-4">
        <p>
          This portfolio demonstrates full-stack data science capabilities using a 
          modern, serverless architecture deployed entirely on GitHub Pages.
        </p>
        
        <h3 className="text-xl font-semibold text-gray-900 mt-6">Architecture</h3>
        <ul className="list-disc pl-5 space-y-2">
          <li>
            <strong>Frontend:</strong> React + TypeScript + Tailwind CSS, built with Vite
          </li>
          <li>
            <strong>Data Processing:</strong> Python scripts run in GitHub Actions at build time
          </li>
          <li>
            <strong>ML Inference:</strong> Models converted to TensorFlow.js or ONNX for browser execution
          </li>
          <li>
            <strong>Deployment:</strong> Automated CI/CD via GitHub Actions to GitHub Pages
          </li>
        </ul>

        <h3 className="text-xl font-semibold text-gray-900 mt-6">Why GitHub Pages?</h3>
        <p>
          Traditional data science portfolios require expensive backend servers. 
          This approach pre-computes predictions and runs lightweight models directly 
          in the browser, making it:
        </p>
        <ul className="list-disc pl-5 space-y-2">
          <li>Completely free to host</li>
          <li>Fast (CDN-distributed)</li>
          <li>Automatically backed up via Git</li>
          <li>Easy to maintain with version control</li>
        </ul>
      </div>
    </div>
  )
}