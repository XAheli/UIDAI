/**
 * ML Models Page
 * Author: Shuvam Banerji Seal's Team
 */

import { Brain, Server, AlertCircle, Terminal, Download, Play } from 'lucide-react';

const MLPage = () => {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
          Machine Learning Models
        </h1>
        <p className="text-slate-500 dark:text-slate-400">
          ML-based predictions and forecasting for Aadhaar enrollment data
        </p>
      </div>

      {/* Important Notice */}
      <div className="card p-6 border-orange-200 dark:border-orange-800 bg-orange-50 dark:bg-orange-900/10">
        <div className="flex items-start space-x-4">
          <div className="p-3 rounded-lg bg-orange-100 dark:bg-orange-900/30">
            <AlertCircle className="w-6 h-6 text-orange-600 dark:text-orange-400" />
          </div>
          <div>
            <h3 className="font-semibold text-orange-800 dark:text-orange-200 mb-2">
              Local Execution Required
            </h3>
            <p className="text-sm text-orange-700 dark:text-orange-300 mb-4">
              Machine learning model inference cannot run on GitHub Pages as it only supports
              static content. You need to run the models locally using Docker and then upload
              the prediction results.
            </p>
            <div className="flex flex-wrap gap-3">
              <code className="px-3 py-1.5 rounded bg-orange-100 dark:bg-orange-900/30 text-orange-800 dark:text-orange-200 text-sm">
                docker-compose up ml-inference
              </code>
            </div>
          </div>
        </div>
      </div>

      {/* Available Models */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[
          {
            name: 'Enrollment Forecaster',
            description: 'ARIMA/Prophet-based time series forecasting for future enrollment predictions',
            status: 'available',
            metrics: { accuracy: '85%', horizon: '30 days' },
          },
          {
            name: 'Regional Demand Predictor',
            description: 'Random Forest classifier for predicting high-demand regions',
            status: 'available',
            metrics: { accuracy: '82%', features: '15' },
          },
          {
            name: 'Anomaly Detector',
            description: 'Isolation Forest for detecting unusual enrollment patterns',
            status: 'available',
            metrics: { precision: '78%', recall: '85%' },
          },
          {
            name: 'Age Group Classifier',
            description: 'Multi-class classification for age distribution predictions',
            status: 'development',
            metrics: { status: 'In Progress' },
          },
          {
            name: 'Resource Optimizer',
            description: 'Optimization model for resource allocation across centers',
            status: 'planned',
            metrics: { status: 'Planned' },
          },
          {
            name: 'Trend Analyzer',
            description: 'Deep learning model for complex pattern recognition',
            status: 'planned',
            metrics: { status: 'Planned' },
          },
        ].map((model) => (
          <div key={model.name} className="card p-6">
            <div className="flex items-start justify-between mb-4">
              <div className="p-2 rounded-lg bg-purple-100 dark:bg-purple-900/30">
                <Brain className="w-5 h-5 text-purple-600 dark:text-purple-400" />
              </div>
              <span
                className={`badge ${
                  model.status === 'available'
                    ? 'badge-green'
                    : model.status === 'development'
                    ? 'badge-orange'
                    : 'badge-blue'
                }`}
              >
                {model.status === 'available' ? 'Available' : model.status === 'development' ? 'In Dev' : 'Planned'}
              </span>
            </div>
            <h3 className="font-semibold text-slate-900 dark:text-white mb-2">
              {model.name}
            </h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
              {model.description}
            </p>
            <div className="flex flex-wrap gap-2">
              {Object.entries(model.metrics).map(([key, value]) => (
                <span
                  key={key}
                  className="px-2 py-1 rounded bg-slate-100 dark:bg-slate-800 text-xs text-slate-600 dark:text-slate-400"
                >
                  {key}: {value}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Docker Instructions */}
      <div className="card p-6">
        <div className="flex items-center space-x-3 mb-4">
          <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900/30">
            <Server className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          </div>
          <h3 className="font-semibold text-slate-900 dark:text-white">
            Running Models Locally
          </h3>
        </div>
        
        <div className="space-y-4">
          <div className="p-4 rounded-lg bg-slate-900 dark:bg-slate-950 text-slate-100 font-mono text-sm overflow-x-auto">
            <div className="flex items-center space-x-2 text-slate-400 mb-2">
              <Terminal className="w-4 h-4" />
              <span>Terminal</span>
            </div>
            <pre className="whitespace-pre-wrap">
{`# Clone the repository (if not already done)
git clone https://github.com/YOUR_USERNAME/UIDAI_hackathon.git
cd UIDAI_hackathon

# Pull LFS files
git lfs pull

# Build and run ML inference container
docker-compose up ml-inference

# Or run specific model
docker-compose run ml-inference python -m analysis.codes.ml_models.run_forecasting

# Export results for web
docker-compose run ml-inference python -m analysis.codes.ml_models.export_for_web`}
            </pre>
          </div>
          
          <div className="flex flex-wrap gap-3">
            <button className="btn btn-primary flex items-center space-x-2">
              <Play className="w-4 h-4" />
              <span>View Sample Output</span>
            </button>
            <button className="btn btn-outline flex items-center space-x-2">
              <Download className="w-4 h-4" />
              <span>Download Pre-computed Results</span>
            </button>
          </div>
        </div>
      </div>

      {/* Model Architecture */}
      <div className="card p-6">
        <h3 className="font-semibold text-slate-900 dark:text-white mb-4">
          ML Pipeline Architecture
        </h3>
        <div className="space-y-4">
          <div className="flex items-center space-x-4">
            <div className="w-32 text-sm text-slate-500 dark:text-slate-400">Data Input</div>
            <div className="flex-1 h-2 rounded bg-blue-500" />
            <div className="text-sm text-slate-900 dark:text-white">6.1M Records</div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="w-32 text-sm text-slate-500 dark:text-slate-400">Preprocessing</div>
            <div className="flex-1 h-2 rounded bg-green-500" />
            <div className="text-sm text-slate-900 dark:text-white">Feature Engineering</div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="w-32 text-sm text-slate-500 dark:text-slate-400">Model Training</div>
            <div className="flex-1 h-2 rounded bg-orange-500" />
            <div className="text-sm text-slate-900 dark:text-white">Multi-threaded</div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="w-32 text-sm text-slate-500 dark:text-slate-400">Inference</div>
            <div className="flex-1 h-2 rounded bg-purple-500" />
            <div className="text-sm text-slate-900 dark:text-white">Batch Processing</div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="w-32 text-sm text-slate-500 dark:text-slate-400">Export</div>
            <div className="flex-1 h-2 rounded bg-pink-500" />
            <div className="text-sm text-slate-900 dark:text-white">JSON/CSV</div>
          </div>
        </div>
      </div>

      {/* Author */}
      <div className="text-center text-slate-500 dark:text-slate-400 text-sm">
        <p>
          ML Models developed by <strong>Shuvam Banerji Seal's Team</strong>
        </p>
      </div>
    </div>
  );
};

export default MLPage;
