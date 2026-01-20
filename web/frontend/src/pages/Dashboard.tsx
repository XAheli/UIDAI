/**
 * Dashboard Page - Comprehensive Analysis Display
 * Author: Shuvam Banerji Seal's Team
 */

import { useEffect, useState } from 'react';
import {
  Users,
  MapPin,
  Calendar,
  TrendingUp,
  BarChart3,
  Database,
  Clock,
  Cpu,
  Target,
  Layers,
  AlertTriangle,
  CheckCircle2,
  Brain,
  GitBranch,
  Zap,
  PieChart,
  LineChart,
} from 'lucide-react';
import {
  GradientStatCard,
  LoadingCard,
  ErrorDisplay,
  ChartContainer,
  SimplePieChart,
  SimpleBarChart,
  SimpleAreaChart,
} from '../components';
import { useData } from '../hooks';
import { formatTimestamp } from '../utils/formatters';
import type { AnalysisSummary } from '../utils/dataFetcher';

interface MLModelResult {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  r2?: number;
  rmse?: number;
  silhouette_score?: number;
  n_anomalies?: number;
  anomaly_percentage?: number;
}

interface MLResults {
  generated_at: string;
  author: string;
  sample_size: number;
  datasets: Record<string, {
    n_records: number;
    n_features: number;
    classification?: {
      target: string;
      n_classes: number;
      models: Record<string, MLModelResult>;
      best_model?: { name: string; accuracy: number };
    };
    regression?: {
      target: string;
      models: Record<string, MLModelResult>;
      best_model?: { name: string; r2: number };
    };
    clustering?: {
      models: Record<string, MLModelResult>;
      best_model?: { name: string; silhouette_score: number };
    };
    anomaly_detection?: {
      models: Record<string, MLModelResult>;
    };
  }>;
  summary: {
    total_models_trained: number;
    datasets_processed: string[];
  };
}

const Dashboard = () => {
  const { data: summary, isLoading, error, refetch } = useData<AnalysisSummary>('analysis_summary.json');
  const { data: timeSeriesData } = useData<Record<string, unknown>>('time_series.json');
  const { data: geographicData } = useData<Record<string, unknown>>('geographic.json');
  const { data: mlResults } = useData<MLResults>('ml_results.json');

  // Chart data states
  const [enrollmentTrend, setEnrollmentTrend] = useState<Array<{name: string; value: number}>>([]);
  const [stateDistribution, setStateDistribution] = useState<Array<{name: string; value: number}>>([]);
  const [mlModelComparison, setMLModelComparison] = useState<Array<{name: string; accuracy: number}>>([]);
  const [datasetStats, setDatasetStats] = useState<Array<{name: string; records: number; features: number}>>([]);
  const [regressionComparison, setRegressionComparison] = useState<Array<{name: string; r2: number}>>([]);
  const [clusteringComparison, setClusteringComparison] = useState<Array<{name: string; silhouette: number}>>([]);

  useEffect(() => {
    // Process time series data
    if (timeSeriesData) {
      const trends = Object.values(timeSeriesData).find(
        (v: unknown) => typeof v === 'object' && v !== null && 'daily_data' in (v as object)
      ) as { daily_data?: Array<{date: string; total_enrollment: number}> } | undefined;
      
      if (trends?.daily_data) {
        const processedData = trends.daily_data.slice(-30).map((d) => ({
          name: new Date(d.date).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }),
          value: d.total_enrollment,
        }));
        setEnrollmentTrend(processedData);
      }
    }

    // Process geographic data
    if (geographicData) {
      const stateData = Object.values(geographicData).find(
        (v: unknown) => typeof v === 'object' && v !== null && 'top_10_states' in (v as object)
      ) as { top_10_states?: Array<{state: string; total_enrollment: number}> } | undefined;
      
      if (stateData?.top_10_states) {
        const processedData = stateData.top_10_states.slice(0, 8).map((s) => ({
          name: s.state.length > 12 ? s.state.slice(0, 12) + '...' : s.state,
          value: s.total_enrollment,
        }));
        setStateDistribution(processedData);
      }
    }

    // Process ML results
    if (mlResults?.datasets) {
      // Get classification model comparison from first dataset
      const firstDataset = Object.values(mlResults.datasets)[0];
      if (firstDataset?.classification?.models) {
        const modelData = Object.entries(firstDataset.classification.models)
          .filter(([_, v]) => v.accuracy !== undefined)
          .map(([name, v]) => ({
            name: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
            accuracy: Math.round((v.accuracy || 0) * 100),
          }))
          .sort((a, b) => b.accuracy - a.accuracy)
          .slice(0, 10);
        setMLModelComparison(modelData);
      }

      // Get regression comparison
      if (firstDataset?.regression?.models) {
        const regData = Object.entries(firstDataset.regression.models)
          .filter(([_, v]) => v.r2 !== undefined && v.r2 > 0 && v.r2 <= 1)
          .map(([name, v]) => ({
            name: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
            r2: Math.round((v.r2 || 0) * 100),
          }))
          .sort((a, b) => b.r2 - a.r2)
          .slice(0, 10);
        setRegressionComparison(regData);
      }

      // Get clustering comparison
      if (firstDataset?.clustering?.models) {
        const clusterData = Object.entries(firstDataset.clustering.models)
          .filter(([_, v]) => v.silhouette_score !== undefined)
          .map(([name, v]) => ({
            name: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
            silhouette: Math.round((v.silhouette_score || 0) * 100),
          }))
          .sort((a, b) => b.silhouette - a.silhouette);
        setClusteringComparison(clusterData);
      }

      // Dataset statistics
      const dsStats = Object.entries(mlResults.datasets).map(([name, ds]) => ({
        name: name.charAt(0).toUpperCase() + name.slice(1),
        records: ds.n_records,
        features: ds.n_features,
      }));
      setDatasetStats(dsStats);
    }
  }, [timeSeriesData, geographicData, mlResults]);

  if (error) {
    return (
      <div className="p-6">
        <ErrorDisplay
          title="Failed to load dashboard"
          message={error.message}
          onRetry={refetch}
        />
      </div>
    );
  }

  // Calculate totals
  const totalModels = mlResults?.summary?.total_models_trained || 0;
  const datasetsProcessed = mlResults?.summary?.datasets_processed?.length || 0;

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-2xl gradient-bg p-8 text-white">
        <div className="absolute inset-0 bg-black/10" />
        <div className="relative z-10">
          <h1 className="text-3xl md:text-4xl font-bold mb-2">
            UIDAI Analysis Dashboard
          </h1>
          <p className="text-white/80 text-lg mb-4">
            Comprehensive analysis of Aadhaar enrollment data with {totalModels} ML models trained
          </p>
          <div className="flex flex-wrap items-center gap-4 text-sm text-white/70">
            <span className="flex items-center">
              <Clock className="w-4 h-4 mr-1" />
              Last updated: {summary ? formatTimestamp(summary.generated_at) : 'Loading...'}
            </span>
            <span className="flex items-center">
              <Database className="w-4 h-4 mr-1" />
              {summary?.total_analyses || 0} analyses completed
            </span>
            <span className="flex items-center">
              <Brain className="w-4 h-4 mr-1" />
              {totalModels} ML models trained
            </span>
            <span className="flex items-center">
              <Layers className="w-4 h-4 mr-1" />
              {datasetsProcessed} datasets processed
            </span>
          </div>
        </div>
        <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full -translate-y-1/2 translate-x-1/2" />
        <div className="absolute bottom-0 left-0 w-48 h-48 bg-white/5 rounded-full translate-y-1/2 -translate-x-1/2" />
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {isLoading ? (
          <>
            <LoadingCard />
            <LoadingCard />
            <LoadingCard />
            <LoadingCard />
          </>
        ) : (
          <>
            <GradientStatCard
              title="Total Records Analyzed"
              value="6.1M+"
              subtitle="Across all datasets"
              icon={<Users className="w-6 h-6 text-white" />}
              color="green"
            />
            <GradientStatCard
              title="States & UTs Covered"
              value="36"
              subtitle="Complete coverage"
              icon={<MapPin className="w-6 h-6 text-white" />}
              color="green"
            />
            <GradientStatCard
              title="ML Models Trained"
              value={totalModels.toString()}
              subtitle="Classification, Regression, Clustering"
              icon={<Brain className="w-6 h-6 text-white" />}
              color="orange"
            />
            <GradientStatCard
              title="Analysis Success"
              value={`${summary?.success_rate || 100}%`}
              subtitle="All analyses completed"
              icon={<CheckCircle2 className="w-6 h-6 text-white" />}
              color="purple"
            />
          </>
        )}
      </div>

      {/* ML Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card p-4 border-l-4 border-coffee-500">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-coffee-100 dark:bg-coffee-900/30">
              <Target className="w-5 h-5 text-coffee-600 dark:text-coffee-400" />
            </div>
            <div>
              <p className="text-sm text-slate-500 dark:text-slate-400">Classification Models</p>
              <p className="text-xl font-bold text-slate-900 dark:text-white">13</p>
            </div>
          </div>
        </div>
        <div className="card p-4 border-l-4 border-sage-500">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-sage-100 dark:bg-sage-900/30">
              <LineChart className="w-5 h-5 text-sage-600 dark:text-sage-400" />
            </div>
            <div>
              <p className="text-sm text-slate-500 dark:text-slate-400">Regression Models</p>
              <p className="text-xl font-bold text-slate-900 dark:text-white">16</p>
            </div>
          </div>
        </div>
        <div className="card p-4 border-l-4 border-blue-500">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-blue-100 dark:bg-blue-900/30">
              <GitBranch className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <p className="text-sm text-slate-500 dark:text-slate-400">Clustering Configs</p>
              <p className="text-xl font-bold text-slate-900 dark:text-white">7</p>
            </div>
          </div>
        </div>
        <div className="card p-4 border-l-4 border-orange-500">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-orange-100 dark:bg-orange-900/30">
              <AlertTriangle className="w-5 h-5 text-orange-600 dark:text-orange-400" />
            </div>
            <div>
              <p className="text-sm text-slate-500 dark:text-slate-400">Anomaly Detectors</p>
              <p className="text-xl font-bold text-slate-900 dark:text-white">3</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartContainer
          title="Classification Model Performance"
          subtitle="Accuracy comparison across models"
        >
          {mlModelComparison.length > 0 ? (
            <SimpleBarChart
              data={mlModelComparison}
              dataKeys={['accuracy']}
              xAxisKey="name"
              height={300}
              layout="vertical"
            />
          ) : (
            <div className="h-[300px] flex items-center justify-center text-slate-500 dark:text-slate-400">
              <div className="text-center">
                <Cpu className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>Loading ML results...</p>
              </div>
            </div>
          )}
        </ChartContainer>

        <ChartContainer
          title="State-wise Distribution"
          subtitle="Top 8 states by enrollment"
        >
          {stateDistribution.length > 0 ? (
            <SimplePieChart
              data={stateDistribution}
              height={300}
              innerRadius={60}
              outerRadius={100}
            />
          ) : (
            <div className="h-[300px] flex items-center justify-center text-slate-500 dark:text-slate-400">
              <div className="text-center">
                <PieChart className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>Run analysis to see distribution</p>
              </div>
            </div>
          )}
        </ChartContainer>
      </div>

      {/* Regression and Clustering Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartContainer
          title="Regression Model Performance"
          subtitle="R² score comparison (higher is better)"
        >
          {regressionComparison.length > 0 ? (
            <SimpleBarChart
              data={regressionComparison}
              dataKeys={['r2']}
              xAxisKey="name"
              height={280}
              layout="vertical"
            />
          ) : (
            <div className="h-[280px] flex items-center justify-center text-slate-500 dark:text-slate-400">
              Loading regression results...
            </div>
          )}
        </ChartContainer>

        <ChartContainer
          title="Clustering Performance"
          subtitle="Silhouette score comparison"
        >
          {clusteringComparison.length > 0 ? (
            <SimpleBarChart
              data={clusteringComparison}
              dataKeys={['silhouette']}
              xAxisKey="name"
              height={280}
            />
          ) : (
            <div className="h-[280px] flex items-center justify-center text-slate-500 dark:text-slate-400">
              Loading clustering results...
            </div>
          )}
        </ChartContainer>
      </div>

      {/* Enrollment Trends */}
      <ChartContainer
        title="Enrollment Trends Over Time"
        subtitle="Daily enrollment patterns (last 30 days)"
      >
        {enrollmentTrend.length > 0 ? (
          <SimpleAreaChart
            data={enrollmentTrend}
            dataKeys={['value']}
            height={300}
          />
        ) : (
          <div className="h-[300px] flex items-center justify-center text-slate-500 dark:text-slate-400">
            Run analysis to see trends
          </div>
        )}
      </ChartContainer>

      {/* Dataset Stats and Analysis Status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4 flex items-center">
            <Database className="w-5 h-5 mr-2 text-coffee-500" />
            Dataset Overview
          </h3>
          <div className="space-y-4">
            {datasetStats.length > 0 ? (
              datasetStats.map((ds) => (
                <div
                  key={ds.name}
                  className="flex items-center justify-between p-4 rounded-lg bg-coffee-50 dark:bg-coffee-900/20"
                >
                  <div className="flex items-center space-x-3">
                    <div className="p-2 rounded-lg bg-coffee-100 dark:bg-coffee-800/30">
                      <Layers className="w-4 h-4 text-coffee-600 dark:text-coffee-400" />
                    </div>
                    <div>
                      <span className="font-medium text-slate-700 dark:text-slate-300">
                        {ds.name}
                      </span>
                      <p className="text-sm text-slate-500 dark:text-slate-400">
                        {ds.features} features extracted
                      </p>
                    </div>
                  </div>
                  <span className="badge badge-green">
                    {ds.records.toLocaleString()} records
                  </span>
                </div>
              ))
            ) : (
              [
                { name: 'Biometric', records: '3.5M', icon: Users },
                { name: 'Demographic', records: '1.6M', icon: BarChart3 },
                { name: 'Enrollment', records: '982K', icon: Calendar },
              ].map((dataset) => (
                <div
                  key={dataset.name}
                  className="flex items-center justify-between p-4 rounded-lg bg-coffee-50 dark:bg-coffee-900/20"
                >
                  <div className="flex items-center space-x-3">
                    <div className="p-2 rounded-lg bg-coffee-100 dark:bg-coffee-800/30">
                      <dataset.icon className="w-4 h-4 text-coffee-600 dark:text-coffee-400" />
                    </div>
                    <span className="font-medium text-slate-700 dark:text-slate-300">
                      {dataset.name}
                    </span>
                  </div>
                  <span className="text-slate-600 dark:text-slate-400">
                    {dataset.records} records
                  </span>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="card p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4 flex items-center">
            <CheckCircle2 className="w-5 h-5 mr-2 text-green-500" />
            Completed Analyses
          </h3>
          <div className="space-y-3">
            {summary?.analyses_completed?.map((analysis) => (
              <div
                key={analysis.type}
                className="flex items-center justify-between p-3 rounded-lg bg-green-50 dark:bg-green-900/20"
              >
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 rounded-full bg-green-500" />
                  <span className="font-medium text-slate-700 dark:text-slate-300 capitalize">
                    {analysis.type.replace('_', ' ')}
                  </span>
                </div>
                <span className="badge badge-green">
                  {analysis.result_count} results
                </span>
              </div>
            )) || (
              <p className="text-slate-500 dark:text-slate-400">Loading analyses...</p>
            )}
            {/* Add ML training status */}
            <div className="flex items-center justify-between p-3 rounded-lg bg-coffee-50 dark:bg-coffee-900/20">
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 rounded-full bg-coffee-500" />
                <span className="font-medium text-slate-700 dark:text-slate-300">
                  Machine Learning
                </span>
              </div>
              <span className="badge badge-green">
                {totalModels} models
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Best Models Summary */}
      {mlResults?.datasets && (
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4 flex items-center">
            <Zap className="w-5 h-5 mr-2 text-amber-500" />
            Best Performing Models by Dataset
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(mlResults.datasets).map(([name, ds]) => (
              <div key={name} className="p-4 rounded-lg bg-slate-50 dark:bg-slate-800/50">
                <h4 className="font-semibold text-coffee-600 dark:text-coffee-400 mb-3 capitalize">
                  {name} Dataset
                </h4>
                <div className="space-y-2 text-sm">
                  {ds.classification?.best_model && (
                    <div className="flex justify-between">
                      <span className="text-slate-500 dark:text-slate-400">Classification:</span>
                      <span className="font-medium text-slate-700 dark:text-slate-300">
                        {ds.classification.best_model.name.replace(/_/g, ' ')}
                        <span className="ml-1 text-green-500">
                          ({(ds.classification.best_model.accuracy * 100).toFixed(1)}%)
                        </span>
                      </span>
                    </div>
                  )}
                  {ds.regression?.best_model && (
                    <div className="flex justify-between">
                      <span className="text-slate-500 dark:text-slate-400">Regression:</span>
                      <span className="font-medium text-slate-700 dark:text-slate-300">
                        {ds.regression.best_model.name.replace(/_/g, ' ')}
                        <span className="ml-1 text-blue-500">
                          (R²: {(ds.regression.best_model.r2 * 100).toFixed(1)}%)
                        </span>
                      </span>
                    </div>
                  )}
                  {ds.clustering?.best_model && (
                    <div className="flex justify-between">
                      <span className="text-slate-500 dark:text-slate-400">Clustering:</span>
                      <span className="font-medium text-slate-700 dark:text-slate-300">
                        {ds.clustering.best_model.name.replace(/_/g, ' ')}
                        <span className="ml-1 text-purple-500">
                          (Sil: {(ds.clustering.best_model.silhouette_score * 100).toFixed(0)}%)
                        </span>
                      </span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card p-6 hover:shadow-lg transition-shadow">
          <div className="flex items-center space-x-3 mb-3">
            <div className="p-2 rounded-lg bg-coffee-100 dark:bg-coffee-900/30">
              <TrendingUp className="w-5 h-5 text-coffee-600 dark:text-coffee-400" />
            </div>
            <h4 className="font-semibold text-slate-900 dark:text-white">
              Time Series Analysis
            </h4>
          </div>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Analyze enrollment trends over time, detect seasonality patterns,
            and identify anomalies in the data with ARIMA forecasting.
          </p>
        </div>

        <div className="card p-6 hover:shadow-lg transition-shadow">
          <div className="flex items-center space-x-3 mb-3">
            <div className="p-2 rounded-lg bg-sage-100 dark:bg-sage-900/30">
              <MapPin className="w-5 h-5 text-sage-600 dark:text-sage-400" />
            </div>
            <h4 className="font-semibold text-slate-900 dark:text-white">
              Geographic Analysis
            </h4>
          </div>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Explore state-wise and district-wise enrollment patterns,
            regional comparisons, and geographic clustering insights.
          </p>
        </div>

        <div className="card p-6 hover:shadow-lg transition-shadow">
          <div className="flex items-center space-x-3 mb-3">
            <div className="p-2 rounded-lg bg-orange-100 dark:bg-orange-900/30">
              <Users className="w-5 h-5 text-orange-600 dark:text-orange-400" />
            </div>
            <h4 className="font-semibold text-slate-900 dark:text-white">
              Demographic Analysis
            </h4>
          </div>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Understand age group distributions, population correlations,
            and demographic trends in enrollment data across India.
          </p>
        </div>
      </div>

      {/* ML Summary Box */}
      <div className="card p-6 border-coffee-200 dark:border-coffee-800 bg-gradient-to-r from-coffee-50 to-sage-50 dark:from-coffee-900/20 dark:to-sage-900/20">
        <div className="flex items-start space-x-4">
          <div className="p-3 rounded-lg bg-coffee-100 dark:bg-coffee-900/30">
            <Brain className="w-6 h-6 text-coffee-600 dark:text-coffee-400" />
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-coffee-800 dark:text-coffee-200 mb-1">
              Machine Learning Summary
            </h3>
            <p className="text-sm text-coffee-700 dark:text-coffee-300 mb-3">
              {totalModels} models trained across {datasetsProcessed} datasets including:
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              <div className="p-2 bg-white/50 dark:bg-black/20 rounded">
                <span className="font-semibold">Classification:</span> Logistic Regression, Random Forest, XGBoost, SVM, KNN, etc.
              </div>
              <div className="p-2 bg-white/50 dark:bg-black/20 rounded">
                <span className="font-semibold">Regression:</span> Linear, Ridge, Lasso, ElasticNet, Tree-based models
              </div>
              <div className="p-2 bg-white/50 dark:bg-black/20 rounded">
                <span className="font-semibold">Clustering:</span> K-Means, GMM, Agglomerative
              </div>
              <div className="p-2 bg-white/50 dark:bg-black/20 rounded">
                <span className="font-semibold">Anomaly:</span> Isolation Forest, LOF, Elliptic Envelope
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Author Attribution */}
      <div className="text-center text-slate-500 dark:text-slate-400 text-sm py-4">
        <p>
          Built with ❤️ by <strong className="text-coffee-600 dark:text-coffee-400">Shuvam Banerji Seal, Alok Mishra, Aheli Poddar</strong> for
          UIDAI Hackathon
        </p>
      </div>
    </div>
  );
};

export default Dashboard;
