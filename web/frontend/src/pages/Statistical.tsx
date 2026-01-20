/**
 * Statistical Analysis Page
 * Author: Shuvam Banerji Seal's Team
 */

import { useState } from 'react';
import { BarChart2, TrendingUp, AlertTriangle, Sigma } from 'lucide-react';
import {
  StatCard,
  LoadingCard,
  ErrorPage,
} from '../components';
import { useData } from '../hooks';
import { formatNumber, formatPercentage } from '../utils/formatters';

const StatisticalPage = () => {
  const { data, isLoading, error, refetch } = useData<Record<string, unknown>>('statistical.json');
  const [selectedDataset, setSelectedDataset] = useState<string>('biometric');

  if (error) {
    return (
      <ErrorPage
        title="Failed to load statistical data"
        message={error.message}
        onRetry={refetch}
      />
    );
  }

  const datasets = data
    ? [...new Set(Object.keys(data).map((k) => k.split('_')[0]))]
    : ['biometric', 'demographic', 'enrollment'];

  const descriptive = data?.[`${selectedDataset}_descriptive`] as {
    record_count?: number;
    statistics?: Record<string, {
      mean: number;
      std: number;
      min: number;
      max: number;
      median: number;
      skewness: number;
      kurtosis: number;
    }>;
  } | undefined;

  const distribution = data?.[`${selectedDataset}_distribution`] as {
    normality_tests?: {
      shapiro_wilk?: { is_normal: boolean; p_value: number };
      dagostino_pearson?: { is_normal: boolean };
    };
    best_fit_distribution?: { distribution: string; aic: number };
    percentiles?: Record<string, number>;
  } | undefined;

  const correlation = data?.[`${selectedDataset}_correlation`] as {
    strong_correlations?: Array<{
      var1: string;
      var2: string;
      correlation: number;
      strength: string;
    }>;
    total_strong_correlations?: number;
  } | undefined;

  const hypothesis = data?.[`${selectedDataset}_hypothesis`] as {
    tests?: Record<string, {
      description: string;
      p_value: number;
      significant: boolean;
    }>;
  } | undefined;

  const outliers = data?.[`${selectedDataset}_outliers`] as {
    total_records?: number;
    methods?: Record<string, {
      outlier_count: number;
      outlier_pct: number;
    }>;
  } | undefined;

  const variance = data?.[`${selectedDataset}_variance`] as {
    overall_cv?: number;
    state_variance?: {
      mean_cv: number;
      highest_cv_states?: Array<{ state: string; cv: number }>;
    };
  } | undefined;

  // Stats for display
  const totalEnrollmentStats = descriptive?.statistics?.['total_enrollment'];
  const significantTests = hypothesis?.tests
    ? Object.values(hypothesis.tests).filter((t) => t.significant).length
    : 0;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
            Statistical Analysis
          </h1>
          <p className="text-slate-500 dark:text-slate-400">
            Descriptive statistics, hypothesis testing, and distribution analysis
          </p>
        </div>

        <select
          value={selectedDataset}
          onChange={(e) => setSelectedDataset(e.target.value)}
          className="px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
        >
          {datasets.map((d) => (
            <option key={d} value={d}>
              {d.charAt(0).toUpperCase() + d.slice(1)}
            </option>
          ))}
        </select>
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
            <StatCard
              title="Total Records"
              value={formatNumber(descriptive?.record_count || 0)}
              subtitle="Analyzed"
              icon={<BarChart2 className="w-5 h-5" />}
              color="blue"
            />
            <StatCard
              title="Mean Enrollment"
              value={formatNumber(totalEnrollmentStats?.mean || 0)}
              subtitle={`σ = ${formatNumber(totalEnrollmentStats?.std || 0)}`}
              icon={<Sigma className="w-5 h-5" />}
              color="green"
            />
            <StatCard
              title="Significant Tests"
              value={`${significantTests}/${Object.keys(hypothesis?.tests || {}).length}`}
              subtitle="Hypothesis tests"
              icon={<TrendingUp className="w-5 h-5" />}
              color="orange"
            />
            <StatCard
              title="Outliers (IQR)"
              value={formatPercentage(outliers?.methods?.iqr?.outlier_pct || 0)}
              subtitle={`${outliers?.methods?.iqr?.outlier_count || 0} records`}
              icon={<AlertTriangle className="w-5 h-5" />}
              color="red"
            />
          </>
        )}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Descriptive Statistics */}
        <div className="card p-6">
          <h3 className="font-semibold text-slate-900 dark:text-white mb-4">
            Descriptive Statistics
          </h3>
          {totalEnrollmentStats ? (
            <div className="grid grid-cols-2 gap-4">
              {[
                { label: 'Mean', value: formatNumber(totalEnrollmentStats.mean) },
                { label: 'Median', value: formatNumber(totalEnrollmentStats.median) },
                { label: 'Std Dev', value: formatNumber(totalEnrollmentStats.std) },
                { label: 'Min', value: formatNumber(totalEnrollmentStats.min) },
                { label: 'Max', value: formatNumber(totalEnrollmentStats.max) },
                { label: 'Skewness', value: totalEnrollmentStats.skewness?.toFixed(3) },
                { label: 'Kurtosis', value: totalEnrollmentStats.kurtosis?.toFixed(3) },
                { label: 'CV %', value: formatPercentage(variance?.overall_cv || 0) },
              ].map((stat) => (
                <div key={stat.label} className="p-3 rounded-lg bg-slate-50 dark:bg-slate-800">
                  <p className="text-xs text-slate-500 dark:text-slate-400">{stat.label}</p>
                  <p className="font-semibold text-slate-900 dark:text-white">{stat.value}</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-slate-500">No data available</p>
          )}
        </div>

        {/* Distribution Analysis */}
        <div className="card p-6">
          <h3 className="font-semibold text-slate-900 dark:text-white mb-4">
            Distribution Analysis
          </h3>
          <div className="space-y-4">
            <div className="p-4 rounded-lg bg-slate-50 dark:bg-slate-800">
              <p className="text-sm text-slate-500 dark:text-slate-400">
                Best Fit Distribution
              </p>
              <p className="font-semibold text-slate-900 dark:text-white capitalize">
                {distribution?.best_fit_distribution?.distribution || 'N/A'}
              </p>
              <p className="text-xs text-slate-500">
                AIC: {distribution?.best_fit_distribution?.aic?.toFixed(2) || 'N/A'}
              </p>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-400">Shapiro-Wilk Test</span>
              <span className={`badge ${distribution?.normality_tests?.shapiro_wilk?.is_normal ? 'badge-green' : 'badge-red'}`}>
                {distribution?.normality_tests?.shapiro_wilk?.is_normal ? 'Normal' : 'Non-normal'}
              </span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-400">D'Agostino-Pearson</span>
              <span className={`badge ${distribution?.normality_tests?.dagostino_pearson?.is_normal ? 'badge-green' : 'badge-red'}`}>
                {distribution?.normality_tests?.dagostino_pearson?.is_normal ? 'Normal' : 'Non-normal'}
              </span>
            </div>

            <div className="pt-2 border-t border-slate-200 dark:border-slate-700">
              <p className="text-sm text-slate-500 dark:text-slate-400 mb-2">Percentiles</p>
              <div className="grid grid-cols-3 gap-2 text-sm">
                {distribution?.percentiles && Object.entries(distribution.percentiles).slice(0, 6).map(([key, value]) => (
                  <div key={key} className="text-center">
                    <p className="text-slate-500">{key.toUpperCase()}</p>
                    <p className="font-medium text-slate-900 dark:text-white">{formatNumber(value as number)}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Strong Correlations */}
        <div className="card overflow-hidden">
          <div className="p-4 border-b border-slate-200 dark:border-slate-700">
            <h3 className="font-semibold text-slate-900 dark:text-white">
              Strong Correlations ({correlation?.total_strong_correlations || 0})
            </h3>
          </div>
          <div className="overflow-x-auto max-h-[300px]">
            <table className="table">
              <thead className="sticky top-0 bg-white dark:bg-slate-900">
                <tr>
                  <th>Variable 1</th>
                  <th>Variable 2</th>
                  <th>r</th>
                  <th>Strength</th>
                </tr>
              </thead>
              <tbody>
                {correlation?.strong_correlations?.slice(0, 10).map((c, i) => (
                  <tr key={i}>
                    <td className="text-xs">{c.var1}</td>
                    <td className="text-xs">{c.var2}</td>
                    <td className={c.correlation > 0 ? 'text-green-600' : 'text-red-600'}>
                      {c.correlation.toFixed(3)}
                    </td>
                    <td>
                      <span className={`badge ${c.strength === 'strong' ? 'badge-orange' : 'badge-blue'}`}>
                        {c.strength}
                      </span>
                    </td>
                  </tr>
                )) || (
                  <tr>
                    <td colSpan={4} className="text-center text-slate-500">No strong correlations found</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Hypothesis Tests */}
        <div className="card overflow-hidden">
          <div className="p-4 border-b border-slate-200 dark:border-slate-700">
            <h3 className="font-semibold text-slate-900 dark:text-white">
              Hypothesis Tests
            </h3>
          </div>
          <div className="p-4 space-y-3">
            {hypothesis?.tests && Object.entries(hypothesis.tests).map(([key, test]) => (
              <div
                key={key}
                className="flex items-start justify-between p-3 rounded-lg bg-slate-50 dark:bg-slate-800"
              >
                <div className="flex-1">
                  <p className="font-medium text-slate-900 dark:text-white text-sm">
                    {test.description}
                  </p>
                  <p className="text-xs text-slate-500">
                    p-value: {test.p_value?.toExponential(2)}
                  </p>
                </div>
                <span className={`badge ${test.significant ? 'badge-green' : 'badge-red'}`}>
                  {test.significant ? 'Significant' : 'Not Significant'}
                </span>
              </div>
            )) || (
              <p className="text-slate-500 text-center">No hypothesis tests available</p>
            )}
          </div>
        </div>
      </div>

      {/* Outlier Methods Comparison */}
      {outliers?.methods && (
        <div className="card p-6">
          <h3 className="font-semibold text-slate-900 dark:text-white mb-4">
            Outlier Detection Methods Comparison
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(outliers.methods).map(([method, stats]) => (
              <div
                key={method}
                className="p-4 rounded-lg bg-slate-50 dark:bg-slate-800 text-center"
              >
                <p className="text-sm text-slate-500 dark:text-slate-400 capitalize mb-1">
                  {method.replace('_', ' ')}
                </p>
                <p className="text-2xl font-bold text-slate-900 dark:text-white">
                  {formatPercentage(stats.outlier_pct)}
                </p>
                <p className="text-xs text-slate-500">
                  {formatNumber(stats.outlier_count)} records
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* High CV States */}
      {variance?.state_variance?.highest_cv_states && (
        <div className="card overflow-hidden">
          <div className="p-4 border-b border-slate-200 dark:border-slate-700">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-slate-900 dark:text-white">
                States with Highest Variance (CV)
              </h3>
              <span className="text-sm text-slate-500">
                Mean CV: {formatPercentage(variance.state_variance.mean_cv)}
              </span>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>State</th>
                  <th>Coefficient of Variation</th>
                </tr>
              </thead>
              <tbody>
                {variance.state_variance.highest_cv_states.slice(0, 10).map((s, i) => (
                  <tr key={s.state}>
                    <td className="font-medium">{i + 1}</td>
                    <td>{s.state}</td>
                    <td>{formatPercentage(s.cv)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default StatisticalPage;
