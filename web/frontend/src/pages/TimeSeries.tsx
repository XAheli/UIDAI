/**
 * Time Series Page
 * Author: Shuvam Banerji Seal's Team
 */

import { useState } from 'react';
import {
  Calendar,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  BarChart2,
} from 'lucide-react';
import {
  StatCard,
  LoadingCard,
  LoadingChart,
  ErrorPage,
  ChartContainer,
  SimpleLineChart,
  SimpleBarChart,
} from '../components';
import { useData } from '../hooks';
import { formatNumber } from '../utils/formatters';

interface TimeSeriesResult {
  dataset_name: string;
  date_range?: {
    start: string;
    end: string;
    total_days: number;
  };
  enrollment_stats?: {
    total: number;
    daily_mean: number;
    daily_std: number;
    daily_max: number;
    daily_min: number;
  };
  trend?: {
    slope: number;
    r_squared: number;
    direction: string;
  };
  daily_data?: Array<{
    date: string;
    total_enrollment: number;
    ma_7?: number;
    ma_30?: number;
  }>;
  day_of_week?: {
    stats: Array<{
      day_name: string;
      mean: number;
    }>;
    best_day: string;
    worst_day: string;
  };
  anomalies?: {
    total_count: number;
    high_anomalies: number;
    low_anomalies: number;
  };
}

const TimeSeriesPage = () => {
  const { data, isLoading, error, refetch } = useData<Record<string, TimeSeriesResult>>('time_series.json');
  const [selectedDataset, setSelectedDataset] = useState<string>('biometric');

  if (error) {
    return (
      <ErrorPage
        title="Failed to load time series data"
        message={error.message}
        onRetry={refetch}
      />
    );
  }

  // Get available datasets
  const datasets = data
    ? [...new Set(Object.keys(data).map((k) => k.split('_')[0]))]
    : ['biometric', 'demographic', 'enrollment'];

  // Get current dataset results
  const dailyTrends = data?.[`${selectedDataset}_daily_trends`];
  const seasonality = data?.[`${selectedDataset}_seasonality`] as TimeSeriesResult & { weekend_effect?: { weekday_mean: number; weekend_mean: number } };
  const anomalies = data?.[`${selectedDataset}_anomalies`];

  // Process chart data
  const trendChartData = dailyTrends?.daily_data?.slice(-60).map((d) => ({
    name: new Date(d.date).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }),
    enrollment: d.total_enrollment,
    ma7: d.ma_7 || 0,
    ma30: d.ma_30 || 0,
  })) || [];

  const dowChartData = seasonality?.day_of_week?.stats?.map((d) => ({
    name: d.day_name?.substring(0, 3),
    value: d.mean,
  })) || [];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
            Time Series Analysis
          </h1>
          <p className="text-slate-500 dark:text-slate-400">
            Analyze enrollment trends, seasonality, and anomalies over time
          </p>
        </div>

        {/* Dataset Selector */}
        <div className="flex items-center space-x-2">
          <span className="text-sm text-slate-500 dark:text-slate-400">Dataset:</span>
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
              title="Total Enrollment"
              value={formatNumber(dailyTrends?.enrollment_stats?.total || 0)}
              subtitle={`${dailyTrends?.date_range?.total_days || 0} days`}
              icon={<Calendar className="w-5 h-5" />}
              color="blue"
            />
            <StatCard
              title="Daily Average"
              value={formatNumber(dailyTrends?.enrollment_stats?.daily_mean || 0)}
              subtitle={`σ = ${formatNumber(dailyTrends?.enrollment_stats?.daily_std || 0)}`}
              icon={<BarChart2 className="w-5 h-5" />}
              color="green"
            />
            <StatCard
              title="Trend Direction"
              value={dailyTrends?.trend?.direction === 'increasing' ? 'Increasing' : 'Decreasing'}
              subtitle={`R² = ${(dailyTrends?.trend?.r_squared || 0).toFixed(3)}`}
              icon={
                dailyTrends?.trend?.direction === 'increasing' ? (
                  <TrendingUp className="w-5 h-5" />
                ) : (
                  <TrendingDown className="w-5 h-5" />
                )
              }
              color={dailyTrends?.trend?.direction === 'increasing' ? 'green' : 'red'}
            />
            <StatCard
              title="Anomalies Detected"
              value={anomalies?.anomalies?.total_count?.toString() || '0'}
              subtitle={`${anomalies?.anomalies?.high_anomalies || 0} high, ${anomalies?.anomalies?.low_anomalies || 0} low`}
              icon={<AlertTriangle className="w-5 h-5" />}
              color="orange"
            />
          </>
        )}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {isLoading ? (
          <>
            <LoadingChart />
            <LoadingChart />
          </>
        ) : (
          <>
            <ChartContainer
              title="Enrollment Trend"
              subtitle="Daily enrollment with 7-day and 30-day moving averages"
              className="lg:col-span-2"
            >
              {trendChartData.length > 0 ? (
                <SimpleLineChart
                  data={trendChartData}
                  dataKeys={['enrollment', 'ma7', 'ma30']}
                  height={300}
                />
              ) : (
                <div className="h-[300px] flex items-center justify-center text-slate-500">
                  No data available. Run the analysis first.
                </div>
              )}
            </ChartContainer>

            <ChartContainer
              title="Day of Week Pattern"
              subtitle="Average enrollment by day of week"
            >
              {dowChartData.length > 0 ? (
                <SimpleBarChart
                  data={dowChartData}
                  dataKeys={['value']}
                  height={250}
                />
              ) : (
                <div className="h-[250px] flex items-center justify-center text-slate-500">
                  No data available
                </div>
              )}
            </ChartContainer>

            <div className="card p-6">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                Seasonality Insights
              </h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-slate-600 dark:text-slate-400">Best Day</span>
                  <span className="font-medium text-green-600 dark:text-green-400">
                    {seasonality?.day_of_week?.best_day || 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-600 dark:text-slate-400">Worst Day</span>
                  <span className="font-medium text-red-600 dark:text-red-400">
                    {seasonality?.day_of_week?.worst_day || 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-600 dark:text-slate-400">Weekday Avg</span>
                  <span className="font-medium text-slate-900 dark:text-white">
                    {formatNumber(seasonality?.weekend_effect?.weekday_mean || 0)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-600 dark:text-slate-400">Weekend Avg</span>
                  <span className="font-medium text-slate-900 dark:text-white">
                    {formatNumber(seasonality?.weekend_effect?.weekend_mean || 0)}
                  </span>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Date Range Info */}
      {dailyTrends?.date_range && (
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
            Analysis Period
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-slate-500 dark:text-slate-400">Start Date</p>
              <p className="font-medium text-slate-900 dark:text-white">
                {dailyTrends.date_range.start}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-500 dark:text-slate-400">End Date</p>
              <p className="font-medium text-slate-900 dark:text-white">
                {dailyTrends.date_range.end}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-500 dark:text-slate-400">Total Days</p>
              <p className="font-medium text-slate-900 dark:text-white">
                {dailyTrends.date_range.total_days} days
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TimeSeriesPage;
