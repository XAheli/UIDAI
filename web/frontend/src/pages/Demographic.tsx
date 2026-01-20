/**
 * Demographic Page
 * Author: Shuvam Banerji Seal's Team
 */

import { useState } from 'react';
import { Users, BookOpen, UserCheck, TrendingUp } from 'lucide-react';
import {
  StatCard,
  LoadingCard,
  LoadingChart,
  ErrorPage,
  ChartContainer,
  SimpleBarChart,
  SimplePieChart,
} from '../components';
import { useData } from '../hooks';
import { formatNumber, formatPercentage } from '../utils/formatters';

const DemographicPage = () => {
  const { data, isLoading, error, refetch } = useData<Record<string, unknown>>('demographic.json');
  const [selectedDataset, setSelectedDataset] = useState<string>('biometric');

  if (error) {
    return (
      <ErrorPage
        title="Failed to load demographic data"
        message={error.message}
        onRetry={refetch}
      />
    );
  }

  const datasets = data
    ? [...new Set(Object.keys(data).map((k) => k.split('_')[0]))]
    : ['biometric', 'demographic', 'enrollment'];

  const ageData = data?.[`${selectedDataset}_age_groups`] as {
    percentages?: Record<string, number>;
    totals?: { by_age_group?: Record<string, number>; total?: number };
    top_youth_states?: Array<{ state: string; youth_pct: number }>;
  } | undefined;

  const populationData = data?.[`${selectedDataset}_population`] as {
    correlations?: Record<string, { correlation: number; significant: boolean }>;
    per_capita_stats?: { mean: number; top_states: Array<{ state: string; enrollment_per_capita: number }> };
  } | undefined;

  const literacyData = data?.[`${selectedDataset}_literacy`] as {
    regression?: { r_squared: number; significant: boolean };
    by_literacy_bin?: Array<{ literacy_bin: string; total_enrollment: number }>;
  } | undefined;

  const sexRatioData = data?.[`${selectedDataset}_sex_ratio`] as {
    by_category?: Array<{ category: string; total_enrollment: number }>;
  } | undefined;

  // Chart data
  const ageDistributionData = ageData?.percentages
    ? Object.entries(ageData.percentages).map(([name, value]) => ({
        name: name.replace('_', ' ').replace('children', 'Children').replace('adults', 'Adults').replace('infants', 'Infants'),
        value: value as number,
      }))
    : [];

  const literacyChartData = literacyData?.by_literacy_bin?.map((l) => ({
    name: l.literacy_bin,
    value: l.total_enrollment,
  })) || [];

  const sexRatioChartData = sexRatioData?.by_category?.map((s) => ({
    name: s.category?.substring(0, 15),
    value: s.total_enrollment,
  })) || [];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
            Demographic Analysis
          </h1>
          <p className="text-slate-500 dark:text-slate-400">
            Explore age groups, population correlations, and demographic trends
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
              title="Total Enrollment"
              value={formatNumber(ageData?.totals?.total || 0)}
              subtitle="All age groups"
              icon={<Users className="w-5 h-5" />}
              color="blue"
            />
            <StatCard
              title="Adult Enrollment"
              value={formatPercentage(ageData?.percentages?.adults || 0)}
              subtitle="Age 17+"
              icon={<UserCheck className="w-5 h-5" />}
              color="green"
            />
            <StatCard
              title="Pop. Correlation"
              value={(populationData?.correlations?.total_population?.correlation || 0).toFixed(3)}
              subtitle={populationData?.correlations?.total_population?.significant ? 'Significant' : 'Not significant'}
              icon={<TrendingUp className="w-5 h-5" />}
              color="orange"
            />
            <StatCard
              title="Literacy R²"
              value={(literacyData?.regression?.r_squared || 0).toFixed(3)}
              subtitle={literacyData?.regression?.significant ? 'Significant' : 'Not significant'}
              icon={<BookOpen className="w-5 h-5" />}
              color="purple"
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
            <LoadingChart />
            <LoadingChart />
          </>
        ) : (
          <>
            <ChartContainer
              title="Age Distribution"
              subtitle="Enrollment by age group"
            >
              {ageDistributionData.length > 0 ? (
                <SimplePieChart
                  data={ageDistributionData}
                  height={280}
                  innerRadius={50}
                  outerRadius={90}
                />
              ) : (
                <div className="h-[280px] flex items-center justify-center text-slate-500">
                  No data available
                </div>
              )}
            </ChartContainer>

            <ChartContainer
              title="Enrollment by Literacy Rate"
              subtitle="How literacy correlates with enrollment"
            >
              {literacyChartData.length > 0 ? (
                <SimpleBarChart
                  data={literacyChartData}
                  dataKeys={['value']}
                  height={280}
                />
              ) : (
                <div className="h-[280px] flex items-center justify-center text-slate-500">
                  No data available
                </div>
              )}
            </ChartContainer>

            <ChartContainer
              title="Sex Ratio Distribution"
              subtitle="Enrollment by sex ratio category"
            >
              {sexRatioChartData.length > 0 ? (
                <SimpleBarChart
                  data={sexRatioChartData}
                  dataKeys={['value']}
                  layout="vertical"
                  height={280}
                />
              ) : (
                <div className="h-[280px] flex items-center justify-center text-slate-500">
                  No data available
                </div>
              )}
            </ChartContainer>

            <div className="card p-6">
              <h3 className="font-semibold text-slate-900 dark:text-white mb-4">
                Per Capita Analysis
              </h3>
              <div className="space-y-4">
                <div className="p-4 rounded-lg bg-blue-50 dark:bg-blue-900/20">
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Average Enrollment per 1,000 Population
                  </p>
                  <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    {(populationData?.per_capita_stats?.mean || 0).toFixed(2)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-slate-500 dark:text-slate-400 mb-2">
                    Top States by Per Capita Enrollment
                  </p>
                  <div className="space-y-2">
                    {populationData?.per_capita_stats?.top_states?.slice(0, 5).map((s, i) => (
                      <div
                        key={s.state}
                        className="flex justify-between items-center text-sm"
                      >
                        <span className="text-slate-700 dark:text-slate-300">
                          {i + 1}. {s.state}
                        </span>
                        <span className="font-medium text-slate-900 dark:text-white">
                          {s.enrollment_per_capita?.toFixed(2)}
                        </span>
                      </div>
                    )) || (
                      <p className="text-slate-500">No data available</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Youth States Table */}
      {ageData?.top_youth_states && ageData.top_youth_states.length > 0 && (
        <div className="card overflow-hidden">
          <div className="p-4 border-b border-slate-200 dark:border-slate-700">
            <h3 className="font-semibold text-slate-900 dark:text-white">
              Top States by Youth Enrollment
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>State</th>
                  <th>Youth %</th>
                </tr>
              </thead>
              <tbody>
                {ageData.top_youth_states.map((state, i) => (
                  <tr key={state.state}>
                    <td className="font-medium">{i + 1}</td>
                    <td>{state.state}</td>
                    <td>{formatPercentage(state.youth_pct)}</td>
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

export default DemographicPage;
