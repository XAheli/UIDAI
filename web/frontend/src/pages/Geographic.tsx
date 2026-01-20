/**
 * Geographic Page
 * Author: Shuvam Banerji Seal's Team
 */

import { useState } from 'react';
import { MapPin, Building2, Mail, TrendingUp } from 'lucide-react';
import {
  StatCard,
  LoadingCard,
  LoadingChart,
  LoadingTable,
  ErrorPage,
  ChartContainer,
  SimpleBarChart,
  SimplePieChart,
} from '../components';
import { useData } from '../hooks';
import { formatNumber, formatPercentage } from '../utils/formatters';

interface StateData {
  state: string;
  total_enrollment: number;
  market_share_pct: number;
  rank: number;
  unique_districts: number;
  unique_pincodes: number;
  region?: string;
}

interface GeographicResult {
  dataset_name: string;
  summary?: {
    total_states: number;
    total_enrollment: number;
    gini_coefficient: number;
    top_5_concentration: number;
    top_10_concentration: number;
    total_districts?: number;
    avg_districts_per_state?: number;
  };
  top_10_states?: StateData[];
  state_details?: StateData[];
  regional_stats?: Array<{
    region: string;
    total_enrollment: number;
    share_pct: number;
    unique_states: number;
  }>;
  top_districts?: Array<{
    state: string;
    district: string;
    total_enrollment: number;
  }>;
}

const GeographicPage = () => {
  const { data, isLoading, error, refetch } = useData<Record<string, GeographicResult>>('geographic.json');
  const [selectedDataset, setSelectedDataset] = useState<string>('biometric');
  const [view, setView] = useState<'state' | 'region' | 'district'>('state');

  if (error) {
    return (
      <ErrorPage
        title="Failed to load geographic data"
        message={error.message}
        onRetry={refetch}
      />
    );
  }

  const datasets = data
    ? [...new Set(Object.keys(data).map((k) => k.split('_')[0]))]
    : ['biometric', 'demographic', 'enrollment'];

  const stateData = data?.[`${selectedDataset}_state`];
  const regionalData = data?.[`${selectedDataset}_regional`];
  const districtData = data?.[`${selectedDataset}_district`];

  // Chart data
  const stateChartData = stateData?.top_10_states?.map((s) => ({
    name: s.state?.substring(0, 10) || 'Unknown',
    value: s.total_enrollment,
  })) || [];

  const regionChartData = regionalData?.regional_stats?.map((r) => ({
    name: r.region,
    value: r.total_enrollment,
  })) || [];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
            Geographic Analysis
          </h1>
          <p className="text-slate-500 dark:text-slate-400">
            Explore state-wise, regional, and district-level enrollment patterns
          </p>
        </div>

        <div className="flex items-center space-x-4">
          {/* View Selector */}
          <div className="flex rounded-lg border border-slate-200 dark:border-slate-700 overflow-hidden">
            {(['state', 'region', 'district'] as const).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`px-4 py-2 text-sm font-medium capitalize ${
                  view === v
                    ? 'bg-blue-600 text-white'
                    : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700'
                }`}
              >
                {v}
              </button>
            ))}
          </div>

          {/* Dataset Selector */}
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
              title="Total States/UTs"
              value={stateData?.summary?.total_states?.toString() || '36'}
              subtitle="Complete coverage"
              icon={<MapPin className="w-5 h-5" />}
              color="blue"
            />
            <StatCard
              title="Total Districts"
              value={districtData?.summary?.total_districts?.toString() || '700+'}
              subtitle={`Avg ${districtData?.summary?.avg_districts_per_state?.toFixed(0) || '20'}/state`}
              icon={<Building2 className="w-5 h-5" />}
              color="green"
            />
            <StatCard
              title="Top 5 Concentration"
              value={formatPercentage(stateData?.summary?.top_5_concentration || 0)}
              subtitle="Market share"
              icon={<TrendingUp className="w-5 h-5" />}
              color="orange"
            />
            <StatCard
              title="Gini Coefficient"
              value={(stateData?.summary?.gini_coefficient || 0).toFixed(3)}
              subtitle="Distribution inequality"
              icon={<Mail className="w-5 h-5" />}
              color="purple"
            />
          </>
        )}
      </div>

      {/* Main Content Based on View */}
      {view === 'state' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {isLoading ? (
            <>
              <LoadingChart />
              <LoadingTable />
            </>
          ) : (
            <>
              <ChartContainer
                title="Top 10 States by Enrollment"
                subtitle="State-wise distribution of Aadhaar enrollment"
              >
                {stateChartData.length > 0 ? (
                  <SimpleBarChart
                    data={stateChartData}
                    dataKeys={['value']}
                    layout="vertical"
                    height={350}
                  />
                ) : (
                  <div className="h-[350px] flex items-center justify-center text-slate-500">
                    No data available
                  </div>
                )}
              </ChartContainer>

              <div className="card overflow-hidden">
                <div className="p-4 border-b border-slate-200 dark:border-slate-700">
                  <h3 className="font-semibold text-slate-900 dark:text-white">
                    State Rankings
                  </h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Rank</th>
                        <th>State</th>
                        <th>Enrollment</th>
                        <th>Share</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stateData?.top_10_states?.map((state, i) => (
                        <tr key={state.state}>
                          <td className="font-medium">{i + 1}</td>
                          <td>{state.state}</td>
                          <td>{formatNumber(state.total_enrollment)}</td>
                          <td>{formatPercentage(state.market_share_pct)}</td>
                        </tr>
                      )) || (
                        <tr>
                          <td colSpan={4} className="text-center text-slate-500">
                            No data available
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {view === 'region' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {isLoading ? (
            <>
              <LoadingChart />
              <LoadingChart />
            </>
          ) : (
            <>
              <ChartContainer
                title="Regional Distribution"
                subtitle="Enrollment by geographic region"
              >
                {regionChartData.length > 0 ? (
                  <SimplePieChart
                    data={regionChartData}
                    height={300}
                    innerRadius={60}
                    outerRadius={100}
                  />
                ) : (
                  <div className="h-[300px] flex items-center justify-center text-slate-500">
                    No data available
                  </div>
                )}
              </ChartContainer>

              <div className="card p-6">
                <h3 className="font-semibold text-slate-900 dark:text-white mb-4">
                  Regional Breakdown
                </h3>
                <div className="space-y-4">
                  {regionalData?.regional_stats?.map((region) => (
                    <div key={region.region} className="flex items-center justify-between">
                      <div>
                        <p className="font-medium text-slate-900 dark:text-white">
                          {region.region}
                        </p>
                        <p className="text-sm text-slate-500 dark:text-slate-400">
                          {region.unique_states} states
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="font-medium text-slate-900 dark:text-white">
                          {formatNumber(region.total_enrollment)}
                        </p>
                        <p className="text-sm text-slate-500 dark:text-slate-400">
                          {formatPercentage(region.share_pct)}
                        </p>
                      </div>
                    </div>
                  )) || (
                    <p className="text-slate-500">No data available</p>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {view === 'district' && (
        <div className="card overflow-hidden">
          {isLoading ? (
            <LoadingTable rows={10} />
          ) : (
            <>
              <div className="p-4 border-b border-slate-200 dark:border-slate-700">
                <h3 className="font-semibold text-slate-900 dark:text-white">
                  Top Districts by Enrollment
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>State</th>
                      <th>District</th>
                      <th>Total Enrollment</th>
                    </tr>
                  </thead>
                  <tbody>
                    {districtData?.top_districts?.slice(0, 20).map((district, i) => (
                      <tr key={`${district.state}-${district.district}`}>
                        <td className="font-medium">{i + 1}</td>
                        <td>{district.state}</td>
                        <td>{district.district}</td>
                        <td>{formatNumber(district.total_enrollment)}</td>
                      </tr>
                    )) || (
                      <tr>
                        <td colSpan={4} className="text-center text-slate-500">
                          No data available
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default GeographicPage;
