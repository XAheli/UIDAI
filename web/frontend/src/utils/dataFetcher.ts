/**
 * Data Fetching Utilities
 * Author: Shuvam Banerji Seal's Team
 * 
 * Handles fetching data from GitHub raw content (production)
 * or local files (development)
 */

// Configuration
const IS_PRODUCTION = import.meta.env.PROD;
const GITHUB_RAW_BASE = 'https://raw.githubusercontent.com/YOUR_USERNAME/UIDAI_hackathon/main/web/frontend/public/data';
const LOCAL_BASE = '/data';

/**
 * Get the base URL for data fetching
 */
export const getDataBaseUrl = (): string => {
  // In development, use local files
  // In production (GitHub Pages), use raw GitHub content
  return IS_PRODUCTION ? GITHUB_RAW_BASE : LOCAL_BASE;
};

/**
 * Fetch JSON data from the appropriate source
 */
export const fetchData = async <T>(filename: string): Promise<T> => {
  const baseUrl = getDataBaseUrl();
  const url = `${baseUrl}/${filename}`;
  
  try {
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch ${filename}: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    return data as T;
  } catch (error) {
    console.error(`Error fetching ${filename}:`, error);
    throw error;
  }
};

/**
 * Fetch analysis summary
 */
export const fetchAnalysisSummary = async () => {
  return fetchData<AnalysisSummary>('analysis_summary.json');
};

/**
 * Fetch analysis index
 */
export const fetchAnalysisIndex = async () => {
  return fetchData<AnalysisIndex>('index.json');
};

/**
 * Fetch specific analysis data
 */
export const fetchAnalysis = async (analysisType: string) => {
  return fetchData<Record<string, unknown>>(`${analysisType}.json`);
};

// Types
export interface AnalysisSummary {
  generated_at: string;
  author: string;
  project: string;
  analyses_completed: Array<{
    type: string;
    result_count: number;
  }>;
  analyses_failed: Array<{
    type: string;
    error: string;
  }>;
  total_analyses: number;
  success_rate: number;
}

export interface AnalysisIndex {
  generated_at: string;
  available_analyses: Array<{
    name: string;
    file: string;
    has_error: boolean;
  }>;
}

// Export types for specific analyses
export interface TimeSeriesData {
  daily_trends?: Record<string, unknown>;
  seasonality?: Record<string, unknown>;
  growth?: Record<string, unknown>;
  anomalies?: Record<string, unknown>;
}

export interface GeographicData {
  state?: Record<string, unknown>;
  regional?: Record<string, unknown>;
  district?: Record<string, unknown>;
  pincode?: Record<string, unknown>;
  clustering?: Record<string, unknown>;
}

export interface DemographicData {
  age_groups?: Record<string, unknown>;
  population?: Record<string, unknown>;
  literacy?: Record<string, unknown>;
  sex_ratio?: Record<string, unknown>;
  age_trends?: Record<string, unknown>;
}

export interface StatisticalData {
  descriptive?: Record<string, unknown>;
  distribution?: Record<string, unknown>;
  correlation?: Record<string, unknown>;
  hypothesis?: Record<string, unknown>;
  outliers?: Record<string, unknown>;
  variance?: Record<string, unknown>;
}
