/**
 * Data Fetching Hook
 * Author: Shuvam Banerji Seal's Team
 */

import { useState, useEffect, useCallback } from 'react';
import { fetchData } from '../utils/dataFetcher';

interface UseDataOptions {
  enabled?: boolean;
  refetchInterval?: number;
}

interface UseDataResult<T> {
  data: T | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

export function useData<T>(
  filename: string,
  options: UseDataOptions = {}
): UseDataResult<T> {
  const { enabled = true, refetchInterval } = options;
  
  const [data, setData] = useState<T | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchDataFromSource = useCallback(async () => {
    if (!enabled) {
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await fetchData<T>(filename);
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Unknown error'));
    } finally {
      setIsLoading(false);
    }
  }, [filename, enabled]);

  useEffect(() => {
    fetchDataFromSource();
  }, [fetchDataFromSource]);

  // Set up refetch interval if specified
  useEffect(() => {
    if (!refetchInterval || !enabled) return;

    const interval = setInterval(fetchDataFromSource, refetchInterval);
    return () => clearInterval(interval);
  }, [refetchInterval, enabled, fetchDataFromSource]);

  return {
    data,
    isLoading,
    error,
    refetch: fetchDataFromSource,
  };
}

export default useData;
