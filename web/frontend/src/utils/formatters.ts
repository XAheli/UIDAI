/**
 * Formatting Utilities
 * Author: Shuvam Banerji Seal's Team
 */

/**
 * Format large numbers with commas and abbreviations
 */
export const formatNumber = (num: number, decimals: number = 0): string => {
  if (num === null || num === undefined || isNaN(num)) {
    return 'N/A';
  }
  
  if (Math.abs(num) >= 1e9) {
    return (num / 1e9).toFixed(decimals) + 'B';
  }
  if (Math.abs(num) >= 1e6) {
    return (num / 1e6).toFixed(decimals) + 'M';
  }
  if (Math.abs(num) >= 1e3) {
    return (num / 1e3).toFixed(decimals) + 'K';
  }
  
  return num.toLocaleString('en-IN', {
    maximumFractionDigits: decimals,
  });
};

/**
 * Format number with Indian numbering system
 */
export const formatIndianNumber = (num: number): string => {
  if (num === null || num === undefined || isNaN(num)) {
    return 'N/A';
  }
  
  return num.toLocaleString('en-IN');
};

/**
 * Format percentage
 */
export const formatPercentage = (num: number, decimals: number = 1): string => {
  if (num === null || num === undefined || isNaN(num)) {
    return 'N/A';
  }
  
  return `${num.toFixed(decimals)}%`;
};

/**
 * Format date string
 */
export const formatDate = (dateStr: string): string => {
  try {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-IN', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return dateStr;
  }
};

/**
 * Format timestamp
 */
export const formatTimestamp = (timestamp: string): string => {
  try {
    const date = new Date(timestamp);
    return date.toLocaleString('en-IN', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return timestamp;
  }
};

/**
 * Truncate text with ellipsis
 */
export const truncateText = (text: string, maxLength: number): string => {
  if (!text) return '';
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - 3) + '...';
};

/**
 * Get color based on value (for heatmaps)
 */
export const getHeatmapColor = (
  value: number,
  min: number,
  max: number,
  colorScheme: 'blue' | 'green' | 'orange' = 'blue'
): string => {
  const normalized = (value - min) / (max - min);
  
  const schemes = {
    blue: ['#eff6ff', '#bfdbfe', '#60a5fa', '#2563eb', '#1e40af'],
    green: ['#f0fdf4', '#bbf7d0', '#4ade80', '#16a34a', '#166534'],
    orange: ['#fff7ed', '#fed7aa', '#fb923c', '#ea580c', '#c2410c'],
  };
  
  const colors = schemes[colorScheme];
  const index = Math.min(Math.floor(normalized * colors.length), colors.length - 1);
  
  return colors[index];
};

/**
 * Get trend indicator
 */
export const getTrendIndicator = (value: number): { text: string; color: string; icon: string } => {
  if (value > 0) {
    return { text: `+${formatPercentage(value)}`, color: 'text-green-600 dark:text-green-400', icon: '↑' };
  } else if (value < 0) {
    return { text: formatPercentage(value), color: 'text-red-600 dark:text-red-400', icon: '↓' };
  }
  return { text: '0%', color: 'text-slate-500', icon: '→' };
};
