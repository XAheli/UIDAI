/**
 * Loading Components
 * Author: Shuvam Banerji Seal's Team
 */

import { clsx } from 'clsx';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export const LoadingSpinner = ({ size = 'md', className }: LoadingSpinnerProps) => {
  const sizeClasses = {
    sm: 'w-4 h-4 border-2',
    md: 'w-8 h-8 border-4',
    lg: 'w-12 h-12 border-4',
  };

  return (
    <div
      className={clsx(
        'border-blue-600 border-t-transparent rounded-full animate-spin',
        sizeClasses[size],
        className
      )}
    />
  );
};

interface LoadingCardProps {
  className?: string;
}

export const LoadingCard = ({ className }: LoadingCardProps) => {
  return (
    <div className={clsx('card p-6 animate-pulse', className)}>
      <div className="skeleton h-4 w-24 mb-4" />
      <div className="skeleton h-8 w-32 mb-2" />
      <div className="skeleton h-3 w-20" />
    </div>
  );
};

export const LoadingChart = ({ className }: LoadingCardProps) => {
  return (
    <div className={clsx('card p-6 animate-pulse', className)}>
      <div className="skeleton h-5 w-40 mb-6" />
      <div className="flex items-end space-x-2 h-48">
        {[...Array(12)].map((_, i) => (
          <div
            key={i}
            className="skeleton flex-1"
            style={{ height: `${Math.random() * 70 + 30}%` }}
          />
        ))}
      </div>
    </div>
  );
};

export const LoadingTable = ({ className, rows = 5 }: LoadingCardProps & { rows?: number }) => {
  return (
    <div className={clsx('card overflow-hidden animate-pulse', className)}>
      {/* Header */}
      <div className="flex px-4 py-3 bg-slate-50 dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700">
        <div className="skeleton h-4 w-24" />
        <div className="skeleton h-4 w-32 ml-auto" />
        <div className="skeleton h-4 w-20 ml-8" />
      </div>
      {/* Rows */}
      {[...Array(rows)].map((_, i) => (
        <div
          key={i}
          className="flex px-4 py-3 border-b border-slate-100 dark:border-slate-800 last:border-0"
        >
          <div className="skeleton h-4 w-28" />
          <div className="skeleton h-4 w-24 ml-auto" />
          <div className="skeleton h-4 w-16 ml-8" />
        </div>
      ))}
    </div>
  );
};

interface FullPageLoadingProps {
  message?: string;
}

export const FullPageLoading = ({ message = 'Loading...' }: FullPageLoadingProps) => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-slate-50 dark:bg-slate-900">
      <LoadingSpinner size="lg" />
      <p className="mt-4 text-slate-600 dark:text-slate-400">{message}</p>
    </div>
  );
};

export default LoadingSpinner;
