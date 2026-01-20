/**
 * Error Display Component
 * Author: Shuvam Banerji Seal's Team
 */

import { AlertCircle, RefreshCw } from 'lucide-react';
import { clsx } from 'clsx';

interface ErrorDisplayProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  className?: string;
}

export const ErrorDisplay = ({
  title = 'Error',
  message,
  onRetry,
  className,
}: ErrorDisplayProps) => {
  return (
    <div
      className={clsx(
        'card p-6 border-red-200 dark:border-red-800',
        className
      )}
    >
      <div className="flex items-start space-x-4">
        <div className="flex-shrink-0">
          <div className="p-2 rounded-full bg-red-100 dark:bg-red-900/30">
            <AlertCircle className="w-6 h-6 text-red-600 dark:text-red-400" />
          </div>
        </div>
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-red-800 dark:text-red-200">
            {title}
          </h3>
          <p className="mt-1 text-red-600 dark:text-red-400">{message}</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-4 flex items-center space-x-2 px-4 py-2 rounded-lg bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 hover:bg-red-200 dark:hover:bg-red-900/50 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Try Again</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export const ErrorPage = ({
  title = 'Something went wrong',
  message,
  onRetry,
}: ErrorDisplayProps) => {
  return (
    <div className="min-h-[400px] flex flex-col items-center justify-center text-center px-4">
      <div className="p-4 rounded-full bg-red-100 dark:bg-red-900/30 mb-4">
        <AlertCircle className="w-12 h-12 text-red-600 dark:text-red-400" />
      </div>
      <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
        {title}
      </h2>
      <p className="text-slate-600 dark:text-slate-400 max-w-md mb-6">
        {message}
      </p>
      {onRetry && (
        <button onClick={onRetry} className="btn btn-primary flex items-center space-x-2">
          <RefreshCw className="w-4 h-4" />
          <span>Try Again</span>
        </button>
      )}
    </div>
  );
};

export default ErrorDisplay;
