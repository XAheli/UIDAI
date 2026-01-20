/**
 * Stat Card Component
 * Author: Shuvam Banerji Seal's Team
 */

import { ReactNode } from 'react';
import { clsx } from 'clsx';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: ReactNode;
  trend?: {
    value: number;
    label: string;
  };
  color?: 'blue' | 'green' | 'orange' | 'purple' | 'red';
  className?: string;
}

const colorClasses = {
  blue: 'from-blue-500 to-blue-600',
  green: 'from-green-500 to-green-600',
  orange: 'from-orange-500 to-orange-600',
  purple: 'from-purple-500 to-purple-600',
  red: 'from-red-500 to-red-600',
};

const iconBgClasses = {
  blue: 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400',
  green: 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400',
  orange: 'bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400',
  purple: 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400',
  red: 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400',
};

export const StatCard = ({
  title,
  value,
  subtitle,
  icon,
  trend,
  color = 'blue',
  className,
}: StatCardProps) => {
  return (
    <div className={clsx('card p-6', className)}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-slate-500 dark:text-slate-400">
            {title}
          </p>
          <p className="mt-2 text-3xl font-bold text-slate-900 dark:text-white">
            {value}
          </p>
          {subtitle && (
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
              {subtitle}
            </p>
          )}
          {trend && (
            <div className="mt-2 flex items-center">
              <span
                className={clsx(
                  'text-sm font-medium',
                  trend.value >= 0
                    ? 'text-green-600 dark:text-green-400'
                    : 'text-red-600 dark:text-red-400'
                )}
              >
                {trend.value >= 0 ? '+' : ''}
                {trend.value.toFixed(1)}%
              </span>
              <span className="ml-2 text-sm text-slate-500 dark:text-slate-400">
                {trend.label}
              </span>
            </div>
          )}
        </div>
        {icon && (
          <div className={clsx('p-3 rounded-lg', iconBgClasses[color])}>
            {icon}
          </div>
        )}
      </div>
    </div>
  );
};

export const GradientStatCard = ({
  title,
  value,
  subtitle,
  icon,
  color = 'blue',
  className,
}: StatCardProps) => {
  return (
    <div
      className={clsx(
        'rounded-xl p-6 text-white bg-gradient-to-br',
        colorClasses[color],
        className
      )}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-white/80">{title}</p>
          <p className="mt-2 text-3xl font-bold">{value}</p>
          {subtitle && (
            <p className="mt-1 text-sm text-white/70">{subtitle}</p>
          )}
        </div>
        {icon && <div className="p-3 rounded-lg bg-white/20">{icon}</div>}
      </div>
    </div>
  );
};

export default StatCard;
