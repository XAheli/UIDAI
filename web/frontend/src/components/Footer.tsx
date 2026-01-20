/**
 * Footer Component
 * Author: Shuvam Banerji Seal's Team
 */

import { Heart, Github, Linkedin, Mail } from 'lucide-react';

export const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-white dark:bg-slate-900 border-t border-slate-200 dark:border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* About */}
          <div>
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-10 h-10 rounded-lg gradient-bg flex items-center justify-center">
                <span className="text-white font-bold text-lg">UA</span>
              </div>
              <div>
                <h3 className="font-bold text-slate-900 dark:text-white">
                  UIDAI Analysis
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Hackathon Project
                </p>
              </div>
            </div>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Comprehensive analysis of Aadhaar enrollment data across India,
              featuring time series analysis, geographic patterns, demographic
              insights, and machine learning predictions.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-semibold text-slate-900 dark:text-white mb-4">
              Analysis Categories
            </h4>
            <ul className="space-y-2 text-sm">
              <li>
                <a
                  href="/time-series"
                  className="text-slate-600 dark:text-slate-400 hover:text-blue-600 dark:hover:text-blue-400"
                >
                  Time Series Analysis
                </a>
              </li>
              <li>
                <a
                  href="/geographic"
                  className="text-slate-600 dark:text-slate-400 hover:text-blue-600 dark:hover:text-blue-400"
                >
                  Geographic Analysis
                </a>
              </li>
              <li>
                <a
                  href="/demographic"
                  className="text-slate-600 dark:text-slate-400 hover:text-blue-600 dark:hover:text-blue-400"
                >
                  Demographic Analysis
                </a>
              </li>
              <li>
                <a
                  href="/statistical"
                  className="text-slate-600 dark:text-slate-400 hover:text-blue-600 dark:hover:text-blue-400"
                >
                  Statistical Analysis
                </a>
              </li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h4 className="font-semibold text-slate-900 dark:text-white mb-4">
              Connect
            </h4>
            <div className="flex space-x-4">
              <a
                href="https://github.com/YOUR_USERNAME"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
              >
                <Github className="w-5 h-5" />
              </a>
              <a
                href="https://linkedin.com/in/YOUR_PROFILE"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
              >
                <Linkedin className="w-5 h-5" />
              </a>
              <a
                href="mailto:contact@example.com"
                className="p-2 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
              >
                <Mail className="w-5 h-5" />
              </a>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-8 pt-8 border-t border-slate-200 dark:border-slate-800">
          <div className="flex flex-col md:flex-row items-center justify-between space-y-4 md:space-y-0">
            <p className="text-sm text-slate-500 dark:text-slate-400">
              © {currentYear} Shuvam Banerji Seal's Team. All rights reserved.
            </p>
            <p className="text-sm text-slate-500 dark:text-slate-400 flex items-center">
              Made with <Heart className="w-4 h-4 mx-1 text-red-500" /> for UIDAI
              Hackathon
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
