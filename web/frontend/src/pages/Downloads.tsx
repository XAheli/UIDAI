/**
 * Downloads Page - PDF Reports and Analysis Results
 * Author: Shuvam Banerji Seal's Team
 */

import { useState, useEffect } from 'react';
import {
  Download,
  FileText,
  Globe,
  Users,
  Calculator,
  Clock,
  FileSpreadsheet,
  BarChart3,
  ExternalLink,
} from 'lucide-react';

interface PDFFile {
  name: string;
  displayName: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  category: string;
  fileSize?: string;
}

const pdfFiles: PDFFile[] = [
  {
    name: 'analysis_summary.pdf',
    displayName: 'Analysis Summary',
    description: 'Comprehensive overview of all analyses with executive summary and key findings',
    icon: FileText,
    category: 'Summary',
  },
  {
    name: 'time_series_analysis.pdf',
    displayName: 'Time Series Analysis',
    description: 'Daily trends, seasonality patterns, and anomaly detection for enrollment data',
    icon: Clock,
    category: 'Time Series',
  },
  {
    name: 'geographic_analysis.pdf',
    displayName: 'Geographic Analysis',
    description: 'State-wise and regional distribution analysis of Aadhaar enrollments',
    icon: Globe,
    category: 'Geographic',
  },
  {
    name: 'statistical_analysis.pdf',
    displayName: 'Statistical Analysis',
    description: 'Descriptive statistics, distributions, and correlation analysis',
    icon: Calculator,
    category: 'Statistical',
  },
  {
    name: 'demographic_analysis.pdf',
    displayName: 'Demographic Analysis',
    description: 'Age group distribution, population correlation, and literacy impact analysis',
    icon: Users,
    category: 'Demographic',
  },
];

const Downloads = () => {
  const [downloadCounts, setDownloadCounts] = useState<Record<string, number>>({});
  const [manifest, setManifest] = useState<{ pdfs: string[]; generated: string } | null>(null);

  useEffect(() => {
    // Load manifest to get available PDFs
    fetch('/pdfs/manifest.json')
      .then((res) => res.json())
      .then((data) => setManifest(data))
      .catch((err) => console.error('Error loading PDF manifest:', err));

    // Load download counts from localStorage
    const saved = localStorage.getItem('pdfDownloadCounts');
    if (saved) {
      setDownloadCounts(JSON.parse(saved));
    }
  }, []);

  const handleDownload = (fileName: string) => {
    // Update download count
    const newCounts = {
      ...downloadCounts,
      [fileName]: (downloadCounts[fileName] || 0) + 1,
    };
    setDownloadCounts(newCounts);
    localStorage.setItem('pdfDownloadCounts', JSON.stringify(newCounts));

    // Trigger download
    const link = document.createElement('a');
    link.href = `/pdfs/${fileName}`;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const isPdfAvailable = (fileName: string) => {
    return manifest?.pdfs.includes(fileName) ?? false;
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl p-8 text-white shadow-lg">
        <div className="flex items-center gap-4 mb-4">
          <Download className="h-12 w-12" />
          <div>
            <h1 className="text-3xl font-bold">Download Reports</h1>
            <p className="text-indigo-100 mt-1">
              High-quality PDF reports generated from the UIDAI analysis
            </p>
          </div>
        </div>
        {manifest && (
          <p className="text-sm text-indigo-200 mt-4">
            Last generated: {new Date(manifest.generated).toLocaleString()}
          </p>
        )}
      </div>

      {/* PDF List */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {pdfFiles.map((pdf, index) => {
          const Icon = pdf.icon;
          const available = isPdfAvailable(pdf.name);

          return (
            <div
              key={pdf.name}
              className={`bg-white dark:bg-slate-800 rounded-xl shadow-lg overflow-hidden border-2 transition-all duration-300 ${
                available
                  ? 'border-transparent hover:border-indigo-500 hover:shadow-xl transform hover:-translate-y-1'
                  : 'border-slate-200 dark:border-slate-700 opacity-60'
              }`}
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="p-6">
                <div className="flex items-start gap-4">
                  <div
                    className={`p-3 rounded-lg ${
                      available ? 'bg-indigo-100 dark:bg-indigo-900' : 'bg-slate-100 dark:bg-slate-700'
                    }`}
                  >
                    <Icon
                      className={`h-8 w-8 ${
                        available ? 'text-indigo-600 dark:text-indigo-400' : 'text-slate-400'
                      }`}
                    />
                  </div>
                  <div className="flex-1">
                    <span
                      className={`text-xs font-medium px-2 py-1 rounded-full ${
                        available
                          ? 'bg-indigo-100 dark:bg-indigo-900 text-indigo-700 dark:text-indigo-300'
                          : 'bg-slate-100 dark:bg-slate-700 text-slate-500'
                      }`}
                    >
                      {pdf.category}
                    </span>
                    <h3 className="text-lg font-semibold text-slate-900 dark:text-white mt-2">
                      {pdf.displayName}
                    </h3>
                    <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
                      {pdf.description}
                    </p>
                  </div>
                </div>

                <div className="mt-4 flex items-center justify-between">
                  <span className="text-xs text-slate-400 dark:text-slate-500">
                    {downloadCounts[pdf.name]
                      ? `Downloaded ${downloadCounts[pdf.name]} times`
                      : 'Not downloaded yet'}
                  </span>
                  <button
                    onClick={() => handleDownload(pdf.name)}
                    disabled={!available}
                    className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                      available
                        ? 'bg-indigo-600 text-white hover:bg-indigo-700'
                        : 'bg-slate-200 dark:bg-slate-700 text-slate-400 cursor-not-allowed'
                    }`}
                  >
                    <Download className="h-4 w-4" />
                    Download
                  </button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Data Export Section */}
      <div className="bg-white dark:bg-slate-800 rounded-xl shadow-lg p-6 border border-slate-200 dark:border-slate-700">
        <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-4 flex items-center gap-2">
          <FileSpreadsheet className="h-6 w-6 text-indigo-600" />
          Data Exports
        </h2>
        <p className="text-slate-600 dark:text-slate-400 mb-4">
          Download raw analysis data in JSON format for further processing or integration.
        </p>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {['time_series', 'geographic', 'demographic', 'statistical'].map(
            (type) => (
              <a
                key={type}
                href={`/data/${type}.json`}
                download
                className="flex items-center gap-2 p-4 rounded-lg border border-slate-200 dark:border-slate-700 hover:border-indigo-500 dark:hover:border-indigo-500 hover:bg-indigo-50 dark:hover:bg-indigo-900/20 transition-colors"
              >
                <BarChart3 className="h-5 w-5 text-indigo-600" />
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300 capitalize">
                  {type.replace('_', ' ')} JSON
                </span>
                <ExternalLink className="h-4 w-4 text-slate-400 ml-auto" />
              </a>
            )
          )}
        </div>
      </div>

      {/* GitHub Raw Content Notice */}
      <div className="bg-amber-50 dark:bg-amber-900/20 rounded-xl p-6 border border-amber-200 dark:border-amber-800">
        <h3 className="text-lg font-semibold text-amber-800 dark:text-amber-200 mb-2 flex items-center gap-2">
          <ExternalLink className="h-5 w-5" />
          GitHub Raw Content Access
        </h3>
        <p className="text-amber-700 dark:text-amber-300 text-sm">
          All analysis data and PDFs are also available via GitHub raw content
          URLs for direct integration. Use the following base URL pattern:
        </p>
        <code className="block mt-2 p-3 bg-amber-100 dark:bg-amber-900/40 rounded text-amber-900 dark:text-amber-100 text-sm font-mono overflow-x-auto">
          https://raw.githubusercontent.com/[username]/[repo]/main/web/frontend/public/data/[file].json
        </code>
      </div>
    </div>
  );
};

export default Downloads;
