# UIDAI Aadhaar Data Analysis & Visualization Platform

**Authors: Aheli Poddar, Shuvam Banerji Seal, Alok Mishra**

A comprehensive, production-grade data analysis pipeline for Aadhaar enrollment, demographic, and biometric datasets with ML-powered insights and interactive web visualization.

---

## 🌟 Highlights

- **6.1M+ Records** analyzed across biometric, demographic, and enrollment datasets
- **150+ Statistical Analyses** with CSV/JSON exports
- **Interactive Web Dashboard** with dark/light mode
- **ML Models** for forecasting, anomaly detection, and classification
- **Multi-threaded Processing** utilizing all available CPU cores
- **Docker Support** for reproducible environments

---

## 📊 Dataset Summary

| Dataset | Records | Unique States | Unique Districts | File Size |
|---------|---------|---------------|------------------|-----------|
| **Biometric** | 3,531,034 | 36 | 948 | 603 MB |
| **Demographic** | 1,597,301 | 36 | 952 | 265 MB |
| **Enrollment** | 982,502 | 36 | 963 | 164 MB |
| **Total** | **6,110,837** | 36 | ~960 | **1.03 GB** |

---

## 🏗️ Project Structure

```
UIDAI_hackathon/
├── analysis/
│   └── codes/
│       ├── utils/              # Utility modules
│       │   ├── parallel.py     # Multiprocessing utilities
│       │   ├── io_utils.py     # File I/O operations
│       │   ├── validators.py   # Data validation
│       │   └── progress.py     # Progress tracking
│       ├── time_series/        # Time series analysis
│       ├── geographic/         # Geographic analysis
│       ├── demographic/        # Demographic analysis
│       ├── statistical/        # Statistical analysis
│       ├── ml_models/          # ML training & inference
│       ├── api_clients/        # External API clients
│       └── run_all_analyses.py # Master runner
├── web/
│   └── frontend/               # React/TypeScript dashboard
│       ├── src/
│       │   ├── components/     # Reusable UI components
│       │   ├── pages/          # Page components
│       │   ├── hooks/          # Custom React hooks
│       │   └── utils/          # Utility functions
│       └── public/data/        # Analysis results (JSON)
├── Dataset/                    # Data files
│   ├── api_data_aadhar_biometric/
│   ├── api_data_aadhar_demographic/
│   ├── api_data_aadhar_enrolment/
│   └── corrected_dataset/      # Augmented datasets
└── docker/                     # Docker configuration
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (3.13 recommended)
- **Node.js 18+** (for web frontend)
- **npm** (comes with Node.js)
- **Git LFS** (for large data files)

### Step 1: Clone & Setup

```bash
# Clone the repository
git clone git@github.com:XAheli/UIDAI.git
cd UIDAI

# If using Git LFS, pull large files
git lfs pull

# Create Python virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install pandas numpy scipy scikit-learn matplotlib tqdm
```

### Step 2: Run Analysis

```bash
# Navigate to analysis codes
cd analysis/codes

# Run all analyses with sample data (fast, ~5 seconds)
python run_all_analyses.py --sample 5000

# Run all analyses with full dataset (slower, ~2-5 minutes)
python run_all_analyses.py

# Run with sequential processing (for debugging)
python run_all_analyses.py --sample 5000 --sequential
```

**Analysis outputs are saved to:**
- `results/` - Detailed JSON/CSV files
- `web/frontend/public/data/` - JSON for web dashboard

### Step 3: View Web Dashboard

```bash
# Navigate to frontend directory
cd ../../web/frontend

# Install npm dependencies
npm install

# Start development server
npm run dev
```

Open http://localhost:5173 in your browser.

**Build for production:**
```bash
npm run build    # Creates dist/ folder
npm run preview  # Preview production build
```

---

## 📈 Analysis Modules

### Time Series Analysis
- Daily/weekly/monthly enrollment trends
- Moving averages (7-day, 30-day)
- Seasonality analysis (day-of-week patterns)
- Growth rate calculation
- Anomaly detection

### Geographic Analysis
- State-level distribution and market share
- Regional analysis (North, South, East, West, Central, Northeast)
- District-level deep-dive
- Gini coefficient for inequality measurement
- Top N states/districts ranking

### Demographic Analysis
- Population correlation analysis
- Literacy rate correlation
- Gender-based analysis
- Age group distribution (if available)
- Economic indicators correlation

### Statistical Analysis
- Descriptive statistics (mean, median, std, skewness, kurtosis)
- Distribution analysis with normality tests
- Correlation matrices
- Hypothesis testing (t-test, ANOVA, chi-square)
- Outlier detection (IQR, Z-score methods)
- Variance analysis

---

## 🖥️ CLI Reference

### Main Analysis Runner

```bash
# Show help
python run_all_analyses.py --help

# Run with sample data
python run_all_analyses.py --sample 10000

# Run specific analyses only
python run_all_analyses.py --analyses time_series,geographic

# Custom output directory
python run_all_analyses.py --output-dir ./my_results

# Sequential processing (useful for debugging)
python run_all_analyses.py --sequential
```

### Individual Analyzers

```python
# Time Series Analysis
from time_series.analyzer import run_time_series_analysis
results = run_time_series_analysis(sample_rows=5000)

# Geographic Analysis
from geographic.analyzer import run_geographic_analysis
results = run_geographic_analysis(sample_rows=5000)

# Demographic Analysis
from demographic.analyzer import run_demographic_analysis
results = run_demographic_analysis(sample_rows=5000)

# Statistical Analysis
from statistical.analyzer import run_statistical_analysis
results = run_statistical_analysis(sample_rows=5000)
```

---

## 🐳 Docker Support

```bash
# Build and run analysis
docker-compose up analysis

# Run frontend development server
docker-compose up web-dev

# Build frontend for production
docker-compose up web-build

# Run all services
docker-compose up
```

---

## 📁 Output Files

### Results Directory Structure
```
results/
├── analysis/
│   ├── time_series/     # Time series JSON files
│   ├── geographic/      # Geographic JSON files
│   ├── demographic/     # Demographic JSON files
│   └── statistical/     # Statistical JSON files
├── exports/
│   ├── json/           # Combined JSON exports
│   └── csv/            # CSV exports
└── analysis_summary.json
```

### Web Data Files
```
web/frontend/public/data/
├── analysis_summary.json
├── time_series.json
├── geographic.json
├── demographic.json
├── statistical.json
└── ml_results.json
```

---

## 🔧 Configuration

### Analysis Configuration

Edit `analysis/codes/config.py` to customize:
- Dataset paths
- Output directories
- Logging settings
- Processing parameters

### Frontend Configuration

Edit `web/frontend/vite.config.ts` for:
- Build settings
- Proxy configuration
- Environment variables

---

## ✨ Features

### Data Processing
- **Multi-threaded CSV Processing**: Utilizes all CPU cores
- **Intelligent Encoding Detection**: UTF-8, Latin-1, CP1252
- **State/District Normalization**: 100+ name variations standardized
- **Duplicate Removal**: 367K+ duplicates cleaned
- **Pincode Validation**: Invalid pincodes flagged

### Data Augmentation
- Census data (population, literacy, HDI)
- Geographic data (region, zone, climate)
- Economic indicators (per capita income)

### Web Dashboard
- React 18 with TypeScript
- Tailwind CSS with dark/light mode
- Recharts for interactive visualizations
- Responsive design
- Real-time theme switching

---

## 📊 Data Quality Metrics

| Metric | Biometric | Demographic | Enrollment |
|--------|-----------|-------------|------------|
| Duplicates Removed | 1,700 | 364,958 | 1,119 |
| Census Coverage | 100% | 100% | 100% |
| State Normalization | 36/36 | 36/36 | 36/36 |

---

## 🛠️ Technology Stack

### Backend (Python)
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **scipy** - Statistical functions
- **scikit-learn** - ML algorithms
- **matplotlib** - Plotting
- **tqdm** - Progress bars

### Frontend (TypeScript)
- **React 18** - UI framework
- **Vite 5** - Build tool
- **TypeScript 5** - Type safety
- **Tailwind CSS 3** - Styling
- **Recharts** - Charts & visualization
- **Lucide React** - Icons

---

## 🐛 Troubleshooting

### Common Issues

**1. "Module not found" errors**
```bash
# Ensure you're in the correct directory
cd analysis/codes

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/UIDAI_hackathon/analysis/codes"
```

**2. Memory errors with large datasets**
```bash
# Use sample data
python run_all_analyses.py --sample 10000
```

**3. Frontend build errors**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**4. Dataset not found**
```bash
# Check if augmented datasets exist
ls -la Dataset/corrected_dataset/biometric/
ls -la Dataset/corrected_dataset/demographic/
ls -la Dataset/corrected_dataset/enrollement/
```
