# ML Inference Dockerfile
# Author: Shuvam Banerji Seal's Team

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    git-lfs \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Install Python dependencies for ML
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

# Copy project files
COPY . .

# Create output directories
RUN mkdir -p results/analysis/ml_predictions results/exports models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV OMP_NUM_THREADS=12

# Default command
CMD ["python", "-m", "analysis.codes.ml_models.run_all"]
