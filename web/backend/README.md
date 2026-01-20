# ML Inference API Backend

Flask-based REST API for running machine learning inference on Aadhaar enrollment data.

## Features

- Classification inference (predict region/state)
- Regression inference (predict population metrics)
- Anomaly detection
- Cluster assignment
- Feature importance retrieval

## Installation

```bash
cd /home/shuvam/codes/UIDAI_hackathon/web/backend
pip install flask flask-cors numpy pandas
```

## Running the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /api/health
```

### List Models
```
GET /api/models
```

### Classification Inference
```
POST /api/inference/classify
Content-Type: application/json

{
    "dataset": "biometric",
    "model": "random_forest",
    "features": {
        "population_2011": 500000,
        "literacy_rate": 75,
        "sex_ratio": 950
    }
}
```

### Regression Inference
```
POST /api/inference/predict
Content-Type: application/json

{
    "dataset": "biometric",
    "model": "random_forest",
    "features": {
        "literacy_rate": 80,
        "density_per_sq_km": 500
    }
}
```

### Anomaly Detection
```
POST /api/inference/detect_anomaly
Content-Type: application/json

{
    "dataset": "biometric",
    "features": {
        "population_2011": 50000000,
        "literacy_rate": 95
    }
}
```

### Cluster Assignment
```
POST /api/inference/cluster
Content-Type: application/json

{
    "dataset": "biometric",
    "features": {
        "literacy_rate": 75,
        "sex_ratio": 950
    }
}
```

### Feature Importance
```
GET /api/feature_importance/{dataset}/{task}
```

### Model Summary
```
GET /api/model_summary
```

## Authors

- Shuvam Banerji Seal
- Alok Mishra
- Aheli Poddar
