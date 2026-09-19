
# Failure Sensor Predictor — Task 8

## Overview

An AI/ML predictive-maintenance system built using NASA C-MAPSS
FD001 turbofan engine degradation data.

The system uses multivariate engine telemetry to identify whether
an engine is approaching failure within a defined early-warning
horizon.

## Pipeline

Raw C-MAPSS telemetry
→ preprocessing
→ RUL calculation
→ 24-cycle early-warning labeling
→ rolling-window feature engineering
→ feature scaling
→ Random Forest classification
→ precision/recall evaluation
→ failure-risk inference

## Dataset

NASA C-MAPSS FD001.

The raw dataset is not redistributed in this repository.
Users should obtain it from the official NASA dataset source.

## Model

Random Forest Classifier

### Feature Engineering

- Rolling mean
- Rolling standard deviation
- First-order sensor differences
- Operational settings
- Sensor measurements

Rolling window: 12 cycles

Early-warning horizon: 24 cycles

## Final Test Results

| Metric | Score |
|---|---:|
| Precision | 0.6946 |
| Recall | 0.7085 |
| F1 Score | 0.7015 |

## Project Structure

```text
AI-ML-Failure-Sensor-Predictor/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── Failure_Sensor_Predictor.ipynb
│
├── models/
│   └── failure_sensor_predictor_package.joblib
│
├── outputs/
│   ├── figures/
│   │   └── early_warning_timeline.png
│   │
│   └── predictions/
│       └── FD001_test_predictions.csv
│
├── reports/
│   └── final_evaluation_report.json
│
└── src/
    └── inference.py
```

## Reproducibility

1. Obtain the NASA C-MAPSS dataset.
2. Place the FD001 files in the data directory.
3. Install the dependencies.
4. Run the training notebook.
5. Evaluate the model.
6. Use the saved model package for inference.

## Important Note

C-MAPSS records engine operating cycles rather than explicit
clock-hour timestamps. Therefore, the early-warning horizon is
reported in cycles rather than assuming that one cycle equals
one hour.
