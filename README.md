# IDS Research

A comprehensive Python library for Intrusion Detection System (IDS) research. This library provides tools for preprocessing, modeling, and evaluating network flow data for intrusion detection applications.

## Overview

This project contains tools and utilities for academic research in the field of network intrusion detection systems. It includes preprocessors for standardizing network flow data, components for building machine learning pipelines, and experiment tracking.

## Project Structure

```
ids-research/
├── notebooks/                  # Jupyter notebooks for experiments
│   └── cicflowmeter_flows_preprocessing.ipynb
├── src/
│   └── ids_research/           # Main package
│       ├── components/         # Reusable components
│       ├── experiments/        # Experiment tracking
│       ├── models/             # ML models
│       ├── pipelines/          # Processing pipelines
│       └── preprocessors/      # Data preprocessors
│           ├── base.py         # Base preprocessor class
│           └── cicflowmeter.py # CICFlowMeter preprocessor
├── tests/                      # Unit tests
│   ├── test_basepreprocessor.py
│   └── test_cicflowmeter.py
├── pyproject.toml              # Project configuration
├── README.md                   # This file
├── requirements.txt            # Generated dependencies
└── uv.lock                     # Lock file for uv package manager
```

## Installation

```bash
# Clone the repository
git clone https://github.com/thelurps/ids-research.git
cd ids-research

# Install the package with CPU support (default)
uv pip install -e .

# Or install with GPU support
uv pip install -e ".[gpu]"

# Install development dependencies
uv pip install -e ".[dev]"
```

## Usage

### CicFlowMeterPreprocessor

The `CicFlowMeterPreprocessor` is designed to preprocess network flow data generated by CICFlowMeter. It handles feature scaling, one-hot encoding of categorical features, and binary classification mapping.

#### Basic Example

```python
import polars as pl
from ids_research.preprocessors import CicFlowMeterPreprocessor

# Load your CICFlowMeter data
data_path = "path/to/cicflowmeter_data.parquet"
flow_data = pl.read_parquet(data_path)

# Create a preprocessor with default configuration
preprocessor = CicFlowMeterPreprocessor()

# Fit the preprocessor to your data
preprocessor.fit(flow_data)

# Transform the data
transformed_data = preprocessor.transform(flow_data)

# Or fit and transform in one step
transformed_data = preprocessor.fit_transform(flow_data)
```

#### Custom Configuration

You can customize the preprocessor with your own configuration:

```python
# Create a preprocessor with custom configuration
custom_config = {
    "scaler": "StandardScaler",  # Use StandardScaler instead of MinMaxScaler
    "numerical_features": [      # Specify custom numerical features
        "Flow Duration",
        "Total Fwd Packet",
        "Flow Bytes/s",
        # Add other numerical features...
    ],
    "categorical_features": ["Protocol"],  # Features to one-hot encode
    "target_col": "Label",       # Target column for classification
    "binary_classifier": True,   # Convert labels to binary (True/False)
    "target_mappings": {         # How to map labels to binary values
        "BENIGN": False,
        "benign": False,
        "notice": False,
        "*": True,               # Catch-all for attack labels
    },
}

preprocessor = CicFlowMeterPreprocessor(config=custom_config)
```

## Features

- **Preprocessing**: Standardize network flow data from different sources
- **Polars Integration**: Fast data processing with Polars DataFrames
- **Scikit-learn Compatibility**: Seamless integration with scikit-learn
- **Extensibility**: Easy to extend with new preprocessors and models

## Requirements

- Python >= 3.12
- polars >= 1.25.2
- scikit-learn >= 1.6.1
- numpy >= 2.2.5
- pyarrow >= 19.0.1
- duckdb >= 1.2.2

## License

[MIT License](LICENSE)
