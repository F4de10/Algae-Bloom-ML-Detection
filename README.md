# Algae Bloom Prediction

Predictive machine learning system for detecting and forecasting harmful algal blooms (HABs) using geospatial data, satellite imagery, and ensemble learning techniques.

## Overview

This project implements an advanced machine learning pipeline for early detection and prediction of algal blooms in aquatic environments. By combining multi-temporal satellite imagery with environmental features and ensemble learning methods, the system achieves state-of-the-art performance in identifying conditions that lead to harmful algal bloom events.

The project extends the second-place solution from the [DrivenData TickTickBloom competition](https://github.com/drivendataorg/tick-tick-bloom), implementing improvements in feature engineering, model ensembling, and geospatial analysis. This work was developed for the GeoAI course (AG2418) at KTH Royal Institute of Technology.

## Key Features

- **Multi-Source Data Integration**: Combines satellite imagery from Planetary Computer, environmental sensor data, and geospatial indices
- **Advanced Feature Engineering**: Implements domain-specific geospatial features and temporal analysis
- **Ensemble Learning**: Leverages multiple gradient boosting algorithms (CatBoost, XGBoost, LightGBM) for robust predictions
- **Hyperparameter Optimization**: Automated hyperparameter tuning using Optuna for model optimization
- **Geospatial Analysis**: Full support for geographic data using GeoPandas and spatial indexing
- **Production-Ready Pipeline**: Modular code structure with data preparation, feature engineering, and model inference stages

## Technology Stack

**Machine Learning & Data Science:**
- CatBoost, XGBoost, LightGBM
- scikit-learn
- Optuna (hyperparameter optimization)
- Pandas, NumPy, SciPy

**Geospatial & Satellite Imagery:**
- GeoPandas
- Planetary Computer
- PySTAC Client
- rioxarray, ODC-STAC
- OpenCV, Pillow

**Visualization & Analysis:**
- Matplotlib, Seaborn
- Jupyter Notebook

## Prerequisites

- Python 3.9 or later
- Anaconda or Miniconda
- 8GB+ RAM (16GB+ recommended for full pipeline)
- Internet connection for downloading satellite data

## Quick Start

### 1. Set up Python Environment

```bash
# Create conda environment with GeoPandas (installed first due to complex dependencies)
conda create --name bloom python=3.9 pip geopandas
conda activate bloom

# Install remaining dependencies
pip install -r requirements.txt

# Install Jupyter kernel for notebooks
ipython kernel install --name "bloom_jpy" --user
```

### 2. Prepare Data

```bash
# Run from repository root directory
# This script downloads satellite data and prepares the SQLite database
python main_prepdata.py
```

### 3. Train and Evaluate Models

```bash
# Train ensemble models and generate predictions
python extension.py
```

This generates:
- Trained models in `./models` directory (organized by date)
- Analysis figures in `./figures` directory
- Model performance metrics and visualizations

## Usage Guide

### Data Preparation

The `main_prepdata.py` script:
1. Downloads multi-temporal Sentinel-2 satellite imagery via Planetary Computer
2. Extracts spectral indices (NDVI, NDWI, etc.)
3. Aggregates environmental features at target locations
4. Constructs a SQLite database for efficient data access

### Model Training

The `extension.py` script:
1. Loads preprocessed data from SQLite database
2. Applies feature engineering transformations
3. Trains ensemble models using optimized hyperparameters
4. Generates predictions on test datasets
5. Produces evaluation metrics and visualization plots

### Hyperparameter Tuning

For custom hyperparameter optimization:

```bash
# Run hyperparameter tuning experiments
python main_hypertune.py > hypertune_results.txt
```

Results provide guidance for model selection and parameter ranges.

### Modeling Strategy

Detailed information about the modeling approach, feature engineering decisions, and experimental results is available in:

```
model_strategy.ipynb
```

This Jupyter notebook documents:
- Exploratory data analysis
- Feature importance analysis
- Model comparison and validation
- Final ensemble configuration

## Project Architecture

```
.
├── main_prepdata.py           # Data download and preparation pipeline
├── extension.py               # Main model training and evaluation script
├── main_hypertune.py          # Hyperparameter optimization experiments
├── model_strategy.ipynb       # Detailed analysis and methodology documentation
├── src/
│   ├── feat.py               # Feature engineering utilities
│   └── mod.py                # Model definitions and training utilities
├── data/                      # Raw and processed datasets
├── models/                    # Trained model artifacts (organized by timestamp)
├── figures/                   # Generated analysis plots and visualizations
└── requirements.txt           # Python package dependencies
```

## Configuration

Key configuration options in the main scripts:

- **Python Version**: Python 3.9 (specified in conda environment creation)
- **Data Directory**: `./data` (contains raw and processed datasets)
- **Model Output**: `./models/` (timestamped subdirectories)
- **Figure Output**: `./figures/` (analysis and validation plots)

Environment variables are not currently required. All configuration is embedded in script parameters.

## Security Features

- **Data Validation**: Input data is validated before processing to prevent malformed datasets
- **Safe File Operations**: All file paths use secure methods to prevent directory traversal
- **Dependency Verification**: All dependencies are pinned in requirements.txt
- **No Sensitive Data**: No API keys, credentials, or sensitive information in repository

See [LICENSE](LICENSE) for terms of use.

## Performance Optimizations

- **Efficient Data Loading**: SQLite database optimized for sequential and random access patterns
- **Vectorized Operations**: NumPy and Pandas vectorization throughout feature engineering
- **Memory-Efficient Satellite Processing**: Chunked processing of large raster data using rioxarray
- **Parallel Training**: XGBoost and LightGBM leverage multi-core processors
- **Feature Caching**: Preprocessed features cached to avoid recomputation

## Model Performance

The ensemble approach combines three gradient boosting algorithms, achieving:
- Robust predictions across diverse environmental conditions
- High sensitivity to early bloom indicators
- Generalization to geographic regions outside training data

Performance metrics and detailed validation results are available in the generated analysis figures and the `model_strategy.ipynb` notebook.

## Reproducibility

To ensure reproducibility:
1. All random seeds are fixed in training scripts
2. Python version and exact dependencies are specified
3. Data downloads use deterministic sources (Planetary Computer)
4. Generated models are timestamped for versioning

## Contributors

- **Nils Olivier** - Extension and implementation (nolivier@kth.se)
  - KTH Royal Institute of Technology

- **Andy Wheeler** - Original competition solution and methodology (apwheele@gmail.com)
  - [2nd Place Solution - TickTickBloom Competition](https://github.com/drivendataorg/tick-tick-bloom/tree/main/2nd%20Place)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Nils Olivier and Andy Wheeler

## Acknowledgments

- DrivenData for the TickTickBloom competition and challenge dataset
- KTH Royal Institute of Technology for supporting this research
- Planetary Computer for providing satellite imagery and computational resources
- The open-source community for excellent geospatial and machine learning libraries
