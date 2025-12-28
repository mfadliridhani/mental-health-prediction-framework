# Setup and Installation Guide

## Prerequisites

Before setting up the project, ensure you have:

- **Python 3.8 or higher**: Check with `python --version`
- **pip**: Python package installer (usually comes with Python)
- **Git**: For version control (optional but recommended)

## Installation Steps

### 1. Clone or Download the Repository

**Option A: Using Git**

```bash
git clone https://github.com/YOUR_USERNAME/mental-health-prediction.git
cd mental-health-prediction
```

**Option B: Download ZIP**

- Download the ZIP file from GitHub
- Extract it to your desired location
- Navigate to the extracted folder in terminal/command prompt

### 2. Create a Virtual Environment (Recommended)

Creating a virtual environment isolates project dependencies:

**On macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**

```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your command prompt when activated.

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:

- numpy, pandas, scipy (data processing)
- scikit-learn (machine learning)
- xgboost, lightgbm (advanced ML)
- matplotlib, seaborn, plotly (visualization)
- And other required packages

**Note**: If you encounter issues with XGBoost or LightGBM, you can install without them:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn plotly
```

The framework will still work, just without those specific models.

### 4. Verify Installation

Test that everything is installed correctly:

```bash
python -c "import numpy, pandas, sklearn, plotly; print('✓ All packages imported successfully!')"
```

### 5. Prepare the Data

Ensure the CSV files are in the `data/` folder:

```
data/
├── 1- mental-illnesses-prevalence.csv
├── 4- adult-population-covered-in-primary-data-on-the-prevalence-of-mental-illnesses.csv
├── 6- depressive-symptoms-across-us-population.csv
└── 7- number-of-countries-with-primary-data-on-prevalence-of-mental-illnesses-in-the-global-burden-of-disease-study.csv
```

### 6. Run the Analysis

```bash
python main.py
```

Expected output:

- Console output showing progress through each step
- Results saved to `results/` folder
- Visualizations saved to `figures/` folder

## Running Individual Modules

You can also run individual modules for testing:

```bash
# Test data loading
python src/data_loader.py

# Test feature engineering
python src/feature_engineering.py

# Test clustering
python src/clustering.py

# Test ML models
python src/ml_models.py

# Test visualization
python src/visualization.py
```

## Troubleshooting

### Common Issues

**Issue 1: ModuleNotFoundError**

```
Solution: Ensure virtual environment is activated and packages are installed
pip install -r requirements.txt
```

**Issue 2: Data files not found**

```
Solution: Check that CSV files are in the data/ folder with correct names
```

**Issue 3: Memory errors**

```
Solution: Reduce the number of models or use a smaller dataset
Edit src/config.py to adjust parameters
```

**Issue 4: XGBoost/LightGBM installation fails**

```
Solution: These are optional. The framework works without them.
pip install --no-deps xgboost lightgbm
```

**Issue 5: Plotting issues on remote servers**

```
Solution: Set matplotlib to use non-interactive backend
Add to beginning of main.py:
import matplotlib
matplotlib.use('Agg')
```

## System Requirements

- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: ~100MB for code + data
- **CPU**: Multi-core recommended for faster training
- **OS**: Windows, macOS, or Linux

## Expected Runtime

- Data loading: < 1 minute
- Feature engineering: < 1 minute
- Clustering: 1-2 minutes
- ML models: 2-5 minutes
- Ensemble: 2-3 minutes
- Visualization: 1-2 minutes

**Total: ~5-10 minutes** for complete pipeline

## Configuration

To customize the analysis, edit `src/config.py`:

- `RANDOM_STATE`: Change for different random splits
- `TEST_SIZE`: Adjust train/test split ratio
- `OPTIMAL_K`: Number of clusters
- `MODEL_PARAMS`: Hyperparameters for each model

## Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review error messages carefully
3. Ensure all prerequisites are met
4. Check GitHub issues for similar problems
5. Open a new issue with:
   - Your Python version
   - Operating system
   - Complete error message
   - Steps to reproduce

## Updating

To update to the latest version:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## Uninstallation

To remove the project:

```bash
# Deactivate virtual environment
deactivate

# Remove project folder
cd ..
rm -rf mental-health-prediction/
```

## Next Steps

After successful installation:

1. Read the README.md for detailed documentation
2. Run main.py to execute the full pipeline
3. Explore results in `results/` and `figures/`
4. Modify `src/config.py` for custom analyses
5. Adapt the code for your own datasets

---

**Need more help?** Contact: mfadliridhani@gmail.com
