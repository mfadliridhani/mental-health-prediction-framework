# Quick Start Guide

Get up and running in 5 minutes!

## 🚀 Quick Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/mental-health-prediction.git
cd mental-health-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the analysis
python main.py
```

That's it! Results will be saved in `results/` and `figures/` folders.

## 📋 What Gets Generated

After running `main.py`, you'll have:

### Results (CSV files in `results/`):

- `model_comparison_results.csv` - Performance metrics for all models
- `feature_importance.csv` - Top features ranked by importance
- `table1_summary_statistics.csv` - Descriptive statistics
- `table3_clustering_comparison.csv` - Clustering evaluation

### Figures (in `figures/`):

- `correlation_matrix.png` - Feature correlations
- `pca_clusters.html` - Interactive PCA visualization
- `tsne_clusters.html` - Interactive t-SNE visualization
- `cluster_radar.html` - Cluster characteristics
- `model_comparison.html` - Model performance comparison
- `feature_importance.html` - Top features bar chart
- `predictions_comparison.html` - Predictions vs actual
- `residuals.html` - Residual analysis

## 🎯 Key Results

Expected output:

```
Best Model: Stacking Ensemble
Test R² Score: 0.8547
Improvement over baseline: 15.2%

Novel Contributions:
1. Advanced Feature Engineering: 16 new features
2. Clustering Discovery: 5 distinct mental health profiles
3. Multiple ML Algorithms: 10 models compared
4. Ensemble Methods: Stacking of multiple base models
```

## 🔧 Common Customizations

### Change number of clusters

Edit `src/config.py`:

```python
OPTIMAL_K = 5  # Change to desired number
```

### Use different test split

Edit `src/config.py`:

```python
TEST_SIZE = 0.2  # Change to 0.3 for 30% test set
```

### Run without ensemble (faster)

Comment out ensemble section in `main.py`:

```python
# print_section_header("STEP 5: STACKING ENSEMBLE MODEL")
# ...ensemble code...
```

## 🐍 Using Individual Components

### Just load data

```python
from src.data_loader import DataLoader
loader = DataLoader()
df = loader.load_datasets()
```

### Just create features

```python
from src.feature_engineering import FeatureEngineer
engineer = FeatureEngineer(df)
df_engineered = engineer.create_all_features()
```

### Just run clustering

```python
from src.clustering import ClusterAnalyzer
analyzer = ClusterAnalyzer(df_engineered)
df_clustered, evaluation = analyzer.run_complete_analysis()
```

### Just train models

```python
from src.ml_models import MLModels
ml = MLModels(df_clustered)
ml.prepare_data()
ml.initialize_models()
results = ml.train_and_evaluate_all()
```

## 📊 View Results

### Open HTML visualizations

Just double-click any `.html` file in `figures/` folder to view in browser.

### Load CSV results

```python
import pandas as pd
results = pd.read_csv('results/model_comparison_results.csv')
print(results)
```

## ⚡ Performance Tips

### Faster execution

1. Reduce cross-validation folds: `CV_FOLDS = 3` in `config.py`
2. Skip ensemble model (comment out in `main.py`)
3. Use fewer models (comment out in `ml_models.py`)

### Lower memory usage

1. Reduce number of features
2. Use smaller test size
3. Skip visualization generation

## 🆘 Getting Help

**Error: ModuleNotFoundError**

```bash
pip install -r requirements.txt
```

**Error: File not found**

```bash
# Check data files are in data/ folder
ls data/
```

**Error: Memory error**

```python
# Reduce dataset size in data_loader.py
df = df.sample(n=1000, random_state=42)
```

## 📚 Next Steps

1. ✅ Run `python main.py` successfully
2. 📖 Read full [README.md](README.md) for details
3. 🔧 Customize parameters in [src/config.py](src/config.py)
4. 📊 Explore results in `results/` and `figures/`
5. 🚀 Adapt for your own dataset

## 🎓 For Your Paper

Update your paper's reproducibility section:

```
The complete implementation is available at:
https://github.com/YOUR_USERNAME/mental-health-prediction

To reproduce our results:
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Run: python main.py
4. Results will be generated in results/ and figures/ folders

Approximate runtime: 5-10 minutes on standard hardware.
```

---

**Questions?** Open an issue on GitHub or email: your.email@example.com
