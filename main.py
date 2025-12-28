"""
Main execution script for mental health prediction framework.
Orchestrates the complete analysis pipeline.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from clustering import ClusterAnalyzer
from ml_models import MLModels
from visualization import Visualizer
from config import RESULTS_DIR, FIGURES_DIR

import pandas as pd
import numpy as np


def print_section_header(title: str):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80 + "\n")


def main():
    """
    Execute complete mental health prediction analysis pipeline.
    
    Pipeline steps:
    1. Load and preprocess data
    2. Feature engineering
    3. Clustering analysis
    4. Machine learning modeling
    5. Ensemble methods
    6. Visualization and reporting
    """
    
    print_section_header("MENTAL HEALTH PREDICTION FRAMEWORK")
    print("Novel Approach: Feature Engineering + Clustering + Ensemble Learning")
    print("Repository: https://github.com/YOUR_USERNAME/mental-health-prediction")
    print("-"*80)
    
    # =========================================================================
    # STEP 1: DATA LOADING
    # =========================================================================
    print_section_header("STEP 1: DATA LOADING AND PREPROCESSING")
    
    loader = DataLoader()
    df = loader.load_datasets()
    summary_stats = loader.get_summary_statistics()
    
    # Save summary statistics
    summary_stats.to_csv(os.path.join(RESULTS_DIR, 'table1_summary_statistics.csv'), index=False)
    print("\n✓ Summary statistics saved")
    
    # =========================================================================
    # STEP 2: FEATURE ENGINEERING
    # =========================================================================
    print_section_header("STEP 2: FEATURE ENGINEERING")
    
    engineer = FeatureEngineer(df)
    df_engineered = engineer.create_all_features()
    
    print(f"\n✓ Feature engineering complete!")
    print(f"  - Original features: 5")
    print(f"  - Engineered features: {len(engineer.get_engineered_feature_names())}")
    print(f"  - Total features: {len(df_engineered.columns)}")
    
    # =========================================================================
    # STEP 3: CLUSTERING ANALYSIS
    # =========================================================================
    print_section_header("STEP 3: CLUSTERING ANALYSIS")
    
    analyzer = ClusterAnalyzer(df_engineered)
    df_clustered, cluster_eval = analyzer.run_complete_analysis()
    
    # Get cluster characteristics
    cluster_chars = analyzer.get_cluster_characteristics()
    
    # Save clustering results
    cluster_eval.to_csv(os.path.join(RESULTS_DIR, 'table3_clustering_comparison.csv'))
    print("\n✓ Clustering comparison saved")
    
    # =========================================================================
    # STEP 4: MACHINE LEARNING MODELS
    # =========================================================================
    print_section_header("STEP 4: MACHINE LEARNING MODELS")
    
    ml = MLModels(df_clustered)
    ml.prepare_data()
    ml.initialize_models()
    results_df = ml.train_and_evaluate_all()
    
    # =========================================================================
    # STEP 5: ENSEMBLE MODEL
    # =========================================================================
    print_section_header("STEP 5: STACKING ENSEMBLE MODEL")
    
    try:
        ensemble_model, ensemble_pred, ensemble_metrics = ml.create_stacking_ensemble()
        
        # Update results with ensemble
        results_df.loc['Stacking Ensemble'] = ensemble_metrics
        results_df = results_df.sort_values('Test R²', ascending=False)
        
    except Exception as e:
        print(f"Note: Ensemble creation skipped: {str(e)}")
    
    # Save ML results
    ml.save_results()
    
    # =========================================================================
    # STEP 6: FEATURE IMPORTANCE
    # =========================================================================
    print_section_header("STEP 6: FEATURE IMPORTANCE ANALYSIS")
    
    try:
        importance_df = ml.get_feature_importance('Random Forest')
    except Exception as e:
        print(f"Note: Feature importance extraction skipped: {str(e)}")
        importance_df = None
    
    # =========================================================================
    # STEP 7: VISUALIZATION
    # =========================================================================
    print_section_header("STEP 7: CREATING VISUALIZATIONS")
    
    viz = Visualizer()
    
    # Correlation matrix
    feature_cols = ['Schizophrenia', 'Depression', 'Anxiety', 'Bipolar', 'Eating',
                   'Depression_Anxiety', 'Total_Burden', 'Depression_sq', 'Anxiety_sq']
    viz.plot_correlation_matrix(df_clustered, feature_cols)
    
    # Clustering visualizations
    viz.plot_pca_clusters(df_clustered)
    viz.plot_tsne_clusters(df_clustered)
    viz.plot_cluster_radar(cluster_chars)
    
    # Model performance
    viz.plot_model_comparison(results_df)
    
    # Feature importance
    if importance_df is not None:
        viz.plot_feature_importance(importance_df)
    
    # Predictions vs actual
    selected_models = {}
    if 'Linear Regression (Baseline)' in ml.predictions:
        selected_models['Baseline'] = ml.predictions['Linear Regression (Baseline)']
    if 'Random Forest' in ml.predictions:
        selected_models['Random Forest'] = ml.predictions['Random Forest']
    if 'Stacking Ensemble' in ml.predictions:
        selected_models['Ensemble'] = ml.predictions['Stacking Ensemble']
    
    if selected_models:
        viz.plot_predictions_vs_actual(ml.y_test.values, selected_models)
    
    # Residuals for best model
    if 'Stacking Ensemble' in ml.predictions:
        viz.plot_residuals(ml.y_test.values, ml.predictions['Stacking Ensemble'], 
                          'Stacking Ensemble')
    
    print("\n✓ All visualizations created")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_section_header("ANALYSIS COMPLETE - FINAL SUMMARY")
    
    print("📊 KEY RESULTS:")
    print(f"   • Best Model: {results_df.index[0]}")
    print(f"   • Best R² Score: {results_df['Test R²'].iloc[0]:.4f}")
    print(f"   • Best RMSE: {results_df['Test RMSE'].iloc[0]:.4f}")
    
    if 'Linear Regression (Baseline)' in results_df.index:
        baseline_r2 = results_df.loc['Linear Regression (Baseline)', 'Test R²']
        best_r2 = results_df['Test R²'].iloc[0]
        improvement = ((best_r2 - baseline_r2) / baseline_r2) * 100
        print(f"   • Improvement over baseline: {improvement:.1f}%")
    
    print(f"\n🔬 NOVEL CONTRIBUTIONS:")
    print(f"   1. Advanced Feature Engineering: {len(engineer.get_engineered_feature_names())} new features")
    print(f"   2. Clustering Discovery: {analyzer.kmeans.n_clusters} distinct mental health profiles")
    print(f"   3. Multiple ML Algorithms: {len(ml.models)} models compared")
    print(f"   4. Ensemble Methods: Stacking of multiple base models")
    print(f"   5. Comprehensive Evaluation: Cross-validation + multiple metrics")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   • Results: {RESULTS_DIR}/")
    print(f"     - model_comparison_results.csv")
    print(f"     - feature_importance.csv")
    print(f"     - table1_summary_statistics.csv")
    print(f"     - table3_clustering_comparison.csv")
    print(f"   • Figures: {FIGURES_DIR}/")
    print(f"     - correlation_matrix.png")
    print(f"     - pca_clusters.html")
    print(f"     - tsne_clusters.html")
    print(f"     - cluster_radar.html")
    print(f"     - model_comparison.html")
    print(f"     - feature_importance.html")
    print(f"     - predictions_comparison.html")
    
    print("\n" + "="*80)
    print("✓ ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n📖 For reproducibility, see: https://github.com/YOUR_USERNAME/mental-health-prediction")
    print("📧 For questions or collaboration: your.email@example.com\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
