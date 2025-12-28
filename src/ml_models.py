"""
Machine learning models module.
Implements multiple regression models and ensemble methods.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             StackingRegressor, VotingRegressor)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from config import (RANDOM_STATE, TEST_SIZE, CV_FOLDS, MODEL_PARAMS, 
                   TARGET_VARIABLE, EXCLUDE_COLUMNS, RESULTS_DIR)

# Try importing gradient boosting libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Note: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Note: LightGBM not available. Install with: pip install lightgbm")


class MLModels:
    """Class for training and evaluating multiple ML models."""
    
    def __init__(self, df: pd.DataFrame, target: str = TARGET_VARIABLE):
        """
        Initialize MLModels.
        
        Args:
            df: Input DataFrame
            target: Target variable name
        """
        self.df = df.copy()
        self.target = target
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.predictions = {}
        
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare train/test splits and scale features.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("Preparing data for machine learning...")
        
        # Select features (exclude metadata and target)
        feature_cols = [col for col in self.df.columns if col not in EXCLUDE_COLUMNS]
        X = self.df[feature_cols].fillna(0)
        y = self.df[self.target]
        
        print(f"  - Features: {len(feature_cols)}")
        print(f"  - Samples: {len(X)}")
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"  - Training set: {self.X_train_scaled.shape}")
        print(f"  - Test set: {self.X_test_scaled.shape}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def initialize_models(self) -> Dict:
        """
        Initialize all ML models with their parameters.
        
        Returns:
            Dictionary of model instances
        """
        print("\nInitializing models...")
        
        self.models = {
            'Linear Regression (Baseline)': LinearRegression(),
            'Ridge Regression': Ridge(**MODEL_PARAMS['ridge']),
            'Lasso Regression': Lasso(**MODEL_PARAMS['lasso']),
            'ElasticNet': ElasticNet(**MODEL_PARAMS['elastic_net']),
            'Random Forest': RandomForestRegressor(**MODEL_PARAMS['random_forest']),
            'Gradient Boosting': GradientBoostingRegressor(**MODEL_PARAMS['gradient_boosting']),
            'SVR': SVR(**MODEL_PARAMS['svr']),
            'Neural Network': MLPRegressor(**MODEL_PARAMS['neural_network'])
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            self.models['XGBoost'] = xgb.XGBRegressor(**MODEL_PARAMS['xgboost'])
        
        # Add LightGBM if available
        if HAS_LIGHTGBM:
            self.models['LightGBM'] = lgb.LGBMRegressor(**MODEL_PARAMS['lightgbm'])
        
        print(f"✓ {len(self.models)} models initialized")
        return self.models
    
    def train_and_evaluate_all(self) -> pd.DataFrame:
        """
        Train and evaluate all models.
        
        Returns:
            DataFrame with evaluation metrics for all models
        """
        print("\n" + "="*80)
        print("TRAINING AND EVALUATING MODELS")
        print("="*80 + "\n")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            try:
                # Train
                model.fit(self.X_train_scaled, self.y_train)
                
                # Predictions
                y_pred_train = model.predict(self.X_train_scaled)
                y_pred_test = model.predict(self.X_test_scaled)
                
                # Metrics
                train_r2 = r2_score(self.y_train, y_pred_train)
                test_r2 = r2_score(self.y_test, y_pred_test)
                test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
                test_mae = mean_absolute_error(self.y_test, y_pred_test)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, self.X_train_scaled, self.y_train,
                    cv=CV_FOLDS, scoring='r2', n_jobs=-1
                )
                
                # Store results
                self.results[name] = {
                    'Train R²': train_r2,
                    'Test R²': test_r2,
                    'Test RMSE': test_rmse,
                    'Test MAE': test_mae,
                    'CV Mean': cv_scores.mean(),
                    'CV Std': cv_scores.std()
                }
                
                self.predictions[name] = y_pred_test
                
                print(f"  ✓ Test R²: {test_r2:.4f} | CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
                
            except Exception as e:
                print(f"  ✗ Error training {name}: {str(e)}\n")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('Test R²', ascending=False)
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        print(results_df.round(4))
        
        # Best model info
        best_model = results_df.index[0]
        best_r2 = results_df['Test R²'].iloc[0]
        
        if 'Linear Regression (Baseline)' in results_df.index:
            baseline_r2 = results_df.loc['Linear Regression (Baseline)', 'Test R²']
            improvement = ((best_r2 - baseline_r2) / baseline_r2) * 100
            
            print(f"\n🏆 Best Model: {best_model}")
            print(f"📊 Test R² Score: {best_r2:.4f}")
            print(f"📈 Improvement over Baseline: {improvement:.1f}%")
        else:
            print(f"\n🏆 Best Model: {best_model}")
            print(f"📊 Test R² Score: {best_r2:.4f}")
        
        return results_df
    
    def create_stacking_ensemble(self) -> Tuple:
        """
        Create and train stacking ensemble model.
        
        Returns:
            Tuple of (ensemble_model, predictions, metrics)
        """
        print("\n" + "="*80)
        print("CREATING STACKING ENSEMBLE")
        print("="*80 + "\n")
        
        # Define base models
        base_models = [
            ('rf', RandomForestRegressor(**MODEL_PARAMS['random_forest'])),
            ('gb', GradientBoostingRegressor(**MODEL_PARAMS['gradient_boosting'])),
            ('svr', SVR(**MODEL_PARAMS['svr'])),
            ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, 
                                random_state=RANDOM_STATE, early_stopping=True))
        ]
        
        if HAS_XGBOOST:
            base_models.append(('xgb', xgb.XGBRegressor(**MODEL_PARAMS['xgboost'])))
        
        if HAS_LIGHTGBM:
            base_models.append(('lgb', lgb.LGBMRegressor(**MODEL_PARAMS['lightgbm'])))
        
        print(f"Base models: {len(base_models)}")
        
        # Meta-model
        meta_model = Ridge(**MODEL_PARAMS['ridge'])
        
        # Create ensemble
        ensemble_model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=CV_FOLDS,
            n_jobs=-1
        )
        
        print("Training ensemble (this may take a few minutes)...")
        ensemble_model.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate
        y_pred_train = ensemble_model.predict(self.X_train_scaled)
        y_pred_test = ensemble_model.predict(self.X_test_scaled)
        
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        
        # Cross-validation
        print("Performing cross-validation on ensemble...")
        cv_scores = cross_val_score(
            ensemble_model, self.X_train_scaled, self.y_train,
            cv=CV_FOLDS, scoring='r2', n_jobs=-1
        )
        
        metrics = {
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae,
            'CV Mean': cv_scores.mean(),
            'CV Std': cv_scores.std()
        }
        
        print("\n✓ Ensemble trained successfully!")
        print(f"  - Train R²: {train_r2:.4f}")
        print(f"  - Test R²: {test_r2:.4f}")
        print(f"  - Test RMSE: {test_rmse:.4f}")
        print(f"  - CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Add to results
        self.results['Stacking Ensemble'] = metrics
        self.predictions['Stacking Ensemble'] = y_pred_test
        
        return ensemble_model, y_pred_test, metrics
    
    def get_feature_importance(self, model_name: str = 'Random Forest') -> pd.DataFrame:
        """
        Extract feature importance from tree-based model.
        
        Args:
            model_name: Name of the model to extract importance from
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model '{model_name}' does not have feature_importances_")
        
        feature_names = [col for col in self.df.columns if col not in EXCLUDE_COLUMNS]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop 15 Most Important Features ({model_name}):")
        print(importance_df.head(15).to_string(index=False))
        
        return importance_df
    
    def save_results(self, output_dir: str = RESULTS_DIR):
        """
        Save model results to CSV files.
        
        Args:
            output_dir: Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model comparison
        results_df = pd.DataFrame(self.results).T
        results_path = os.path.join(output_dir, 'model_comparison_results.csv')
        results_df.to_csv(results_path)
        print(f"\n✓ Results saved to: {results_path}")
        
        # Save feature importance if available
        try:
            importance_df = self.get_feature_importance('Random Forest')
            importance_path = os.path.join(output_dir, 'feature_importance.csv')
            importance_df.to_csv(importance_path, index=False)
            print(f"✓ Feature importance saved to: {importance_path}")
        except:
            pass


def main():
    """Test the ML models functionality."""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'Entity': [f'Country {i}' for i in range(n_samples)],
        'Code': [f'C{i}' for i in range(n_samples)],
        'Year': [2019] * n_samples,
        'Schizophrenia': np.random.uniform(0.2, 0.4, n_samples),
        'Depression': np.random.uniform(3.0, 5.0, n_samples),
        'Anxiety': np.random.uniform(3.5, 5.5, n_samples),
        'Bipolar': np.random.uniform(0.5, 0.9, n_samples),
        'Eating': np.random.uniform(0.2, 0.5, n_samples),
        'Depression_Anxiety': np.random.uniform(10, 25, n_samples),
        'Total_Burden': np.random.uniform(7, 11, n_samples)
    }
    df = pd.DataFrame(sample_data)
    
    # Train models
    ml = MLModels(df)
    ml.prepare_data()
    ml.initialize_models()
    results = ml.train_and_evaluate_all()
    
    print(f"\n✓ ML models test complete!")
    print(f"  - Models trained: {len(ml.models)}")
    print(f"  - Best R²: {results['Test R²'].iloc[0]:.4f}")


if __name__ == "__main__":
    main()
