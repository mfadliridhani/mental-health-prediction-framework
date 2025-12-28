"""
Feature engineering module.
Creates interaction, polynomial, ratio, and aggregate features.
"""

import pandas as pd
import numpy as np
from typing import List
from config import BASE_FEATURES


class FeatureEngineer:
    """Class for creating advanced features from mental health data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize FeatureEngineer.
        
        Args:
            df: Input DataFrame with base features
        """
        self.df = df.copy()
        self.original_features = len(df.columns)
        
    def create_interaction_features(self) -> pd.DataFrame:
        """
        Create interaction features (comorbidity patterns).
        
        Returns:
            DataFrame with interaction features added
        """
        print("Creating interaction features...")
        
        # Mental health disorder interactions
        self.df['Depression_Anxiety'] = self.df['Depression'] * self.df['Anxiety']
        self.df['Bipolar_Anxiety'] = self.df['Bipolar'] * self.df['Anxiety']
        self.df['Schizophrenia_Depression'] = self.df['Schizophrenia'] * self.df['Depression']
        self.df['Depression_Bipolar'] = self.df['Depression'] * self.df['Bipolar']
        
        print("✓ 4 interaction features created")
        return self.df
    
    def create_polynomial_features(self) -> pd.DataFrame:
        """
        Create polynomial features (non-linear relationships).
        
        Returns:
            DataFrame with polynomial features added
        """
        print("Creating polynomial features...")
        
        # Square terms for key disorders
        self.df['Depression_sq'] = self.df['Depression'] ** 2
        self.df['Anxiety_sq'] = self.df['Anxiety'] ** 2
        self.df['Bipolar_sq'] = self.df['Bipolar'] ** 2
        self.df['Schizophrenia_sq'] = self.df['Schizophrenia'] ** 2
        
        print("✓ 4 polynomial features created")
        return self.df
    
    def create_ratio_features(self) -> pd.DataFrame:
        """
        Create ratio features (relative proportions).
        
        Returns:
            DataFrame with ratio features added
        """
        print("Creating ratio features...")
        
        # Ratios between disorders (add small epsilon to avoid division by zero)
        eps = 1e-6
        self.df['Depression_to_Anxiety'] = self.df['Depression'] / (self.df['Anxiety'] + eps)
        self.df['Bipolar_to_Schizophrenia'] = self.df['Bipolar'] / (self.df['Schizophrenia'] + eps)
        self.df['Anxiety_to_Depression'] = self.df['Anxiety'] / (self.df['Depression'] + eps)
        
        print("✓ 3 ratio features created")
        return self.df
    
    def create_aggregate_features(self) -> pd.DataFrame:
        """
        Create aggregate features (total burden).
        
        Returns:
            DataFrame with aggregate features added
        """
        print("Creating aggregate features...")
        
        # Total mental health burden
        disorder_cols = ['Schizophrenia', 'Depression', 'Anxiety', 'Bipolar']
        self.df['Total_Burden'] = self.df[disorder_cols].sum(axis=1)
        self.df['Mean_Disorder'] = self.df[disorder_cols].mean(axis=1)
        self.df['Std_Disorder'] = self.df[disorder_cols].std(axis=1)
        
        print("✓ 3 aggregate features created")
        return self.df
    
    def create_log_features(self) -> pd.DataFrame:
        """
        Create log-transformed features (for skewed distributions).
        
        Returns:
            DataFrame with log features added
        """
        print("Creating log transformation features...")
        
        # Log transformations
        self.df['log_Depression'] = np.log1p(self.df['Depression'])
        self.df['log_Anxiety'] = np.log1p(self.df['Anxiety'])
        
        print("✓ 2 log transformation features created")
        return self.df
    
    def create_all_features(self) -> pd.DataFrame:
        """
        Create all engineered features at once.
        
        Returns:
            DataFrame with all engineered features
        """
        print("\n" + "="*80)
        print("FEATURE ENGINEERING")
        print("="*80 + "\n")
        
        self.create_interaction_features()
        self.create_polynomial_features()
        self.create_ratio_features()
        self.create_aggregate_features()
        self.create_log_features()
        
        new_features = len(self.df.columns) - self.original_features
        
        print(f"\n✓ Total new features created: {new_features}")
        print(f"✓ Total features: {len(self.df.columns)}")
        
        return self.df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names.
        
        Returns:
            List of feature column names
        """
        return self.df.columns.tolist()
    
    def get_engineered_feature_names(self) -> List[str]:
        """
        Get list of only engineered feature names.
        
        Returns:
            List of engineered feature names
        """
        base_cols = set(BASE_FEATURES + ['Entity', 'Code', 'Year'])
        all_cols = set(self.df.columns)
        return list(all_cols - base_cols)


def main():
    """Test the feature engineering functionality."""
    # Create sample data
    sample_data = {
        'Entity': ['Country A', 'Country B'],
        'Code': ['CA', 'CB'],
        'Year': [2019, 2019],
        'Schizophrenia': [0.25, 0.30],
        'Depression': [3.5, 4.0],
        'Anxiety': [4.0, 4.5],
        'Bipolar': [0.6, 0.7],
        'Eating': [0.3, 0.35]
    }
    df = pd.DataFrame(sample_data)
    
    # Create features
    engineer = FeatureEngineer(df)
    df_engineered = engineer.create_all_features()
    
    print("\n✓ Feature engineering complete!")
    print(f"  - Original features: {len(sample_data.keys())}")
    print(f"  - Final features: {len(df_engineered.columns)}")
    print(f"\nSample of engineered features:")
    print(df_engineered[['Depression_Anxiety', 'Total_Burden', 'log_Depression']].head())


if __name__ == "__main__":
    main()
