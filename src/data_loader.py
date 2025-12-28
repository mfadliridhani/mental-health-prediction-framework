"""
Data loading and preprocessing module.
Handles data loading, cleaning, and initial preprocessing.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple
from config import DATA_DIR, DATA_FILES, COLUMN_MAPPING, BASE_FEATURES


class DataLoader:
    """Class for loading and preprocessing mental health datasets."""
    
    def __init__(self, data_dir: str = DATA_DIR):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = data_dir
        self.df = None
        
    def load_datasets(self) -> pd.DataFrame:
        """
        Load all datasets from CSV files.
        
        Returns:
            Combined DataFrame with mental health data
        """
        print("Loading datasets...")
        
        # Load main prevalence dataset
        prevalence_path = os.path.join(self.data_dir, DATA_FILES['prevalence'])
        df = pd.read_csv(prevalence_path)
        
        # Rename columns for easier manipulation
        df = df.rename(columns=COLUMN_MAPPING)
        
        print(f"✓ Loaded {len(df)} records")
        print(f"  - Entities: {df['Entity'].nunique()}")
        print(f"  - Years: {df['Year'].min()} - {df['Year'].max()}")
        
        self.df = df
        return df
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Calculate comprehensive summary statistics.
        
        Returns:
            DataFrame with summary statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_datasets() first.")
        
        summary_vars = [col for col in BASE_FEATURES if col in self.df.columns]
        
        summary_stats = pd.DataFrame({
            'Variable': summary_vars,
            'Mean': [self.df[var].mean() for var in summary_vars],
            'Std Dev': [self.df[var].std() for var in summary_vars],
            'Min': [self.df[var].min() for var in summary_vars],
            'Max': [self.df[var].max() for var in summary_vars],
            'Skewness': [self.df[var].skew() for var in summary_vars],
            'Kurtosis': [self.df[var].kurtosis() for var in summary_vars]
        })
        
        print("\n" + "="*80)
        print("DATASET SUMMARY STATISTICS")
        print("="*80)
        print(summary_stats.round(4).to_string(index=False))
        
        # Additional statistics
        print(f"\nTotal Observations: {len(self.df):,}")
        print(f"Total Countries/Entities: {self.df['Entity'].nunique():,}")
        print(f"Year Range: {self.df['Year'].min()} - {self.df['Year'].max()}")
        
        # Missing values
        print(f"\nMissing Values:")
        for var in summary_vars:
            missing_pct = (self.df[var].isna().sum() / len(self.df)) * 100
            print(f"  - {var}: {self.df[var].isna().sum()} ({missing_pct:.2f}%)")
        
        return summary_stats
    
    def check_data_quality(self) -> Dict:
        """
        Perform data quality checks.
        
        Returns:
            Dictionary with data quality metrics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_datasets() first.")
        
        quality_metrics = {
            'total_records': len(self.df),
            'total_features': len(self.df.columns),
            'missing_values': self.df.isna().sum().to_dict(),
            'duplicate_rows': self.df.duplicated().sum(),
            'unique_entities': self.df['Entity'].nunique()
        }
        
        return quality_metrics


def main():
    """Test the data loader functionality."""
    loader = DataLoader()
    df = loader.load_datasets()
    summary = loader.get_summary_statistics()
    quality = loader.check_data_quality()
    
    print("\n✓ Data loading complete!")
    print(f"  - Shape: {df.shape}")
    print(f"  - Duplicates: {quality['duplicate_rows']}")


if __name__ == "__main__":
    main()
