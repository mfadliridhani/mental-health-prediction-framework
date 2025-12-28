"""
Clustering analysis module.
Implements K-Means, DBSCAN, Hierarchical clustering with evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, Tuple
from config import (OPTIMAL_K, DBSCAN_EPS, DBSCAN_MIN_SAMPLES, 
                   PCA_COMPONENTS, TSNE_COMPONENTS, TSNE_PERPLEXITY, 
                   TSNE_N_ITER, RANDOM_STATE, BASE_FEATURES)


class ClusterAnalyzer:
    """Class for performing clustering analysis on mental health data."""
    
    def __init__(self, df: pd.DataFrame, features: list = None):
        """
        Initialize ClusterAnalyzer.
        
        Args:
            df: Input DataFrame
            features: List of features to use for clustering (default: BASE_FEATURES)
        """
        self.df = df.copy()
        self.features = features if features else [f for f in BASE_FEATURES if f in df.columns and f != 'Eating']
        self.scaler = StandardScaler()
        self.X_scaled = None
        self.kmeans = None
        self.dbscan = None
        self.hierarchical = None
        
    def prepare_data(self) -> np.ndarray:
        """
        Prepare and scale data for clustering.
        
        Returns:
            Scaled feature array
        """
        print(f"Preparing data for clustering with features: {self.features}")
        X = self.df[self.features].values
        self.X_scaled = self.scaler.fit_transform(X)
        print(f"✓ Data scaled, shape: {self.X_scaled.shape}")
        return self.X_scaled
    
    def find_optimal_clusters(self, k_range: range = range(2, 11)) -> list:
        """
        Use elbow method to find optimal number of clusters.
        
        Args:
            k_range: Range of k values to test
            
        Returns:
            List of inertia values
        """
        print(f"\nFinding optimal number of clusters...")
        inertias = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)
        
        print(f"✓ Elbow method complete. Recommended k: {OPTIMAL_K}")
        return inertias
    
    def apply_kmeans(self, n_clusters: int = OPTIMAL_K) -> pd.DataFrame:
        """
        Apply K-Means clustering.
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            DataFrame with cluster labels added
        """
        print(f"\nApplying K-Means clustering (k={n_clusters})...")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        self.df['KMeans_Cluster'] = self.kmeans.fit_predict(self.X_scaled)
        
        print(f"✓ K-Means complete. Clusters: {n_clusters}")
        print(f"Cluster distribution:")
        print(self.df['KMeans_Cluster'].value_counts().sort_index())
        
        return self.df
    
    def apply_dbscan(self, eps: float = DBSCAN_EPS, 
                     min_samples: int = DBSCAN_MIN_SAMPLES) -> pd.DataFrame:
        """
        Apply DBSCAN clustering.
        
        Args:
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood
            
        Returns:
            DataFrame with cluster labels added
        """
        print(f"\nApplying DBSCAN clustering (eps={eps}, min_samples={min_samples})...")
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.df['DBSCAN_Cluster'] = self.dbscan.fit_predict(self.X_scaled)
        
        n_clusters = len(set(self.df['DBSCAN_Cluster'])) - (1 if -1 in self.df['DBSCAN_Cluster'].values else 0)
        n_noise = (self.df['DBSCAN_Cluster'] == -1).sum()
        
        print(f"✓ DBSCAN complete. Clusters: {n_clusters}, Noise points: {n_noise}")
        
        return self.df
    
    def apply_hierarchical(self, n_clusters: int = OPTIMAL_K) -> pd.DataFrame:
        """
        Apply Hierarchical clustering.
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            DataFrame with cluster labels added
        """
        print(f"\nApplying Hierarchical clustering (k={n_clusters})...")
        self.hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        self.df['Hierarchical_Cluster'] = self.hierarchical.fit_predict(self.X_scaled)
        
        print(f"✓ Hierarchical clustering complete. Clusters: {n_clusters}")
        
        return self.df
    
    def apply_pca(self, n_components: int = PCA_COMPONENTS) -> pd.DataFrame:
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            n_components: Number of principal components
            
        Returns:
            DataFrame with PCA components added
        """
        print(f"\nApplying PCA (n_components={n_components})...")
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(self.X_scaled)
        
        for i in range(n_components):
            self.df[f'PCA{i+1}'] = X_pca[:, i]
        
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"✓ PCA complete. Total variance explained: {explained_var:.2%}")
        
        return self.df
    
    def apply_tsne(self, n_components: int = TSNE_COMPONENTS,
                   perplexity: int = TSNE_PERPLEXITY,
                   n_iter: int = TSNE_N_ITER) -> pd.DataFrame:
        """
        Apply t-SNE for dimensionality reduction.
        
        Args:
            n_components: Number of dimensions
            perplexity: t-SNE perplexity parameter
            n_iter: Number of iterations
            
        Returns:
            DataFrame with t-SNE components added
        """
        print(f"\nApplying t-SNE (this may take a moment)...")
        tsne = TSNE(n_components=n_components, random_state=RANDOM_STATE,
                   perplexity=perplexity, n_iter=n_iter)
        X_tsne = tsne.fit_transform(self.X_scaled)
        
        for i in range(n_components):
            self.df[f'TSNE{i+1}'] = X_tsne[:, i]
        
        print(f"✓ t-SNE complete.")
        
        return self.df
    
    def evaluate_clustering(self) -> pd.DataFrame:
        """
        Evaluate clustering algorithms using multiple metrics.
        
        Returns:
            DataFrame with evaluation metrics
        """
        print("\n" + "="*80)
        print("CLUSTERING EVALUATION")
        print("="*80 + "\n")
        
        results = {}
        
        # Evaluate K-Means
        if 'KMeans_Cluster' in self.df.columns:
            kmeans_labels = self.df['KMeans_Cluster']
            results['K-Means'] = {
                'Silhouette Score': silhouette_score(self.X_scaled, kmeans_labels),
                'Davies-Bouldin Index': davies_bouldin_score(self.X_scaled, kmeans_labels),
                'Calinski-Harabasz Index': calinski_harabasz_score(self.X_scaled, kmeans_labels)
            }
        
        # Evaluate DBSCAN
        if 'DBSCAN_Cluster' in self.df.columns:
            dbscan_labels = self.df['DBSCAN_Cluster']
            mask = dbscan_labels != -1
            
            if mask.sum() > 0 and len(set(dbscan_labels[mask])) > 1:
                results['DBSCAN'] = {
                    'Silhouette Score': silhouette_score(self.X_scaled[mask], dbscan_labels[mask]),
                    'Davies-Bouldin Index': davies_bouldin_score(self.X_scaled[mask], dbscan_labels[mask]),
                    'Calinski-Harabasz Index': calinski_harabasz_score(self.X_scaled[mask], dbscan_labels[mask])
                }
            else:
                results['DBSCAN'] = {
                    'Silhouette Score': np.nan,
                    'Davies-Bouldin Index': np.nan,
                    'Calinski-Harabasz Index': np.nan
                }
        
        # Evaluate Hierarchical
        if 'Hierarchical_Cluster' in self.df.columns:
            hier_labels = self.df['Hierarchical_Cluster']
            results['Hierarchical'] = {
                'Silhouette Score': silhouette_score(self.X_scaled, hier_labels),
                'Davies-Bouldin Index': davies_bouldin_score(self.X_scaled, hier_labels),
                'Calinski-Harabasz Index': calinski_harabasz_score(self.X_scaled, hier_labels)
            }
        
        results_df = pd.DataFrame(results).T
        
        print("Clustering Evaluation Metrics:")
        print(results_df.round(4))
        
        # Determine best algorithm
        valid_algos = results_df[results_df['Silhouette Score'].notna()]
        if len(valid_algos) > 0:
            best_silhouette = valid_algos['Silhouette Score'].idxmax()
            best_davies = valid_algos['Davies-Bouldin Index'].idxmin()
            best_calinski = valid_algos['Calinski-Harabasz Index'].idxmax()
            
            print(f"\n🏆 Best Algorithms:")
            print(f"  - Silhouette Score: {best_silhouette}")
            print(f"  - Davies-Bouldin Index: {best_davies}")
            print(f"  - Calinski-Harabasz Index: {best_calinski}")
        
        return results_df
    
    def get_cluster_characteristics(self, cluster_col: str = 'KMeans_Cluster') -> pd.DataFrame:
        """
        Analyze characteristics of each cluster.
        
        Args:
            cluster_col: Name of cluster column to analyze
            
        Returns:
            DataFrame with cluster characteristics
        """
        if cluster_col not in self.df.columns:
            raise ValueError(f"Cluster column '{cluster_col}' not found")
        
        cluster_chars = self.df.groupby(cluster_col)[self.features].mean()
        
        print(f"\n{cluster_col} Characteristics:")
        print(cluster_chars.round(3))
        
        return cluster_chars
    
    def run_complete_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run complete clustering analysis pipeline.
        
        Returns:
            Tuple of (DataFrame with clusters, evaluation metrics)
        """
        print("\n" + "="*80)
        print("CLUSTERING ANALYSIS")
        print("="*80)
        
        self.prepare_data()
        self.find_optimal_clusters()
        self.apply_kmeans()
        self.apply_dbscan()
        self.apply_hierarchical()
        self.apply_pca()
        self.apply_tsne()
        
        evaluation = self.evaluate_clustering()
        characteristics = self.get_cluster_characteristics()
        
        print("\n✓ Clustering analysis complete!")
        
        return self.df, evaluation


def main():
    """Test the clustering functionality."""
    # Create sample data
    np.random.seed(42)
    sample_data = {
        'Entity': [f'Country {i}' for i in range(100)],
        'Schizophrenia': np.random.uniform(0.2, 0.4, 100),
        'Depression': np.random.uniform(3.0, 5.0, 100),
        'Anxiety': np.random.uniform(3.5, 5.5, 100),
        'Bipolar': np.random.uniform(0.5, 0.9, 100),
        'Eating': np.random.uniform(0.2, 0.5, 100)
    }
    df = pd.DataFrame(sample_data)
    
    # Run clustering
    analyzer = ClusterAnalyzer(df)
    df_clustered, evaluation = analyzer.run_complete_analysis()
    
    print(f"\n✓ Clustering test complete!")
    print(f"  - Shape: {df_clustered.shape}")
    print(f"  - Clusters identified: {df_clustered['KMeans_Cluster'].nunique()}")


if __name__ == "__main__":
    main()
