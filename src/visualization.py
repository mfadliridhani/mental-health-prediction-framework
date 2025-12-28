"""
Visualization module.
Creates publication-quality plots and figures.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional
import os

from config import FIGURES_DIR, PLOT_WIDTH, PLOT_HEIGHT, DPI, COLOR_PALETTE, BASE_FEATURES


class Visualizer:
    """Class for creating visualizations of mental health data and model results."""
    
    def __init__(self, output_dir: str = FIGURES_DIR):
        """
        Initialize Visualizer.
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = DPI
    
    def plot_correlation_matrix(self, df: pd.DataFrame, features: List[str],
                                save: bool = True, filename: str = 'correlation_matrix.png'):
        """
        Plot correlation matrix heatmap.
        
        Args:
            df: Input DataFrame
            features: List of features to include
            save: Whether to save the figure
            filename: Output filename
        """
        plt.figure(figsize=(12, 10), dpi=DPI)
        correlation_matrix = df[features].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1)
        plt.title('Correlation Matrix - Mental Health Features', fontsize=16, pad=20)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.close()
    
    def plot_elbow_curve(self, k_range: range, inertias: List[float],
                        save: bool = True, filename: str = 'elbow_curve.png'):
        """
        Plot elbow curve for K-Means clustering.
        
        Args:
            k_range: Range of k values
            inertias: List of inertia values
            save: Whether to save the figure
            filename: Output filename
        """
        plt.figure(figsize=(10, 6), dpi=DPI)
        plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Inertia', fontsize=12)
        plt.title('Elbow Method for Optimal k', fontsize=14, pad=15)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.close()
    
    def plot_pca_clusters(self, df: pd.DataFrame, cluster_col: str = 'KMeans_Cluster',
                         save: bool = True, filename: str = 'pca_clusters.html'):
        """
        Create interactive PCA visualization with clusters.
        
        Args:
            df: DataFrame with PCA components and clusters
            cluster_col: Name of cluster column
            save: Whether to save the figure
            filename: Output filename
        """
        if 'PCA1' not in df.columns or 'PCA2' not in df.columns:
            print("Warning: PCA components not found in DataFrame")
            return
        
        fig = px.scatter(
            df, x='PCA1', y='PCA2',
            color=cluster_col,
            hover_data=['Entity'] if 'Entity' in df.columns else None,
            title='Mental Health Clusters - PCA Visualization',
            labels={cluster_col: 'Cluster'},
            color_continuous_scale=COLOR_PALETTE,
            width=PLOT_WIDTH, height=PLOT_HEIGHT
        )
        
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
        fig.update_layout(font=dict(size=12))
        
        if save:
            save_path = os.path.join(self.output_dir, filename)
            fig.write_html(save_path)
            print(f"✓ Saved: {save_path}")
        
        return fig
    
    def plot_tsne_clusters(self, df: pd.DataFrame, cluster_col: str = 'KMeans_Cluster',
                          save: bool = True, filename: str = 'tsne_clusters.html'):
        """
        Create interactive t-SNE visualization with clusters.
        
        Args:
            df: DataFrame with t-SNE components and clusters
            cluster_col: Name of cluster column
            save: Whether to save the figure
            filename: Output filename
        """
        if 'TSNE1' not in df.columns or 'TSNE2' not in df.columns:
            print("Warning: t-SNE components not found in DataFrame")
            return
        
        fig = px.scatter(
            df, x='TSNE1', y='TSNE2',
            color=cluster_col,
            hover_data=['Entity'] if 'Entity' in df.columns else None,
            title='Mental Health Clusters - t-SNE Visualization',
            labels={cluster_col: 'Cluster'},
            color_continuous_scale='plasma',
            width=PLOT_WIDTH, height=PLOT_HEIGHT
        )
        
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
        
        if save:
            save_path = os.path.join(self.output_dir, filename)
            fig.write_html(save_path)
            print(f"✓ Saved: {save_path}")
        
        return fig
    
    def plot_cluster_radar(self, cluster_chars: pd.DataFrame,
                          save: bool = True, filename: str = 'cluster_radar.html'):
        """
        Create radar chart showing cluster characteristics.
        
        Args:
            cluster_chars: DataFrame with cluster characteristics
            save: Whether to save the figure
            filename: Output filename
        """
        fig = go.Figure()
        
        for idx in cluster_chars.index:
            fig.add_trace(go.Scatterpolar(
                r=cluster_chars.loc[idx].values,
                theta=cluster_chars.columns,
                fill='toself',
                name=f'Cluster {idx}'
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, 
                                       range=[0, cluster_chars.max().max() * 1.1])),
            title='Mental Health Profile by Cluster',
            showlegend=True,
            width=800, height=600
        )
        
        if save:
            save_path = os.path.join(self.output_dir, filename)
            fig.write_html(save_path)
            print(f"✓ Saved: {save_path}")
        
        return fig
    
    def plot_model_comparison(self, results_df: pd.DataFrame,
                             save: bool = True, filename: str = 'model_comparison.html'):
        """
        Create bar chart comparing model performance.
        
        Args:
            results_df: DataFrame with model results
            save: Whether to save the figure
            filename: Output filename
        """
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=results_df.index,
            y=results_df['Test R²'],
            name='Test R²',
            marker_color='lightblue',
            error_y=dict(type='data', array=results_df['CV Std'])
        ))
        
        fig.update_layout(
            title='Model Performance Comparison (R² Score)',
            xaxis_title='Model',
            yaxis_title='R² Score',
            yaxis_range=[0, 1],
            template='plotly_white',
            width=1000,
            height=600,
            xaxis_tickangle=-45
        )
        
        # Add baseline reference
        fig.add_hline(y=0.70, line_dash="dash", line_color="red",
                     annotation_text="Previous Baseline (70%)")
        
        if save:
            save_path = os.path.join(self.output_dir, filename)
            fig.write_html(save_path)
            print(f"✓ Saved: {save_path}")
        
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 15,
                               save: bool = True, filename: str = 'feature_importance.html'):
        """
        Create horizontal bar chart of feature importance.
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to show
            save: Whether to save the figure
            filename: Output filename
        """
        top_features = importance_df.head(top_n)
        
        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Top {top_n} Feature Importance (Random Forest)',
            labels={'Importance': 'Feature Importance Score'},
            color='Importance',
            color_continuous_scale=COLOR_PALETTE
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            width=PLOT_WIDTH,
            height=PLOT_HEIGHT
        )
        
        if save:
            save_path = os.path.join(self.output_dir, filename)
            fig.write_html(save_path)
            print(f"✓ Saved: {save_path}")
        
        return fig
    
    def plot_predictions_vs_actual(self, y_test: np.ndarray, predictions: dict,
                                   save: bool = True, filename: str = 'predictions_comparison.html'):
        """
        Create scatter plot comparing predictions to actual values.
        
        Args:
            y_test: Actual test values
            predictions: Dictionary of model predictions
            save: Whether to save the figure
            filename: Output filename
        """
        fig = go.Figure()
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=list(range(len(y_test))),
            y=y_test,
            mode='markers',
            name='Actual',
            marker=dict(size=10, color='black', symbol='circle')
        ))
        
        # Add prediction traces
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (name, pred) in enumerate(predictions.items()):
            fig.add_trace(go.Scatter(
                x=list(range(len(y_test))),
                y=pred,
                mode='markers',
                name=name,
                marker=dict(size=8, color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            title='Prediction Comparison: Actual vs Models',
            xaxis_title='Sample Index',
            yaxis_title='Eating Disorder Prevalence',
            template='plotly_white',
            width=1000,
            height=600
        )
        
        if save:
            save_path = os.path.join(self.output_dir, filename)
            fig.write_html(save_path)
            print(f"✓ Saved: {save_path}")
        
        return fig
    
    def plot_residuals(self, y_test: np.ndarray, y_pred: np.ndarray,
                      model_name: str = 'Model',
                      save: bool = True, filename: str = 'residuals.html'):
        """
        Create residual plot.
        
        Args:
            y_test: Actual test values
            y_pred: Predicted values
            model_name: Name of the model
            save: Whether to save the figure
            filename: Output filename
        """
        residuals = y_test - y_pred
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.6)
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title=f'Residual Plot - {model_name}',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals',
            template='plotly_white',
            width=PLOT_WIDTH,
            height=PLOT_HEIGHT
        )
        
        if save:
            save_path = os.path.join(self.output_dir, filename)
            fig.write_html(save_path)
            print(f"✓ Saved: {save_path}")
        
        return fig


def main():
    """Test the visualization functionality."""
    # Create sample data
    np.random.seed(42)
    sample_data = {
        'Depression': np.random.uniform(3, 5, 100),
        'Anxiety': np.random.uniform(3.5, 5.5, 100),
        'Bipolar': np.random.uniform(0.5, 0.9, 100),
        'PCA1': np.random.randn(100),
        'PCA2': np.random.randn(100),
        'KMeans_Cluster': np.random.randint(0, 5, 100),
        'Entity': [f'Country {i}' for i in range(100)]
    }
    df = pd.DataFrame(sample_data)
    
    # Create visualizer
    viz = Visualizer()
    
    # Test some plots
    viz.plot_correlation_matrix(df, ['Depression', 'Anxiety', 'Bipolar'])
    viz.plot_pca_clusters(df)
    
    print("\n✓ Visualization test complete!")


if __name__ == "__main__":
    main()
