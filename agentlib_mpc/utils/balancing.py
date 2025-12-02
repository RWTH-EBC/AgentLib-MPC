import pandas as pd
import numpy as np
from enum import Enum
from typing import Callable, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt


class FilteringMethodTimeSeries(Enum):
    ROC = "rate_of_change"
    VOL = "volatility"
    KNN = "k_nearest_neighbors"
    CaS = "clustering_and_sampling"

class FilteringMethodTrainingSamples(Enum):
    CaS = "clustering_and_sampling"


def filter_time_series(time_series: pd.DataFrame, method: FilteringMethodTimeSeries, window_size: int = 50, threshold: float = None, verbose: bool = False) -> pd.DataFrame:
    """
    Filter dataframe of time series data to remove less insightful parts using different methods.
    Args:
        time_series: DataFrame with time series data (time index, multiple columns)
        method: FilteringMethod enum value specifying the filtering method to use
        window_size: Size of the moving window for filtering (default: 50)
        threshold: Threshold value for filtering (method-dependent)
        verbose: If True, plot the original and filtered data for comparison.
    Returns:
        Tuple of filtered DataFrames: (df_inputs_filtered, df_targets_filtered).
    """

    filter_strategies: Dict[FilteringMethodTimeSeries, Callable] = {
        FilteringMethodTimeSeries.ROC: _filter_by_rate_of_change,
        FilteringMethodTimeSeries.VOL: _filter_by_volatility,
        FilteringMethodTimeSeries.KNN: _filter_by_knn
    }

    if time_series.empty:
        raise ValueError("Input time series is empty.")

    time_series_filtered = time_series.copy()
    time_series_filtered = time_series_filtered.ffill().bfill()

    if method in filter_strategies:
        time_series_filtered = filter_strategies[method](time_series_filtered, window_size, threshold)
    else:
        raise ValueError(f"Unknown filtering method: {method}")
    
    if verbose:
        print(f"Filtering method: {method.value}, original points: {len(time_series)}, filtered points: {len(time_series_filtered)}")
        plot_filtering(time_series, time_series_filtered, title=f"Filtering: {method.value}")
    
    return time_series_filtered

from typing import Tuple

def filter_training_samples(df_inputs: pd.DataFrame, df_targets: pd.DataFrame, method: FilteringMethodTrainingSamples  , threshold: float = None, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter training samples in the DataFrame using specified filtering method.
    
    Args:
        df_inputs: DataFrame with input time series data (time index, multiple columns)
        df_targets: DataFrame with target time series data (time index, multiple columns)
        method: FilteringMethod enum value specifying the filtering method to use
        threshold: Threshold value for filtering (method-dependent). For clustering, refers to the number of clusters.
        verbose: If True, plot the original and filtered data for comparison.
    Returns:
        Filtered DataFrames (inputs and targets) with less insightful parts removed.
    """

    filter_strategies: Dict[FilteringMethodTrainingSamples, Callable] = {
        FilteringMethodTrainingSamples.CaS: _filter_by_clustering_and_sampling_training_samples,
    }

    if df_inputs.empty or df_targets.empty:
        raise ValueError("Input DataFrames are empty.")
    
    if method in filter_strategies:
        df_inputs_filtered, df_targets_filtered = filter_strategies[method](df_inputs, df_targets, int(threshold))
    else:
        raise ValueError(f"Unknown filtering method for training samples: {method}")

    return df_inputs_filtered, df_targets_filtered



def remove_gaps(df: pd.DataFrame, gap_threshold: float = 3600, verbose: bool = False) -> pd.DataFrame:
    """
    Remove large gaps in time series data by shifting data points behind gaps back in time.
    
    Args:
        df: DataFrame with numeric time index in seconds
        gap_threshold: Maximum allowed gap in seconds (default: 3600 = 1 hour)
        verbose: If True, plot the original and filtered data for comparison.
    Returns:
        DataFrame with gaps removed by shifting time index.
    """
    df = df.sort_index()
    
    df_index = df.index.to_series()
    time_diff = df_index.diff()
    gaps = time_diff > gap_threshold
    
    num_gaps = gaps.sum()
    print(f"Removed {num_gaps} gaps larger than {gap_threshold} seconds")
    
    cumulative_shift = 0
    new_index = []
    
    for i, idx in enumerate(df_index):
        
        if i > 0 and gaps.iloc[i]:
            cumulative_shift += time_diff.iloc[i]

        new_index.append(idx - cumulative_shift)
    
    result_df = df.copy()
    result_df.index = new_index

    if verbose:
        plot_filtering(df, result_df, title="Gap Removal")
    
    return result_df


def _filter_by_rate_of_change(df: pd.DataFrame, window_size: int = 50, threshold: float = 2) -> pd.DataFrame:
    
    # Normalize data to [0,1] range
    df_copy = df.copy()
    df_normalized = (df_copy - df_copy.min()) / (df_copy.max() - df_copy.min())
    df_normalized = df_normalized.fillna(0)  # Handle constant columns
    print("Data normalized to [0,1] range for rate of change calculation")

    # Calculate global rate of change (mean of absolute differences) for comparison
    global_rate_of_change = df_normalized.diff().abs().mean()
    global_rate_of_change = global_rate_of_change.replace(0, 1.0)  # Avoid division by zero
    
    drop_indices = []
    
    for i in tqdm(range(len(df_normalized))):
        # Adjust window boundaries at edges
        start_idx = max(0, i - window_size//2)
        end_idx = min(len(df_normalized), i + window_size//2 + 1)
        
        window = df_normalized.iloc[start_idx:end_idx]
        
        # Calculate relative rate of change for the window
        window_rate_of_change = window.diff().abs().mean()
        relative_rate_of_change = window_rate_of_change / global_rate_of_change
        
        # Filter points where all variables have low relative rate of change
        if (relative_rate_of_change < threshold).all():
            drop_indices.append(df_copy.index[i])
    
    # Return the original filtered data (not normalized) with identified indices removed
    filtered_data = df_copy.drop(index=drop_indices)
    
    print(f"Rate of change filter: Removed {len(drop_indices)} points out of {len(df_copy)} total points.")
    return filtered_data

def _filter_by_volatility(df: pd.DataFrame, window_size: int = 50, threshold: float = 0.3) -> pd.DataFrame:

    df_copy = df.copy()

    # Calculate global volatility (std of differences) for comparison
    global_volatility = df_copy.diff().std()
    global_volatility = global_volatility.replace(0, 1.0)  # Avoid division by zero
    
    print("global volatility:", global_volatility)
    
    drop_indices = []
    
    for i in tqdm(range(len(df_copy))):
        # Adjust window boundaries at edges
        start_idx = max(0, i - window_size//2)
        end_idx = min(len(df_copy), i + window_size//2 + 1)
        
        window = df_copy.iloc[start_idx:end_idx]
        
        # Calculate relative volatility for the window
        window_volatility = window.diff().std()
        relative_volatility = window_volatility / global_volatility
        
        # Filter points where all variables have low relative volatility
        if (relative_volatility < threshold).all():
            drop_indices.append(df_copy.index[i])
    
    filtered_data = df_copy.drop(index=drop_indices)
    
    print(f"Volatility filter: Removed {len(drop_indices)} points out of {len(df_copy)} total points.")
    return filtered_data

def _filter_by_knn(df: pd.DataFrame, window_size: int = 50, threshold: float = 0.05, k: int = None) -> pd.DataFrame:
    from sklearn.neighbors import NearestNeighbors
    # Set default k value if not provided
    if k is None:
        k = window_size // 5

    df_copy = df.copy()
    
    drop_indices = []
    
    for i in tqdm(range(len(df_copy))):
        # Adjust window boundaries at edges
        start_idx = max(0, i - window_size//2)
        end_idx = min(len(df_copy), i + window_size//2 + 1)
        
        window = df_copy.iloc[start_idx:end_idx]
        
        # Calculate KNN diversity score
        if len(window) < k + 1:
            continue  # Not enough data, keep the point
        
        X = window.values
        center_idx = i - start_idx  # Adjust center index relative to window
        
        # Fit KNN model
        neighbors = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
        neighbors.fit(X)
        
        # Get distances to k nearest neighbors for the center point
        distances, _ = neighbors.kneighbors(X[center_idx:center_idx+1])
        
        # Calculate diversity score (mean distance to neighbors)
        diversity_score = distances[0][1:].mean()
        
        # Filter points with low KNN diversity
        if diversity_score < threshold:
            drop_indices.append(df_copy.index[i])
    
    filtered_data = df_copy.drop(index=drop_indices)
    
    print(f"KNN filter: Removed {len(drop_indices)} points out of {len(df_copy)} total points.")
    return filtered_data



def _filter_by_clustering_and_sampling_training_samples(df_inputs: pd.DataFrame, df_targets: pd.DataFrame, threshold: int) -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    MAX_SAMPLES = 100000

    print(f"\n{'='*60}")
    print("Clustering and Sampling Training Samples")
    print(f"{'='*60}")
    print(f"Total samples before filtering: {len(df_inputs)}")
    print(f"Number of clusters (k): {threshold}")

    scaler = StandardScaler()
    inputs_scaled = scaler.fit_transform(df_inputs)

    kmeans = KMeans(n_clusters=threshold, random_state=0)
    cluster_labels = kmeans.fit_predict(inputs_scaled)

    cluster_sizes = np.bincount(cluster_labels)
    min_cluster_size = int(np.min(cluster_sizes))
    max_cluster_size = int(np.max(cluster_sizes))
    mean_cluster_size = int(np.mean(cluster_sizes))

    print(f"\nCluster size statistics:")
    print(f"  Min: {min_cluster_size}, Max: {max_cluster_size}, Mean: {mean_cluster_size}")
    print(f"\nCluster distribution before sampling:")
    for cid, size in enumerate(cluster_sizes):
        print(f"  Cluster {cid}: {size} samples")

    samples_per_cluster = min(mean_cluster_size * 2, MAX_SAMPLES // threshold)
    print(f"\nSamples per cluster (target): {samples_per_cluster}")
    
    # Sample from clusters
    sampled_indices = _sample_from_clusters(cluster_labels, samples_per_cluster=samples_per_cluster, verbose=False)

    # Calculate actual samples per cluster after sampling
    sampled_cluster_labels = cluster_labels[sampled_indices]
    sampled_cluster_sizes = np.bincount(sampled_cluster_labels, minlength=threshold)
    
    print(f"\nCluster distribution after sampling:")
    for cid, size in enumerate(sampled_cluster_sizes):
        print(f"  Cluster {cid}: {size} samples")
    
    print(f"\nTotal samples after filtering: {len(sampled_indices)}")
    print(f"Reduction: {len(df_inputs) - len(sampled_indices)} samples ({(1 - len(sampled_indices)/len(df_inputs))*100:.1f}%)")
    print(f"{'='*60}\n")
    
    # Filter both inputs and targets using the mask
    df_inputs_filtered = df_inputs.iloc[sampled_indices]
    df_targets_filtered = df_targets.iloc[sampled_indices]
    
    return df_inputs_filtered, df_targets_filtered


def _cluster_by_kmeans(windows, n_clusters, verbose=False):
    import time
    from tslearn.clustering import TimeSeriesKMeans

    if verbose:
        print("Starting K-means clustering...")
    start_time = time.time()
    
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", verbose=verbose, random_state=0, n_jobs=-1)
    cluster_labels = kmeans.fit_predict(windows)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    if verbose:
        print(f"K-means clustering completed in {elapsed_time:.2f} seconds")

    return cluster_labels

def _sample_from_clusters(cluster_labels: np.ndarray, samples_per_cluster: int = 100, verbose: bool = False) -> list:
    import random
    
    unique_clusters = np.unique(cluster_labels)
    
    # If samples_per_cluster not specified, use size of smallest cluster
    if samples_per_cluster is None:
        cluster_sizes = [np.sum(cluster_labels == cluster) for cluster in unique_clusters]
        samples_per_cluster = min(cluster_sizes)
    
    if verbose:
        print(f"Total samples: {len(cluster_labels)}")
        print(f"Total clusters: {len(unique_clusters)}")
        print(f"Samples per cluster: {samples_per_cluster}")
    
    sampled_indices = []
    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) >= samples_per_cluster:
            sampled_indices.extend(random.sample(cluster_indices.tolist(), samples_per_cluster))
        else:
            sampled_indices.extend(cluster_indices.tolist())
    
    if verbose:
        print(f"Total samples after filtering: {len(sampled_indices)}")
        print(f"Filtered out: {len(cluster_labels) - len(sampled_indices)} samples")
    
    sampled_indices.sort()
    return sampled_indices

def plot_filtering(original: pd.DataFrame, filtered: pd.DataFrame, title: str = "Data Filtering Comparison"):

    n_columns = len(original.columns)
    
    # Convert time from seconds to days for better readability
    original_time_days = original.index / 86400
    filtered_time_days = filtered.index / 86400
    
    # Create subplots - arrange in rows for better visibility
    fig, axes = plt.subplots(n_columns, 1, figsize=(12, 4 * n_columns), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    # Handle case where there's only one column (axes won't be a list)
    if n_columns == 1:
        axes = [axes]
    
    for i, column in enumerate(original.columns):
        axes[i].plot(original_time_days, original[column], label=f'Original {column}', alpha=0.5, color='blue')
        axes[i].plot(filtered_time_days, filtered[column], label=f'Filtered {column}', linewidth=2, color='red')
        axes[i].set_title(f'{column}')
        axes[i].set_ylabel('Values')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Only set xlabel on the bottom subplot
    axes[-1].set_xlabel('Time (days)')
    
    plt.tight_layout()
    plt.show()


def main():
    pass

if __name__ == "__main__":
    main()