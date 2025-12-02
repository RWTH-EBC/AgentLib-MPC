# Readme Balancing Agentlib-MPC
The balancing functionality includes posssibilities to filter two different types of data, corresponding to two different stages of the model trainers' retraining process.


## Balancing of Time Series Data
Time series data can be filtered in its pure form of a time series. In the model trainer, this is currently applied in the `_update_time_series_data` function with a very low threshold to filter out steady-state periods. Its also possible to set a ratio of latest, most current points that will not be filtered. Attention is required, as for a chosen window only the most central point is deleted. This can lead to incorrect time dependencies in the resulting time series.

### Usage

Time series filtering can be used via the `filter_time_series()` function:

```python
from agentlib_mpc.utils.balancing import filter_time_series, FilteringMethod

# Filter time series data
filtered_df = filter_time_series(
    time_series=your_dataframe,
    method=FilteringMethod.ROC,  # or VOL, KNN, CaS
    window_size=50,
    threshold=2.0,  # method-dependent
    verbose=True
)
```

**Parameters:**
- `time_series`: DataFrame with time index and multiple columns
- `method`: FilteringMethod enum (ROC, VOL, KNN, or CaS)
- `window_size`: Size of the moving window (default: 50)
- `threshold`: Filtering threshold - higher values = more aggressive filtering (method-dependent)
- `verbose`: If True, plots original vs filtered data for comparison

### Available filtering functions:

#### Rate of Change (ROC)
Filters time series data by removing points where the rate of change is below a threshold relative to the global rate of change. Data is normalized to [0,1] range before calculating changes. Points are removed when all variables show low relative rate of change within a moving window so one changing variable is sufficient to succeed the threshold.

#### Volatility (VOL)
Filters time series data based on local volatility (standard deviation of differences) compared to global volatility. Points are removed when all variables have low relative volatility within a moving window. One variable with high variability is sufficient to succeed the threshold

#### K-Nearest Neighbors (KNN)
Uses k-nearest neighbors to calculate diversity scores for each point within a moving window. Points with low diversity (mean distance to k nearest neighbors below threshold) are filtered out, removing redundant data in dense regions.



## Balancing of Training Samples

Final training samples (input-target pairs) can be filtered after being prepared for model training. This is useful for reducing dataset size while maintaining diversity and representativeness. In the model trainer this is implemented right after the creation of train-, test- and validation split and is only applied on the training split.

### Usage

Training sample filtering can be used via the `filter_training_samples()` function:

```python
from agentlib_mpc.utils.balancing import filter_training_samples, FilteringMethod

# Filter training samples
df_inputs_filtered, df_targets_filtered = filter_training_samples(
    df_inputs=your_input_dataframe,
    df_targets=your_target_dataframe,
    method=FilteringMethod.CaS,
    threshold=10,  # in case of CaS: threshold = number of clusters
    verbose=True
)
```

**Parameters:**
- `df_inputs`: DataFrame with input training data (time index, multiple columns)
- `df_targets`: DataFrame with target training data (time index, multiple columns)
- `method`: FilteringMethod enum (currently only CaS is supported for training samples)
- `threshold`: Number of clusters to create (specific usage for CaS method)
- `verbose`: If True, prints detailed clustering statistics

### Available filtering functions:

#### Clustering and Sampling (CaS)
Clusters the input samples using K-means clustering with StandardScaler normalization, then samples representative points from each cluster. The method automatically balances cluster sizes by sampling up to 2× the mean cluster size per cluster, ensuring diverse coverage while significantly reducing dataset size. Maximum total samples is capped at 100,000.

## Gap Removal
Since the model trainer performs a resampling of the time series, which fills up time gaps via interpolation, it is adviced to perform gap removal to minimize the creation of worthless training points. In the model trainer, the time series filtering includes a gap removal with a threshold of 2 hours.

Gap removal can be used via the `remove_gaps()` function:

```python
from agentlib_mpc.utils.balancing import remove_gaps

# Remove gaps from time series
df_no_gaps = remove_gaps(
    df=your_dataframe,
    gap_threshold=3600,  # in seconds
    verbose=True
)
```

**Parameters:**
- `df`: DataFrame with numeric time index in seconds
- `gap_threshold`: Maximum allowed gap in seconds (default: 3600 = 1 hour). Gaps larger than this will be removed by shifting subsequent data back in time
- `verbose`: If True, plots original vs filtered data for comparison

