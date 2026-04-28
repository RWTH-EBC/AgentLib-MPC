"""
Script to convert simulation_data.csv to a normalized format.

Converts:
- Time from seconds since start to datetime timestamps
- Removes the first line (causality info)
- Removes the third line (type info)
- Renames the first column to 'time'
"""

import os
import pandas as pd
from datetime import datetime, timedelta


def convert_simulation_csv(
    input_file: str,
    target: str,
    output_type: str,
    exclude_vars: list[str] = None,
    wanted_sampling_rate: int = 300,
    lags: dict[str, int] = None,
) -> str:
    """
    Convert simulation CSV to normalized format.
    
    Args:
        input_file: Path to input CSV file
        target: Name of target variable to predict
        output_type: 'absolute' or 'difference' for prediction type
        exclude_vars: List of variable names to exclude from the output (optional)
        wanted_sampling_rate: Desired sampling interval in seconds (default: 300)
        lags: Dictionary indicating the number of lags to add for each feature, e.g., {'T': 2}
        
    Returns:
        Path to the saved output CSV file
    """
    if exclude_vars is None:
        exclude_vars = []
    
    start_datetime = datetime(2024, 1, 1, 0, 0, 0)
    
    # Read the CSV file, skipping the first line (causality)
    df = pd.read_csv(input_file, skiprows=[0])
    
    # Remove the first row (type info, which is now the second row after skipping line 0)
    df = df.iloc[1:].reset_index(drop=True)
    
    # Rename the first column to 'Time'
    df.rename(columns={df.columns[0]: 'Time'}, inplace=True)
    
    # Convert time from seconds to datetime
    df['Time'] = df['Time'].astype(float)
    
    # Infer current sampling rate from the data (assuming consistent step)
    if len(df) > 1:
        current_sampling_rate = int(round(df['Time'].iloc[1] - df['Time'].iloc[0]))
        if current_sampling_rate == 0:
            current_sampling_rate = 1  # Fallback to prevent division by zero
    else:
        current_sampling_rate = 10  # Fallback
        
    df['Time'] = df['Time'].apply(lambda seconds: start_datetime + timedelta(seconds=seconds))

    # Convert numeric columns to float (excluding Time which is already converted)
    for col in df.columns:
        if col != 'Time':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Resample based on the provided sampling rates
    # Assumes input data is sampled uniformly at current_sampling_rate intervals.
    resample_step = max(1, int(wanted_sampling_rate / current_sampling_rate))
    if wanted_sampling_rate % current_sampling_rate != 0:
        print(f"Warning: wanted sampling rate ({wanted_sampling_rate}s) is not a multiple of current ({current_sampling_rate}s). Using step {resample_step}.")
        
    df = df.iloc[::resample_step].reset_index(drop=True)

    # Create target for training based on output_type
    if output_type == "absolute":
        # Absolute: predict next value target(k+1)
        target_col = f"{target}_absolute"
        df[target_col] = df[target].shift(-1)
    elif output_type == "difference":
        # Difference: predict delta = target(k+1) - target(k)
        target_col = f"{target}_difference"
        df[target_col] = df[target].shift(-1) - df[target]
    else:
        raise ValueError(f"output_type must be 'absolute' or 'difference', got '{output_type}'")
    
    # Drop last row with NaN from shifting
    df = df.dropna(subset=[target_col])
    
    # Handle NaN values in other columns
    df = df.ffill().bfill()
    
    # Remove excluded variables
    if exclude_vars:
        cols_to_drop = [col for col in exclude_vars if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"Excluded variables: {cols_to_drop}")
            
    # Add lags according to ADDMO naming convention.
    # The integer specified is the TOTAL number of values for that feature 
    # (e.g. 2 means lag0 and lag1), matching AgentLib's definition.
    target_input_col = [target]
    if lags:
        for feature, total_lags in lags.items():
            if feature in df.columns and total_lags > 1:
                # Get index of the feature to insert new lags right after it
                feature_idx = df.columns.get_loc(feature)
                df.rename(columns={feature: f"{feature}___lag0"}, inplace=True)
                for i in range(1, total_lags):
                    lagged_values = df[f"{feature}___lag0"].shift(i)
                    # Insert the new column right after the previous lag
                    df.insert(feature_idx + i, f"{feature}___lag{i}", lagged_values)
                
                # Update target input columns if target was lagged
                if feature == target:
                    target_input_col = [f"{target}___lag{i}" for i in range(total_lags)]
                    
        # Drop rows with NaN caused by shifting for lags
        df = df.dropna()
    
    # Reorder columns for agentlib-mpc compatibility:
    # 1. Time column first
    # 2. Non-target input features
    # 3. Recursive target feature (original target column)
    # 4. Target output column (_absolute or _difference)
    all_cols = df.columns.tolist()
    
    # Separate columns
    time_col = ['Time']
    target_output_col = [target_col]  # The created _absolute or _difference column
    # target_input_col was set above (either [target] or [target___lag0, ...])
    
    # All other columns (non-target inputs)
    other_cols = [col for col in all_cols 
                  if col not in time_col + target_output_col + target_input_col]
    
    # Reorder: Time -> other inputs -> recursive target -> target output
    ordered_cols = time_col + other_cols + target_input_col + target_output_col
    df = df[ordered_cols]
    
    print(f"Column order (agentlib-mpc compatible):")
    print(f"  - Time: {time_col}")
    print(f"  - Input features: {other_cols}")
    print(f"  - Recursive target (input): {target_input_col}")
    print(f"  - Target output: {target_output_col}")
    
    # Get the directory and filename to create an output file name
    input_dir = os.path.dirname(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(input_dir, f"{base_name}_addmo.csv")
    
    # Save to output file
    df.to_csv(output_file, index=False)
    print(f"Converted CSV saved to: {output_file}")
    print(f"Shape: {df.shape}")
    
    return output_file


if __name__ == "__main__":

    # Fill out/check these parameters before running the script
    input_file = r"C:\Users\sle-fmu\Desktop\Git\AgentLib-MPC\examples\one_room_mpc\addmo_plugin\results\simulation_data.csv"
    target = "T"  # Name of the variable to predict (must match a column in the input CSV, e.g., 'T')
    output_type = "difference"  # 'absolute' for next value, 'difference' for delta
    wanted_sampling_rate = 60  # dt
    exclude_vars = ['T_out','T_in','T_upper','T_slack']  # Add variables to be excluded here, e.g., ['var1', 'var2']
    lags = {'mDot': 2, 'load': 2, 'T': 5}  # Add features and total lags here, e.g., {'feature1': 3, 'feature2': 2}
    
    convert_simulation_csv(
        input_file=input_file,
        target=target,
        output_type=output_type,
        exclude_vars=exclude_vars,
        wanted_sampling_rate=wanted_sampling_rate,
        lags=lags,
    )
