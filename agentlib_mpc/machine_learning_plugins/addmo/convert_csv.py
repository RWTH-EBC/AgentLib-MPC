"""
Script to convert simulation_data.csv to a normalized format.

Converts:
- Time from seconds since start to datetime timestamps
- Removes the first line (causality info)
- Removes the third line (type info)
- Renames the first column to 'time'
"""

import pandas as pd
from datetime import datetime, timedelta


def convert_simulation_csv(input_file, output_file, target: str, output_type: str, exclude_vars: list[str] = None):
    """
    Convert simulation CSV to normalized format.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        target: Name of target variable to predict
        output_type: 'absolute' or 'difference' for prediction type
        exclude_vars: List of variable names to exclude from the output (optional)
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
    df['Time'] = df['Time'].apply(lambda seconds: start_datetime + timedelta(seconds=seconds))

    # Create target for training based on output_type
    if output_type == "absolute":
        # Absolute: predict next value target(k+1)
        target_col = f"{target}_absolute"
        df[target_col] = df[target].shift(-1)
    elif output_type == "difference":
        # Difference: predict delta = target(k+1) - target(k)
        target_col = f"{target}_diff"
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
    
    # Save to output file
    df.to_csv(output_file, index=False)
    print(f"Converted CSV saved to: {output_file}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    import os
    
    input_file = "/Volumes/Samsung_T7/Git/AgentLib-MPC/examples/one_room_mpc/addmo_plugin/results/simulation_data.csv"
    target = "T"  # Column name in simulation CSV
    output_type = "absolute"  # 'absolute' for next value, 'difference' for delta
    
    # List of variable names to exclude from the output CSV
    exclude_vars = ['T_out']  # Add variable names here, e.g., ['var1', 'var2']
    
    # Get the directory and filename
    input_dir = os.path.dirname(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Create output file in the same directory with _addmo suffix
    output_file = os.path.join(input_dir, f"{base_name}_addmo.csv")
    
    convert_simulation_csv(input_file, output_file, target=target, output_type=output_type, exclude_vars=exclude_vars)
