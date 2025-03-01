import os
import pandas as pd
from pathlib import Path

def merge_operation_csv():
    input_dir = Path("/root/Virtual_Data_Generation/data/converted4")
    output_dir = Path("/root/Virtual_Data_Generation/data/converted5")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group CSV files by operation (prefix until first underscore in filename)
    operation_files = {}
    for csv_file in input_dir.glob("*.csv"):
        operation = csv_file.stem.split("_")[0]
        operation_files.setdefault(operation, []).append(csv_file)

    for operation, files in operation_files.items():
        all_dfs = []
        for file in files:
            df = pd.read_csv(file)
            # Remove the 'operation' column if it exists
            if "operation" in df.columns:
                df = df.drop(columns=["operation"])
            all_dfs.append(df)
        
        if not all_dfs:
            continue

        # Concatenate dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # If there are duplicate timestamps, drop duplicates leaving one
        combined_df = combined_df.drop_duplicates(subset="timestamp", keep="first")
        # Remove the 'timestamp' column
        combined_df = combined_df.drop(columns=["timestamp"])
        # Save the result as a CSV file with the operation name
        output_file = output_dir / f"{operation}.csv"
        combined_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    merge_operation_csv()
