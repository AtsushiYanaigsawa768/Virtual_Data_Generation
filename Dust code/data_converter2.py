import os
import pandas as pd
from pathlib import Path

def adjust_timestamps():
    input_dir = Path("/root/Virtual_Data_Generation/data/converted")
    output_dir = Path("/root/Virtual_Data_Generation/data/converted2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to group file paths by operation (prefix before the first underscore)
    operation_files = {}
    for csv_file in input_dir.glob("*.csv"):
        # Get operation from file name prefix until first "_"
        operation = csv_file.stem.split("_")[0]
        operation_files.setdefault(operation, []).append(csv_file)
    
    # Process each operation group
    for operation, files in operation_files.items():
        # Get the max last timestamp among all files for this operation
        max_timestamp = -1
        file_last_timestamps = {}
        for file in files:
            df = pd.read_csv(file)
            # Get the last row timestamp
            last_ts = df.iloc[-1]["timestamp"]
            file_last_timestamps[file] = last_ts
            if last_ts > max_timestamp:
                max_timestamp = last_ts
        
        # Adjust each file's timestamps using multiplier = max_timestamp / (file's last timestamp)
        for file in files:
            df = pd.read_csv(file)
            file_last_ts = file_last_timestamps[file]
            if file_last_ts == 0:
                print(f"Skipping file {file} due to last timestamp 0.")
                continue
            multiplier = max_timestamp / file_last_ts
            df["timestamp"] = df["timestamp"] * multiplier
            # Save to output directory with the same filename
            output_file = output_dir / file.name
            df.to_csv(output_file, index=False)

if __name__ == "__main__":
    adjust_timestamps()
