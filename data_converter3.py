import os
import pandas as pd
from pathlib import Path

input_dir = Path("/root/Virtual_Data_Generation/data/converted2")
output_dir = Path("/root/Virtual_Data_Generation/data/converted6")

# Delete all files in output folder
if output_dir.exists():
    for f in output_dir.glob("*"):
        f.unlink()
else:
    output_dir.mkdir(parents=True, exist_ok=True)

# Process each CSV file from input directory.
for filepath in input_dir.glob("*.csv"):
    df = pd.read_csv(filepath)
    # If there are 301 or more rows, randomly sample down to 300 rows.
    if len(df) >= 301:
        output_file = output_dir / filepath.name
        if "operation" in df.columns:
            df = df.drop(columns=["operation"])
        df = df.drop(columns=["timestamp"])
        df.to_csv(output_file, index=False)

