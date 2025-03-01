import csv
import glob
import os
import re

def process_csv(filepath):
    # Extract operation (the number after the underscore)
    basename = os.path.basename(filepath)
    m = re.search(r'_([\d]+)\.csv$', basename)
    if not m:
        print(f"Skipping {filepath}: cannot extract operation value.")
        return
    operation = m.group(1)

    # Read the original CSV content
    with open(filepath, "r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        rows = list(reader)

    if not rows:
        print(f"{filepath} is empty.")
        return

    # Assume first row is a header and add an "operation" column
    header = rows[0] + ["operation"]
    data_rows = [row + [operation] for row in rows[1:]]

    # Write back the updated data (overwrites the original file)
    with open(filepath, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(data_rows)
    print(f"Processed {filepath}")

def main():
    # Directory with the csv files
    directory = "/root/Virtual_Data_Generation/data/test/"
    # Process all CSV files in the directory
    for filepath in glob.glob(os.path.join(directory, "*.csv")):
        process_csv(filepath)

if __name__ == "__main__":
    main()