import os
import glob

def delete_csv_files(folder_path):
    """
    Deletes all CSV files in the specified folder.

    Args:
        folder_path (str): The path to the folder containing the CSV files.
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    for file_path in csv_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

if __name__ == '__main__':
    folder_to_clean = "Virtual_Data_Generation/data/virtual"  # Replace with the actual folder path
    delete_csv_files(folder_to_clean)
