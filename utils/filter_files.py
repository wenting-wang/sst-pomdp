import os
import shutil
import pandas as pd

def copy_filtered_files():
    # --- Configuration ---
    csv_path = "/kyb/agpd/wwang/sst-pomdp/outputs/stats_beh_full.csv"
    source_folder = "/kyb/agpd/wwang/data/abcd_sst_tlb01"
    destination_folder = "/kyb/agpd/wwang/data/sst_valid_baseline"
    
    # Column name containing the filenames in the CSV
    filename_column = "filename"

    # --- Setup ---
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_folder):
        try:
            os.makedirs(destination_folder)
            print(f"Created destination directory: {destination_folder}")
        except OSError as e:
            print(f"Error creating directory {destination_folder}: {e}")
            return

    # Load the CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Check if the column exists
    if filename_column not in df.columns:
        print(f"Error: Column '{filename_column}' not found in CSV.")
        print(f"Available columns: {list(df.columns)}")
        return

    # --- Copying Process ---
    success_count = 0
    error_count = 0
    
    print(f"Starting copy process for {len(df)} files...")

    for index, row in df.iterrows():
        # Get the filename from the CSV
        fname = str(row[filename_column]).strip()
        
        # Construct full file paths
        src_path = os.path.join(source_folder, fname)
        dst_path = os.path.join(destination_folder, fname)

        try:
            if os.path.exists(src_path):
                # Copy file (copy2 preserves metadata like timestamps)
                shutil.copy2(src_path, dst_path)
                success_count += 1
                # Optional: print every 100 files to show progress
                if success_count % 100 == 0:
                    print(f"Copied {success_count} files...")
            else:
                print(f"Warning: Source file not found: {src_path}")
                error_count += 1
        except Exception as e:
            print(f"Error copying {fname}: {e}")
            error_count += 1

    # --- Final Report ---
    print("-" * 30)
    print("Process Complete.")
    print(f"Successfully copied: {success_count}")
    print(f"Errors/Missing files: {error_count}")
    print(f"Files are located in: {destination_folder}")

if __name__ == "__main__":
    copy_filtered_files()