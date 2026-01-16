import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import zipfile

# ================= CONFIGURATION =================
# Path to your data folder
DATA_DIR = '/kyb/agpd/wwang/data/sst_valid_base'

# Output filename for the plot
OUTPUT_PLOT_HIST = '/kyb/agpd/wwang/sst-pomdp/outputs/rt_histogram.png'
# =================================================

def plot_rt_distribution():
    print(f"Scanning {DATA_DIR} for zip files...")
    zip_files = glob.glob(os.path.join(DATA_DIR, '*.zip'))
    print(f"Found {len(zip_files)} zip files.")

    all_rts = []

    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                # Find the CSV inside the zip (ignore MACOSX)
                csv_files = [f for f in z.namelist() if f.endswith('sst.csv') and '__MACOSX' not in f]
                
                if not csv_files:
                    continue
                
                # Read the first valid CSV found
                with z.open(csv_files[0]) as f:
                    # Read only the reaction time column to save memory
                    df = pd.read_csv(f, usecols=['sst_primaryrt'])

            # Filter for valid reaction times (RT > 0)
            # We exclude 0 because it typically represents no response (successful stop or omission)
            valid_rts = df[df['sst_primaryrt'] > 0]['sst_primaryrt'].values
            
            if len(valid_rts) > 0:
                all_rts.append(valid_rts)

        except Exception as e:
            # Extract filename for error reporting
            base_name = os.path.basename(zip_path)
            print(f"Error processing {base_name}: {e}")

    # --- Aggregation ---
    if not all_rts:
        print("No valid reaction time data found.")
        return

    # Combine all arrays into one large numpy array
    print("Aggregating data...")
    combined_rts = np.concatenate(all_rts)
    print(f"Total valid reaction times collected: {len(combined_rts)}")
    print(f"Mean RT: {np.mean(combined_rts):.2f} ms")
    print(f"Median RT: {np.median(combined_rts):.2f} ms")

    # --- Plotting ---
    print("Generating histogram...")
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    # Bins: 100 bins for granularity, range limited to 0-1500ms to ignore extreme outliers if any
    plt.hist(combined_rts, bins=100, range=(0, 1500), color='steelblue', edgecolor='black', alpha=0.7)
    
    plt.title('Distribution of Reaction Times (sst_primaryrt) - All Subjects', fontsize=14)
    plt.xlabel('Reaction Time (ms)', fontsize=12)
    plt.ylabel('Frequency (Count)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add a vertical line for the mean
    plt.axvline(np.mean(combined_rts), color='red', linestyle='dashed', linewidth=1, label=f'Mean RT: {np.mean(combined_rts):.0f}ms')
    plt.legend()

    # Save plot
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PLOT_HIST), exist_ok=True)
    plt.savefig(OUTPUT_PLOT_HIST)
    print(f"\nSaved histogram to {OUTPUT_PLOT_HIST}")

if __name__ == "__main__":
    plot_rt_distribution()