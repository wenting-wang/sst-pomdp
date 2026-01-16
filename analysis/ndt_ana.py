import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent)) 

import pandas as pd
import matplotlib.pyplot as plt
import os

# Your file path
file_path = "/kyb/agpd/wwang/data/res_831633/posterior_summary.csv"

def analyze_ndt(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    # 1. Load the data
    df = pd.read_csv(path)
    
    # 2. Filter for 'ndt' rows in the 'index' column
    # Use .copy() to avoid SettingWithCopy warnings during conversion
    ndt_df = df[df['index'] == 'ndt'].copy()
    
    if ndt_df.empty:
        print("No 'ndt' entries found in the data.")
        return

    # 3. Ensure the 'mean' column is numeric
    ndt_df['mean'] = pd.to_numeric(ndt_df['mean'], errors='coerce')
    ndt_df = ndt_df.dropna(subset=['mean'])

    # 4. Get the range
    ndt_min = ndt_df['mean'].min()
    ndt_max = ndt_df['mean'].max()
    count = len(ndt_df)
    mean = ndt_df['mean'].mean()

    print(f"--- Analysis Results ---")
    print(f"Total ndt records found: {count}")
    print(f"Range of ndt (mean): {ndt_min:.4f} to {ndt_max:.4f}")
    print(f"Average ndt (mean): {mean:.4f}")

    # 5. Plot the histogram
    plt.figure(figsize=(10, 6))
    # Adjust bins based on the number of samples you have
    plt.hist(ndt_df['mean'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    
    plt.title('Distribution of Non-Decision Time (ndt) Means', fontsize=14)
    plt.xlabel('ndt Mean Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save and show the plot
    output_plot = "/kyb/agpd/wwang/sst-pomdp/outputs/ndt_histogram.png"
    plt.savefig(output_plot)
    print(f"Histogram saved as '{output_plot}'")
    plt.show()

if __name__ == "__main__":
    analyze_ndt(file_path)