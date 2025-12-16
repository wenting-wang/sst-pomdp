import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# --- Configuration & Paths ---

BASE_DIR = Path('/kyb/agpd/wwang/data/')
OUT_DIR = Path('/kyb/agpd/wwang/sst-pomdp/outputs/')

# Input Files
PPC_METRICS_CSV = OUT_DIR / "ppc_metrics.csv" # Output from ppc.py
SUBJECT_FILTER_CSV = OUT_DIR / "stats_beh_full.csv"

# Output File
OUTPUT_IMAGE = OUT_DIR / "ppc_popu.png" # population

# Plotting Configuration
PLOT_COLS = [
    "dis_perc_gs", 
    "dis_ws_rt_gs", 
    "dis_ws_rt_se",
    "dis_perc_ss",
    "dis_ks_rt_gs",
    "dis_ks_rt_se"
]

PLOT_TITLES = {
    "dis_perc_gs": r"Abs. Diff. $|P(\text{GS})_{\text{obs}} - P(\text{GS})_{\text{sim}}|$",
    "dis_perc_ss": r"Abs. Diff. $|P(\text{SS})_{\text{obs}} - P(\text{SS})_{\text{sim}}|$",
    "dis_ws_rt_gs": "Wasserstein Distance (GS RT)", 
    "dis_ws_rt_se": "Wasserstein Distance (SE RT)",
    "dis_ks_rt_gs": "K-S Distance (GS RT)",
    "dis_ks_rt_se": "K-S Distance (SE RT)",
}


# --- Core Functions ---

def load_and_filter_data(ppc_path, filter_path):
    """
    Loads PPC metrics and filters for subjects present in the validated dataset (baseline only).
    """
    if not ppc_path.exists():
        raise FileNotFoundError(f"PPC file not found: {ppc_path}")
    if not filter_path.exists():
        raise FileNotFoundError(f"Filter file not found: {filter_path}")

    print("Loading data...")
    df_ppc = pd.read_csv(ppc_path)
    df_filter = pd.read_csv(filter_path, index_col=False)

    # 1. Select only baseline subjects from the filter file
    # Ensure columns match your filter file (stats_all_cbcl_demo.csv)
    baseline_subjects = df_filter[df_filter['year'] == 'baseline']['subject_id'].unique()
    
    # 2. Filter PPC data to include only these subjects
    df_clean = df_ppc[df_ppc['subject_id'].isin(baseline_subjects)].copy()
    
    print(f"Loaded {len(df_ppc)} PPC records.")
    print(f"Filtered down to {len(df_clean)} valid baseline subjects.")
    
    return df_clean


def plot_ppc_histograms(df, output_path):
    """
    Generates and saves a 2x3 grid of histograms for PPC metrics.
    """
    print("Generating plots...")
    
    # Setup Figure: 2 rows, 3 columns
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))

    for ax, col in zip(axes.flatten(), PLOT_COLS):
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in dataframe. Skipping.")
            continue

        # Plot Histogram
        data = df[col].dropna()
        ax.hist(
            data,
            bins=30,
            color='white',
            edgecolor='black',
            linewidth=0.8
        )

        # Styling
        ax.set_title(PLOT_TITLES.get(col, col), fontsize=12, pad=10)
        ax.grid(False)
        ax.set_facecolor('white')
        
        # Axis cleanup (Tufte style-ish)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

        # Ticks and Labels
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_ylabel("Frequency", fontsize=10)

    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    # Save
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")


# --- Main Execution ---

def main():
    try:
        # 1. Load Data
        df = load_and_filter_data(PPC_METRICS_CSV, SUBJECT_FILTER_CSV)
        
        # 2. Plot
        plot_ppc_histograms(df, OUTPUT_IMAGE)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()