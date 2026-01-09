import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent)) 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration & Paths ---

BASE_DIR = Path('/kyb/agpd/wwang/data/')
OUT_DIR = Path('/kyb/agpd/wwang/sst-pomdp/outputs/')

# Inputs
PARAMS_CSV = OUT_DIR / "posterior_summary.csv"
STATS_CSV = OUT_DIR / "stats_beh_full.csv"

# Output
OUTPUT_PLOT = OUT_DIR / "parameters.png"

# Analysis Config
ALL_PARAMS = [
    'cost_go_error', 'cost_go_missing', 'cost_stop_error', 'cost_time',
    'inv_temp', 'q_d', 'q_d_n', 'q_s', 'q_s_n'
]

PLOT_PARAMS = [
    'q_d_n', 'q_d', 'q_s_n', 'q_s', 
    'cost_stop_error', 'cost_time'
]

PARAM_DISPLAY_NAMES = {
    'q_d_n': r"$\chi'$",
    'q_d': r"$\chi$",
    'q_s_n': r"$\sigma'$",
    'q_s': r"$\sigma$",
    'cost_stop_error': r"$c_{\mathrm{se}}$",
    'cost_time': r"$c_t$",
}

# Visualization Settings
FILL_COLOR = "#413D7B"
EDGE_COLOR = "#ffffff"
BINS = 30


# --- Core Functions ---

def load_and_merge_data(params_path, stats_path):
    """
    Loads parameter summaries (Wide format) and merges with demographic/stats data.
    Filters for valid baseline subjects.
    """
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    # 1. Load Parameters
    df_param = pd.read_csv(params_path)
    if 'subject_year' in df_param.columns:
        df_param = df_param.rename(columns={"subject_year": "year"})
    
    # Pivot to Wide Format (One row per subject)
    # Ensure we don't lose subjects if pivot index varies
    df_pivot = df_param.pivot(index=['subject_id', 'year'], columns='param', values='mean').reset_index()
    df_pivot.columns.name = None

    # 2. Load Filter/Stats Data
    df_stats = pd.read_csv(stats_path)
    
    # 3. Merge (Inner Join)
    df_merged = pd.merge(df_stats, df_pivot, on=['subject_id', 'year'], how='inner')
    
    # 4. Filter Baseline
    df_final = df_merged[df_merged['year'] == 'baseline'].dropna().copy()
    
    print(f"Data Loaded. Subjects: {len(df_final)}")
    return df_final


def print_summary_stats(df, params):
    """Prints Min, Max, Mean, Median, and Mode for the specified parameters."""
    print("\n--- Parameter Ranges (Min/Max) ---")
    summary = []
    for p in params:
        if p in df.columns:
            lo = df[p].min()
            hi = df[p].max()
            summary.append((p, lo, hi))
    
    summary_df = pd.DataFrame(summary, columns=["param", "min", "max"])
    print(summary_df.to_string(index=False))

    print("\n--- Central Tendency (Mean/Median) ---")
    # Only calculate for params present in dataframe
    valid_params = [p for p in params if p in df.columns]
    stats_df = pd.DataFrame({
        "Mean": df[valid_params].mean(),
        "Median": df[valid_params].median(),
    })
    print(stats_df.to_string())

    print("\n--- Mode (Rounded to 3 decimals) ---")
    # Mode can return multiple rows if ties exist; usually we just want the first
    modes = df[valid_params].round(3).mode()
    print(modes.head(1).to_string(index=False))


def plot_parameter_distributions(df, params, output_path):
    """Generates and saves a grid of Histograms + KDE plots."""
    n = len(params)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.5 * nrows))
    axes = axes.flatten()

    for i, param in enumerate(params):
        ax = axes[i]
        
        if param not in df.columns:
            ax.set_visible(False)
            continue
            
        data = df[param].dropna()
        
        # 1. Histogram
        ax.hist(
            data, bins=BINS,
            color=FILL_COLOR, edgecolor=EDGE_COLOR,
            alpha=0.7, linewidth=0.8, density=True
        )
        
        # 2. KDE Line
        sns.kdeplot(
            data, ax=ax,
            color=FILL_COLOR, linewidth=1.2,
            fill=False, alpha=1
        )

        # 3. Styling
        label = PARAM_DISPLAY_NAMES.get(param, param)
        ax.set_xlabel('')
        ax.set_title(label, fontsize=14)
        
        if i % ncols == 0:
            ax.set_ylabel("Posterior mean density", fontsize=12)
        else:
            ax.set_ylabel("")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', direction='out', length=4, width=1, top=False, right=False)
        
        # Special formatting for cost_time (ticks starting from 0)
        if param == 'cost_time':
            ax.set_xticks(ax.get_xticks()[::2])
            ax.set_xlim(left=0)

    # Turn off empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {output_path}")


# --- Main Execution ---

def main():
    try:
        # 1. Load Data
        df = load_and_merge_data(PARAMS_CSV, STATS_CSV)
        
        # 2. Text Summary (All Params)
        print_summary_stats(df, ALL_PARAMS)
        
        # 3. Plotting (Selected Params)
        plot_parameter_distributions(df, PLOT_PARAMS, OUTPUT_PLOT)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()