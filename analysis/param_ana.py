import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent)) 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration & Paths ---

OUT_DIR = Path('/kyb/agpd/wwang/sst-pomdp/outputs/')
PARAMS_CSV = OUT_DIR / "posterior_summary.csv"
STATS_CSV = OUT_DIR / "stats_beh_full.csv"
OUTPUT_PLOT = OUT_DIR / "parameters_v2.png"

# Updated params to match your requested layout: 
# Row 1: chi', chi, c_se
# Row 2: delta', delta, c_t
PLOT_PARAMS = [
    'q_d_n', 'q_s_n', 'cost_stop_error',
    'q_d', 'q_s', 'cost_time'
]

PARAM_DISPLAY_NAMES = {
    'q_d_n': r"$\chi'$ (Go Null)",
    'q_d': r"$\chi$ (Go Precision)",
    'q_s_n': r"$\delta'$ (Stop Null)",
    'q_s': r"$\delta$ (Stop Precision)",
    'cost_stop_error': r"$c_{\mathrm{se}}$ (Stop Error Cost)",
    'cost_time': r"$c_t$ (Time Cost)",
}
EDGE_COLOR = "#ffffff"

FILL_COLOR = "#534F57"
COLOR_MEAN = "#B03A2E"   # Deep Carmine (Red)
COLOR_MEDIAN = "#21618C" # Steel Blue (Blue)
COLOR_MODE = "#B7950B"   # Dark Ochre (Gold)

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
    """Generates a 2x3 grid with stats (Mean, Median, Mode) and fixed axes."""
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    axes = axes.flatten()

    for i, param in enumerate(params):
        ax = axes[i]
        if param not in df.columns:
            ax.axis('off')
            continue
            
        data = df[param].dropna()
        
        # Calculate stats
        mean_val = data.mean()
        median_val = data.median()
        mode_val = data.round(3).mode()[0]  # Round to 3 to find a meaningful peak

        # 1. Histogram + KDE
        sns.histplot(data, bins=BINS, color=FILL_COLOR, edgecolor=EDGE_COLOR, 
                     alpha=0.4, stat="density", ax=ax)
        # sns.kdeplot(data, color=FILL_COLOR, linewidth=1, ax=ax)

        # 2. Add Vertical Lines for Stats
        ax.axvline(mean_val, color=COLOR_MEAN, linestyle='--', label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color=COLOR_MEDIAN, linestyle='-', label=f'Median: {median_val:.3f}')
        ax.axvline(mode_val, color=COLOR_MODE, linestyle='-.', label=f'Mode: {mode_val:.3f}')

        # 3. Specific Axis Constraints
        # Set 0-1 range for the chi and delta parameters
        if param in ['q_d_n', 'q_d', 'q_s_n', 'q_s']:
            ax.set_xlim(0, 1)
        if param == 'cost_time':
            ax.set_xlim(0, 0.003)
        

        # 4. Styling
        ax.set_title(PARAM_DISPLAY_NAMES.get(param, param), fontsize=14, fontweight='regular')
        ax.set_xlabel('')
        ax.set_ylabel("Density" if i % ncols == 0 else "")
        ax.legend(fontsize=10, frameon=False)
        
        sns.despine()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")

# --- Main Execution ---

def main():
    try:
        # 1. Load Data
        df = load_and_merge_data(PARAMS_CSV, STATS_CSV)
        
        # 2. Text Summary (All Params)
        print_summary_stats(df, PLOT_PARAMS)
        
        # 3. Plotting (Selected Params)
        plot_parameter_distributions(df, PLOT_PARAMS, OUTPUT_PLOT)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

# $\chi' = 0.548 / 0.565 / 0.611$,
# $\chi = 0.728 / 0.722 / 0.683$,
# $\delta' = 0.059 / 0.058 / 0.062$,
# $\delta = 0.855 / 0.866 / 0.901$,
# $c_{\mathrm{se}} = 1.316 / 1.366 / 1.825$,
# $c_t = 8\times10^{-4} / 6\times10^{-4} / 1.7\times10^{-4}$.