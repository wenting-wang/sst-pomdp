import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent)) 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# --- Configuration & Paths ---

BASE_DIR = Path('/kyb/agpd/wwang/data/')
OUT_DIR = Path('/kyb/agpd/wwang/sst-pomdp/outputs/')

# Inputs
RECOVERY_CSV = Path("/kyb/agpd/wwang/data/res_832861/recovery_details.csv")

# Output
OUTPUT_PLOT = OUT_DIR / "param_recovery_v2.png"

# Parameters Config
PARAMS = [
    "q_d_n", "q_d", "q_s_n", "q_s", 
    "cost_stop_error", "cost_time",
    "ndt",
    "inv_temp",
]

PARAM_DISPLAY_NAMES = {
    'q_d_n': r"$\chi'$",
    'q_d': r"$\chi$",
    'q_s_n': r"$\delta'$",
    'q_s': r"$\delta$",
    'cost_stop_error': r"$c_{\mathrm{se}}$",
    'cost_time': r"$c_t$",
    "ndt": r"$ndt$",
    "inv_temp": r"$\varphi$",
}

# Plot Settings
N_COLS = 4
MAX_POINTS = None  # Set to an int (e.g., 2000) to downsample scatter points for speed
SCATTER_COLOR = "black"
LINE_COLOR = "black"


# --- Core Functions ---

def load_data(csv_path):
    """Loads the recovery details CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Recovery file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded recovery data: {len(df)} records.")
    return df


def safe_corr(x, y, method=pearsonr):
    """
    Computes correlation safely, handling NaNs and insufficient data points.
    Returns: (correlation_coefficient, p_value, valid_mask)
    """
    mask = np.isfinite(x) & np.isfinite(y)
    
    # Need at least 3 points and variance in both x and y
    if mask.sum() >= 3 and np.unique(x[mask]).size > 1 and np.unique(y[mask]).size > 1:
        r, p = method(x[mask], y[mask])
        return float(r), float(p), mask
        
    return np.nan, np.nan, mask


def plot_recovery_grid(df, params, output_path):
    """
    Generates a grid of scatter plots comparing True vs. Recovered parameters.
    """
    n_params = len(params)
    n_rows = int(np.ceil(n_params / N_COLS))
    
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(4.5 * N_COLS, 4.2 * n_rows))
    axes = axes.flatten()

    print("\n--- Correlation Results ---")

    for i, param in enumerate(params):
        ax = axes[i]
        
        # Extract Data
        col_true = f"gt_{param}"
        col_rec = f"mu_{param}"
        
        if col_true not in df.columns or col_rec not in df.columns:
            print(f"Skipping {param}: Columns not found.")
            ax.set_visible(False)
            continue

        x = df[col_true].to_numpy(float)
        y = df[col_rec].to_numpy(float)

        # 1. Calculate Correlation
        r_p, p_p, mask = safe_corr(x, y, pearsonr)
        print(f"{param:<15} Pearson r={r_p:.3f} (p={p_p:.3g})")

        # 2. Filter Valid Data
        x = x[mask]
        y = y[mask]

        # 3. Downsample if requested (for cleaner SVG/PDFs or speed)
        if MAX_POINTS is not None and len(x) > MAX_POINTS:
            rng = np.random.default_rng(2025)
            idx = rng.choice(len(x), size=MAX_POINTS, replace=False)
            x = x[idx]
            y = y[idx]

        # 4. Plot Scatter
        ax.scatter(x, y, s=10, alpha=0.7, color=SCATTER_COLOR, edgecolors='none')

        # 5. Plot Identity Line (x=y)
        # Calculate range with padding
        if len(x) > 0:
            lo = min(x.min(), y.min())
            hi = max(x.max(), y.max())
            pad = 0.1 * (hi - lo + 1e-8)
            lo -= pad
            hi += pad
            
            ax.plot([lo, hi], [lo, hi], "--", color=LINE_COLOR, linewidth=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        # 6. Styling
        display_name = PARAM_DISPLAY_NAMES.get(param, param)
        ax.set_xlabel(f"True {display_name}", fontsize=14)
        ax.set_ylabel(f"Recovered {display_name}", fontsize=14)
        
        # Tufte-style axes (Left/Bottom only)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.tick_params(axis="both", colors="black")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {output_path}")


# --- Main Execution ---

def main():
    try:
        # 1. Load Data
        df = load_data(RECOVERY_CSV)
        
        # 2. Plot Recovery
        plot_recovery_grid(df, PARAMS, OUTPUT_PLOT)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()