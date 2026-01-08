import sys
import os
import csv
import math
import multiprocessing as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- PATH SETUP ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Custom Imports ---
from utils.metrics import get_stats_mean_sim
from model.pomdp import POMDP

# =============================================================================
# CONFIGURATION
# =============================================================================
OUT_DIR = Path('/kyb/agpd/wwang/sst-pomdp/outputs/')
PARAMS_CSV = OUT_DIR / "posterior_summary.csv"
OUTPUT_CSV = OUT_DIR / "sensitivity_res_1d.csv"

SUBJECT_ID = 'INV7C262RWK'
N_REPEAT = 10
BINS = 10
MAX_WORKERS = max(1, mp.cpu_count() - 2)
EPSILON = 1e-6

FIXED_PARAMS = {
    'cost_go_error': 1.0,
    'cost_go_missing': 1.0,
    'inv_temp': 50.0,
    'rate_stop_trial': 1/6
}

PARAM_RANGES = {
    "q_d_n": (0.0, 1.0),
    "q_d":   (0.5, 1.0),
    "q_s_n": (0.0, 1.0),
    "q_s":   (0.5, 1.0),
    "cost_stop_error": (0.0, 2.0),
    "cost_time":       (0.0, 0.003),
}

DYNAMIC_PARAMS_KEYS = ['q_d_n', 'q_d', 'q_s_n', 'q_s', 'cost_stop_error', 'cost_time']
METRIC_NAMES = ["mrt_gs", "ssrt"]

PARAM_DISPLAY_NAMES = {
    "q_d_n": r"$\chi'$",
    "q_d": r"$\chi$",
    "q_s_n": r"$\delta'$",
    "q_s": r"$\delta$",
    "cost_stop_error": r"$c_{\mathrm{se}}$",
    "cost_time": r"$c_t$",
}

METRIC_DISPLAY_NAMES = {
    "mrt_gs": "GSRT",
    "ssrt": "SSRT",
}

# =============================================================================
# DATA LOADING & SIMULATION
# =============================================================================

def get_subject_params(subject_id, csv_path, fixed_params):
    if not csv_path.exists(): raise FileNotFoundError(f"Missing: {csv_path}")
    df = pd.read_csv(csv_path)
    subject_df = df[df['subject_id'] == subject_id]
    if subject_df.empty: raise ValueError(f"Subject {subject_id} not found.")
    param_dict = subject_df.set_index('param')['mean'].to_dict()
    clean_params = {k: param_dict[k] for k in DYNAMIC_PARAMS_KEYS if k in param_dict}
    return {**clean_params, **fixed_params}

def safe_linspace(a, b, bins, eps=1e-3):
    lo, hi = min(a, b), max(a, b)
    return np.linspace(lo + eps, hi - eps, bins)

def simulate_and_extract_stat(param_dict, baseline_params, n_repeat, metric_names):
    stats_buffer = {m: [] for m in metric_names}
    try:
        test_params = baseline_params.copy()
        test_params.update(param_dict)
        rate_stop = test_params.pop('rate_stop_trial', 1/6)
        pomdp = POMDP(**test_params, rate_stop_trial=rate_stop)
        pomdp.value_iteration_tensor()

        for _ in range(n_repeat):
            out = pomdp.simu_task()
            df_out = pd.DataFrame(out, columns=["result", "rt", "ssd"])
            stats = get_stats_mean_sim(df_out)
            stat_dict = {"mrt_gs": stats[4], "ssrt": stats[7]}
            for m in metric_names: stats_buffer[m].append(stat_dict.get(m, np.nan))
    except Exception as e:
        for m in metric_names: stats_buffer[m] = [np.nan] * n_repeat

    rows = []
    key, val = list(param_dict.items())[0]
    for m in metric_names:
        arr = np.asarray(stats_buffer[m], dtype=float)
        rows.append({
            "param_1": key, "value_1": val, "metric": m,
            "mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)),
        })
    return rows

def run_sensitivity_analysis(baseline_params):
    value_grids = {k: safe_linspace(v[0], v[1], BINS, eps=EPSILON) for k, v in PARAM_RANGES.items()}
    header = ["param_1", "value_1", "metric", "mean", "std"]
    
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(simulate_and_extract_stat, {p: float(v)}, baseline_params, N_REPEAT, METRIC_NAMES)
                       for p, vals in value_grids.items() for v in vals]
            for i, fut in enumerate(as_completed(futures)):
                for row in fut.result(): writer.writerow(row)
    print(f"Results saved to {OUTPUT_CSV}")

# =============================================================================
# PLOTTING
# =============================================================================

def plot_sensitivity_merged(df, baseline_params, save_path=None, ncols=3):
    """Plots GSRT and SSRT in the same subplots with a global legend at the top."""
    # Define colors
    metric_colors = {
        "mrt_gs": "#1f77b4", # Blue
        "ssrt":   "#d62728"  # Red
    }
    
    params_in_data = list(df["param_1"].dropna().unique())
    param_order = [p for p in PARAM_RANGES.keys() if p in params_in_data]

    n_plots = len(param_order)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows + 0.5), squeeze=False)
    
    legend_handles = []
    legend_labels = []

    for idx, p in enumerate(param_order):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        
        for m in METRIC_NAMES:
            sub = df[(df["param_1"] == p) & (df["metric"] == m)].copy().sort_values("value_1")
            if sub.empty: continue
            
            x = sub["value_1"].to_numpy()
            y = sub["mean"].to_numpy()
            yerr = sub["std"].to_numpy()
            color = metric_colors.get(m, "black")
            label = METRIC_DISPLAY_NAMES.get(m, m)
            
            line, = ax.plot(x, y, color=color, linewidth=2, label=label)
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.15, linewidth=0)
            ax.errorbar(x, y, yerr=yerr, linestyle="none", capsize=2, color=color, alpha=0.5)
            
            # Collect handles for the global legend only once
            if idx == 0:
                legend_handles.append(line)
                legend_labels.append(label)

        # Plot Subject Baseline Vertical Line
        if p in baseline_params:
            v_line = ax.axvline(baseline_params[p], linestyle="--", color="gray", alpha=0.7, label="Baseline")
            if idx == 0:
                legend_handles.append(v_line)
                legend_labels.append("Baseline")

        # Axis Formatting
        ax.set_xlabel(PARAM_DISPLAY_NAMES.get(p, p), fontsize=14)
        if c == 0: ax.set_ylabel("Response Time (ms)", fontsize=14)
        if p == "cost_time":
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        
        ax.set_ylim(100, 750)

    # Remove empty plots
    for k in range(n_plots, nrows * ncols):
        axes.flat[k].axis("off")

    # Add Global Legend at the Top (3 items in one row)
    fig.legend(
        handles=legend_handles, 
        labels=legend_labels, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 0.98), 
        ncol=3, 
        fontsize=12,
        frameon=False
    )

    # Adjust layout and reserve space (top=0.93) for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved merged plot with top legend to: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    try:
        subject_params = get_subject_params(SUBJECT_ID, PARAMS_CSV, FIXED_PARAMS)
        
        if not os.path.exists(OUTPUT_CSV):
            run_sensitivity_analysis(subject_params)

        if os.path.exists(OUTPUT_CSV):
            df = pd.read_csv(OUTPUT_CSV)
            df.replace(0, np.nan, inplace=True)
            
            plot_sensitivity_merged(
                df, 
                baseline_params=subject_params,
                save_path=OUT_DIR / "sensitivity_merged_1d.png"
            )
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")