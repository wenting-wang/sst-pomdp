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

# --- PATH SETUP (Robotic Fix) ---
# Ensure the script can find the 'model' package in the parent directory
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

# Simulation Settings
SUBJECT_ID = 'INV7C262RWK'
N_REPEAT = 10
BINS = 10
MAX_WORKERS = max(1, mp.cpu_count() - 2) # Leave a couple of cores free
EPSILON = 1e-6

# Parameters that are constant across all subjects
FIXED_PARAMS = {
    'cost_go_error': 1.0,
    'cost_go_missing': 1.0,
    'inv_temp': 50.0,
    'rate_stop_trial': 1/6
}

# Parameters to scan (Sensitivity Analysis Ranges)
PARAM_RANGES = {
    "q_d_n": (0.0, 1.0),
    "q_d":   (0.5, 1.0),
    "q_s_n": (0.0, 1.0),
    "q_s":   (0.5, 1.0),
    "cost_stop_error": (0.0, 2.0),
    "cost_time":       (0.0, 0.003),
}

# Parameters to extract from the CSV for the subject
DYNAMIC_PARAMS_KEYS = ['q_d_n', 'q_d', 'q_s_n', 'q_s', 'cost_stop_error', 'cost_time']

# Metrics to analyze
METRIC_NAMES = ["mrt_gs", "ssrt"]

# Display Configuration
PARAM_DISPLAY_NAMES = {
    "q_d_n": r"$\chi'$",
    "q_d": r"$\chi$",
    "q_s_n": r"$\delta'$",
    "q_s": r"$\delta$",
    "cost_go_error": r"$c_{\mathrm{ge}}$",
    "cost_go_missing": r"$c_{\mathrm{gm}}$",
    "cost_stop_error": r"$c_{\mathrm{se}}$",
    "cost_time": r"$c_t$",
    "inv_temp": r"$\varphi$",
}

METRIC_DISPLAY_NAMES = {
    "mrt_gs": "GSRT (ms)",
    "ssrt": "SSRT (ms)",
}

# =============================================================================
# DATA LOADING
# =============================================================================

def get_subject_params(subject_id, csv_path, fixed_params):
    """
    Loads baseline parameters for a specific subject from CSV.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    subject_df = df[df['subject_id'] == subject_id]
    
    if subject_df.empty:
        raise ValueError(f"Subject ID {subject_id} not found in {csv_path}")

    # Convert to dict {param_name: mean_value}
    param_dict = subject_df.set_index('param')['mean'].to_dict()
    
    # Filter only relevant dynamic keys
    clean_params = {k: param_dict[k] for k in DYNAMIC_PARAMS_KEYS if k in param_dict}
    
    # Merge with fixed constants
    return {**clean_params, **fixed_params}

# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

def safe_linspace(a, b, bins, eps=1e-3):
    lo, hi = min(a, b), max(a, b)
    return np.linspace(lo + eps, hi - eps, bins)

def simulate_and_extract_stat(param_dict, baseline_params, n_repeat, metric_names):
    """
    Runs simulation for a specific test parameter, overriding the baseline.
    """
    stats_buffer = {m: [] for m in metric_names}

    try:
        # Create test params: Start with baseline, overwrite the specific scanned param
        test_params = baseline_params.copy()
        test_params.update(param_dict)
        
        # Pop 'rate_stop_trial' as it is a separate arg in __init__
        rate_stop = test_params.pop('rate_stop_trial', 1/6)

        # Initialize POMDP
        pomdp = POMDP(**test_params, rate_stop_trial=rate_stop)
        pomdp.value_iteration_tensor()

        # Run trials
        for _ in range(n_repeat):
            out = pomdp.simu_task()
            df_out = pd.DataFrame(out, columns=["result", "rt", "ssd"])
            stats = get_stats_mean_sim(df_out)

            # Map stats by index (4: Mean Go RT, 7: SSRT)
            stat_dict = {
                "mrt_gs": stats[4],
                "ssrt": stats[7],
            }
            
            for m in metric_names:
                stats_buffer[m].append(stat_dict.get(m, np.nan))

    except Exception as e:
        print(f"[ERROR] Simulation failed for {param_dict}: {e}")
        for m in metric_names:
            stats_buffer[m] = [np.nan] * n_repeat

    # Format Output
    rows = []
    key = list(param_dict.keys())[0]
    val = float(param_dict[key])
    
    for m in metric_names:
        arr = np.asarray(stats_buffer[m], dtype=float)
        rows.append({
            "param_1": key,
            "value_1": val,
            "metric": m,
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
        })
    return rows

def run_sensitivity_analysis(baseline_params):
    """
    Orchestrates the parallel simulation using the subject's baseline params.
    """
    # Generate scanning grids
    value_grids = {
        k: safe_linspace(v[0], v[1], BINS, eps=EPSILON)
        for k, v in PARAM_RANGES.items()
    }

    header = ["param_1", "value_1", "metric", "mean", "std"]
    
    # Overwrite check done in main, so we just write here
    print(f"Starting simulation with {MAX_WORKERS} workers...")
    
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            
            for p_name, values in value_grids.items():
                for v in values:
                    # Pass the subject's baseline params to the worker
                    futures.append(executor.submit(
                        simulate_and_extract_stat,
                        {p_name: float(v)},
                        baseline_params,
                        N_REPEAT,
                        METRIC_NAMES
                    ))

            for i, fut in enumerate(as_completed(futures)):
                try:
                    rows = fut.result()
                    for row in rows:
                        writer.writerow(row)
                    if (i + 1) % 10 == 0:
                        print(f"Progress: {i + 1}/{len(futures)} tasks...", end="\r")
                except Exception as e:
                    print(f"\n[ERROR] Worker failed: {e}")
    
    print(f"\nSimulation complete. Results saved to {OUTPUT_CSV}")

# =============================================================================
# PLOTTING
# =============================================================================

def plot_sensitivity_1d(df, metric_name, baseline_params, save_path=None, ncols=3):
    d = df[df["metric"] == metric_name].copy()
    if d.empty: return

    label_metric = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)
    params_in_data = list(d["param_1"].dropna().unique())
    
    # Sort order: scanned params first
    priority_list = list(PARAM_RANGES.keys())
    param_order = [p for p in priority_list if p in params_in_data]

    n_plots = len(param_order)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    
    cmap = plt.get_cmap("cividis")
    line_color = cmap(0.7)

    for idx, p in enumerate(param_order):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        
        # Plot Data
        sub = d[d["param_1"] == p].copy().sort_values("value_1")
        x, y, yerr = sub["value_1"].to_numpy(), sub["mean"].to_numpy(), sub["std"].to_numpy()
        
        ax.plot(x, y, color=line_color, linewidth=2)
        ax.fill_between(x, y - yerr, y + yerr, color=line_color, alpha=0.2, linewidth=0)
        ax.errorbar(x, y, yerr=yerr, linestyle="none", capsize=3, color=line_color)


        # Plot Baseline Vertical Line (Specific to Subject)
        if p in baseline_params:
            x0 = baseline_params[p]
            ax.axvline(x0, linestyle="--", linewidth=1.5, alpha=0.9, color="gray")

        # Formatting
        ax.set_xlabel(PARAM_DISPLAY_NAMES.get(p, p), fontsize=14)
        if c == 0: ax.set_ylabel(label_metric, fontsize=14)
        
        if p == "cost_time":
            ax.set_xticks(np.linspace(min(x), max(x), 4))
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

        ax.set_ylim(100, 700)

    # Clean up empty subplots
    for k in range(n_plots, nrows * ncols):
        axes.flat[k].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    try:
        # 1. Load Subject Parameters
        print(f"Loading parameters for Subject: {SUBJECT_ID}...")
        subject_params = get_subject_params(SUBJECT_ID, PARAMS_CSV, FIXED_PARAMS)
        
        # 2. Run Simulation (if output doesn't exist)
        if os.path.exists(OUTPUT_CSV):
            print(f"File '{OUTPUT_CSV}' exists. Skipping simulation.")
        else:
            run_sensitivity_analysis(subject_params)

        # 3. Plot Results
        if os.path.exists(OUTPUT_CSV):
            df = pd.read_csv(OUTPUT_CSV)
            df.replace(0, np.nan, inplace=True)

            for metric in METRIC_NAMES:
                plot_sensitivity_1d(
                    df, 
                    metric,
                    baseline_params=subject_params, # Pass subject params for vertical lines
                    save_path=OUT_DIR / f"sensitivity_{metric}_1d.png"
                )
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")