import os
import csv
import math
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Custom Imports (Ensure these files are in your python path) ---
from utils.metrics import get_stats_mean_sim
from model.pomdp import POMDP

# =============================================================================
# CONFIGURATION & PARAMETERS
# =============================================================================
OUT_DIR = Path('/kyb/agpd/wwang/sst-pomdp/outputs/')
OUTPUT_CSV = OUT_DIR / "sensitivity_res_1d.csv"
N_REPEAT = 10
BINS = 10
MAX_WORKERS = mp.cpu_count()
EPSILON = 1e-6

# Baseline parameters (N=5918 baseline)
DEFAULT_PARAMS = {
    'q_d_n': 0.611,
    'q_d': 0.683,
    'q_s_n': 0.062,
    'q_s': 0.901,
    'cost_go_error': 1.0,
    'cost_go_missing': 1.0,
    'cost_stop_error': 1.825,
    'cost_time': 0.0017,
    'inv_temp': 50.0
}

# Parameters to scan
PARAM_RANGES = {
    "q_d_n": (0.0, 1.0),
    "q_d":   (0.5, 1.0),
    "q_s_n": (0.0, 1.0),
    "q_s":   (0.5, 1.0),
    "cost_stop_error": (0.0, 2.0),
    "cost_time":       (0.0, 0.003),
}

# Metrics to analyze
METRIC_NAMES = ["mrt_gs", "ssrt"]

# Plotting Configuration
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
# SIMULATION HELPER FUNCTIONS
# =============================================================================

def safe_linspace(a, b, bins, eps=1e-3):
    """Generates a linear space avoiding the exact boundaries by epsilon."""
    lo = min(a, b)
    hi = max(a, b)
    return np.linspace(lo + eps, hi - eps, bins)


def simulate_and_extract_stat(param_dict, default_params, n_repeat, metric_names):
    """
    Runs the POMDP simulation for a specific parameter set and calculates statistics.
    """
    stats_buffer = {m: [] for m in metric_names}

    try:
        # Merge default params with the specific parameter being tested
        test_params = default_params.copy()
        test_params.update(param_dict)

        # Initialize and solve POMDP
        pomdp = POMDP(**test_params)
        pomdp.value_iteration_tensor()

        # Run simulations
        for _ in range(n_repeat):
            out = pomdp.simu_task()
            df_out = pd.DataFrame(out, columns=["result", "rt", "ssd"])
            stats = get_stats_mean_sim(df_out)

            # Map stats by index (Assuming indices based on perf_within logic)
            # Index 4: Mean RT on Go Signals, Index 7: SSRT
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

    # Prepare rows for CSV output
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


def run_sensitivity_analysis():
    """
    Orchestrates the parallel simulation of parameters and writes results to CSV.
    """
    # 1. Generate scanning grids
    value_grids = {
        k: safe_linspace(v[0], v[1], BINS, eps=EPSILON)
        for k, v in PARAM_RANGES.items()
    }

    # 2. Setup CSV file
    header = ["param_1", "value_1", "metric", "mean", "std"]
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    print(f"Starting simulation with {MAX_WORKERS} workers...")
    
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            
            # Submit tasks
            for p_name, values in value_grids.items():
                for v in values:
                    futures.append(executor.submit(
                        simulate_and_extract_stat,
                        {p_name: float(v)},
                        DEFAULT_PARAMS,
                        N_REPEAT,
                        METRIC_NAMES
                    ))

            # Process results as they complete
            for i, fut in enumerate(as_completed(futures)):
                try:
                    rows = fut.result()
                    for row in rows:
                        writer.writerow(row)
                    
                    if (i + 1) % 10 == 0:
                        print(f"Progress: {i + 1}/{len(futures)} tasks completed.", end="\r")
                        
                except Exception as e:
                    print(f"\n[ERROR] Future processing failed: {e}")
    
    print(f"\nSimulation complete. Results saved to {OUTPUT_CSV}")


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_sensitivity_1d(
    df,
    metric_name,
    save_path=None,
    param_order=None,
    show_baseline=True,
    baseline_params=DEFAULT_PARAMS,
    ncols=3,
):
    """
    Generates 1D sensitivity plots for a specific metric across different parameters.
    """
    # Filter data for the specific metric
    d = df[df["metric"] == metric_name].copy()
    if d.empty:
        print(f"No data found for metric: {metric_name}")
        return

    label_metric = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)

    # Determine parameter order
    params_in_data = list(d["param_1"].dropna().unique())
    
    if param_order is None:
        # Default order: prioritize known params, then append others
        priority_list = ["q_d_n", "q_d", "q_s_n", "q_s", "cost_stop_error", "cost_time"]
        param_order = [p for p in priority_list if p in params_in_data]
        param_order += [p for p in params_in_data if p not in priority_list]
    else:
        # Filter provided order by what exists in data
        param_order = [p for p in param_order if p in params_in_data]

    n_plots = len(param_order)
    if n_plots == 0:
        print("No parameters to plot.")
        return

    # Setup Plot
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    
    # Color settings
    cmap = plt.get_cmap("cividis")
    line_color = cmap(0.7)

    for idx, p in enumerate(param_order):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        label_p = PARAM_DISPLAY_NAMES.get(p, p)

        # Prepare sub-data
        sub = d[d["param_1"] == p].copy().sort_values("value_1")
        x = sub["value_1"].to_numpy()
        y = sub["mean"].to_numpy()
        yerr = sub["std"].to_numpy()

        # Plot Line + Error Bars + Shaded Region
        ax.plot(x, y, color=line_color, linewidth=2)
        ax.fill_between(x, y - yerr, y + yerr, color=line_color, alpha=0.2, linewidth=0)
        ax.errorbar(x, y, yerr=yerr, linestyle="none", capsize=3, color=line_color)

        # Plot Baseline Vertical Line
        if show_baseline and (p in baseline_params):
            x0 = baseline_params[p]
            ax.axvline(x0, linestyle="--", linewidth=1.5, alpha=0.9, color="gray")

        # Formatting
        ax.set_xlabel(label_p, fontsize=14)
        if c == 0:
            ax.set_ylabel(label_metric, fontsize=14)

        # Special handling for 'cost_time' (small values)
        if p == "cost_time":
            xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
            x_range = xmax - xmin
            ax.set_xlim(xmin - 0.05 * x_range, xmax + 0.05 * x_range)
            
            # Format ticks for small numbers
            xticks = np.linspace(xmin, xmax, 5)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{t:.3g}" for t in xticks])

        # Y-Axis Limits (Hardcoded based on requirements)
        ax.set_ylim(100, 700)

    # Hide empty subplots
    for k in range(n_plots, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].axis("off")

    plt.tight_layout()
    if save_path:
        print(f"Saving plot to {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # 1. Run the Simulation only if the output file does not exist
    if os.path.exists(OUTPUT_CSV):
        print(f"File '{OUTPUT_CSV}' already exists. Skipping simulation step.")
    else:
        print(f"File '{OUTPUT_CSV}' not found. Starting simulation...")
        run_sensitivity_analysis()

    # 2. Load Data and Plot
    if os.path.exists(OUTPUT_CSV):
        try:
            df = pd.read_csv(OUTPUT_CSV)
            
            # Replace 0 placeholders with NaN for cleaner plotting
            df.replace(0, np.nan, inplace=True)

            # 3. Generate Plots for each metric
            for metric in METRIC_NAMES:
                plot_sensitivity_1d(
                    df, 
                    metric,
                    # Using Path syntax as requested
                    save_path=OUT_DIR / f"sensitivity_{metric}_1d.png",
                    show_baseline=True,
                    ncols=3
                )
        except Exception as e:
            print(f"[ERROR] An error occurred while processing the CSV: {e}")
    else:
        print(f"[ERROR] {OUTPUT_CSV} could not be found. Cannot generate plots.")
