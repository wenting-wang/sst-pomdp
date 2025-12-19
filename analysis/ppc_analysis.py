import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pathlib import Path

# Custom Modules
from model.pomdp import POMDP
from utils.preprocessing import preprocessing

# --- Configuration & Paths ---

BASE_DIR = Path('/kyb/agpd/wwang/data/')
OUT_DIR = Path('/kyb/agpd/wwang/sst-pomdp/outputs/')

# Inputs
RAW_DATA_DIR = BASE_DIR / "sst_valid_raw"
FILTER_CSV = OUT_DIR / "stats_beh_full.csv"
PPC_METRICS_CSV = OUT_DIR / "ppc_metrics.csv"  # Output from ppc.py
PARAMS_CSV = OUT_DIR / "posterior_summary.csv"

# Output
OUTPUT_IMAGE = OUT_DIR / "ppc.png" # representative subjects

# Plot Settings
COLOR_OBS = 'k'      # Black
COLOR_SIM = 'teal'   # Teal
ALPHA_OBS = 0.6
ALPHA_SIM = 0.4


# --- Core Functions ---

def select_representative_subjects(ppc_file, filter_file):
    """
    Identifies Best (5%), Median (50%), and Worst (95%) fitting subjects 
    based on a composite cost of standardized distance metrics.
    """
    # 1. Load Data
    if not ppc_file.exists() or not filter_file.exists():
        raise FileNotFoundError("PPC metrics or Filter file not found.")

    df_dist = pd.read_csv(ppc_file)
    df_filter = pd.read_csv(filter_file)
    
    # 2. Filter for baseline subjects only
    valid_ids = df_filter[df_filter['year'] == 'baseline']['subject_id'].values
    df_dist = df_dist[df_dist['subject_id'].isin(valid_ids)].copy()

    # 3. Calculate Composite Cost (Z-score sum)
    # Identify distance columns (starting with 'dis_')
    dis_cols = [col for col in df_dist.columns if col.startswith('dis_')]
    
    # Standardize (Z-score)
    df_z = df_dist.copy()
    for col in dis_cols:
        mean = df_dist[col].mean()
        std = df_dist[col].std()
        if std != 0:
            df_z[col] = (df_dist[col] - mean) / std
        else:
            df_z[col] = 0
            
    df_dist['composite_cost'] = df_z[dis_cols].sum(axis=1)

    # 4. Sort and Select
    df_sorted = df_dist.sort_values(by='composite_cost').reset_index(drop=True)
    n = len(df_sorted)
    
    selection = [
        (int(n * 0.05), "Good Fit (5th Percentile)"),
        (int(n * 0.50), "Moderate Fit (50th Percentile)"),
        (int(n * 0.95), "Poor Fit (95th Percentile)")
    ]
    
    subjects = []
    for idx, label in selection:
        row = df_sorted.iloc[idx]
        subjects.append({
            "sid": row['subject_id'],
            "label": label,
            "cost": row['composite_cost']
        })
        print(f"Selected {label}: {row['subject_id']} (Cost: {row['composite_cost']:.2f})")
        
    return subjects


def get_subject_params(params_file, subject_id, year='baseline'):
    """Loads the MAP parameters for a specific subject."""
    df = pd.read_csv(params_file)
    # Handle column naming variations if necessary
    year_col = 'subject_year' if 'subject_year' in df.columns else 'year'
    
    mask = (df["subject_id"] == subject_id) & (df[year_col] == year)
    if not mask.any():
        return None
        
    return df.loc[mask, ["param", "mean"]].set_index("param")["mean"].to_dict()


def run_simulation(params, n_repeat=1):
    """Runs the POMDP simulation n times and returns concatenated results."""
    pomdp = POMDP(**params)
    pomdp.value_iteration_tensor()

    sim_dfs = []
    for _ in range(n_repeat):
        out = pomdp.simu_task()
        df = pd.DataFrame(out, columns=['result', 'rt', 'ssd'])
        sim_dfs.append(df)
        
    return pd.concat(sim_dfs, ignore_index=True)


def plot_model_fit(subjects, raw_data_dir, params_file, output_path):
    """Generates the 5x3 grid of model fit visualizations."""
    
    # Setup Figure (5 Rows x 3 Columns)
    fig, axs = plt.subplots(5, 3, figsize=(9, 10), constrained_layout=True)
    sns.set_style("ticks")
    
    n_repeat = 1  # Number of simulations to run per subject

    for col_idx, subj_info in enumerate(subjects):
        sid = subj_info['sid']
        label = subj_info['label']
        
        # --- A. Prepare Data ---
        # 1. Get Params
        params = get_subject_params(params_file, sid)
        if not params:
            print(f"Skipping {sid}: Parameters not found.")
            continue
            
        # 2. Run Simulation
        df_sim = run_simulation(params, n_repeat=n_repeat)
        
        # 3. Load Observed Data
        # Construct filename (Adjust pattern if your files are named differently)
        # Assuming: NDAR_INV123_baseline_year_1_arm_1_sst.csv.zip
        file_path = raw_data_dir / f"NDAR_{sid}_baseline_year_1_arm_1_sst.csv.zip"
        
        if not file_path.exists():
            print(f"Skipping {sid}: Raw file not found at {file_path}")
            continue
            
        df_obs = preprocessing(str(file_path))

        # --- B. Plotting Rows ---
        
        # Row 0: Outcome Rates (Bar Chart)
        ax = axs[0, col_idx]
        outcomes = ['GS', 'GE', 'GM', 'SS', 'SE']
        rates_obs = [df_obs['result'].value_counts(normalize=True).get(o, 0) for o in outcomes]
        rates_sim = [df_sim['result'].value_counts(normalize=True).get(o, 0) for o in outcomes]
        
        x = np.arange(len(outcomes))
        width = 0.35
        ax.bar(x - width/2, rates_obs, width, label='Obs', color=COLOR_OBS, alpha=ALPHA_OBS)
        ax.bar(x + width/2, rates_sim, width, label='Sim', color=COLOR_SIM, alpha=ALPHA_SIM)
        ax.set_xticks(x)
        ax.set_xticklabels(outcomes)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Outcomes", fontsize=10)
        
        # Titles
        ax.set_title(f"{sid}\n{label}", fontsize=12, fontweight='regular')
        if col_idx == 0: ax.set_ylabel("Outcome Rates", fontsize=12)
        _clean_spines(ax)

        # Row 1: Go Success RT (Histogram)
        ax = axs[1, col_idx]
        bins = np.linspace(1, 41, 41)
        ax.hist(df_obs[df_obs['result']=='GS']['rt'], bins=bins, density=True, 
                alpha=ALPHA_OBS, label='Obs', color=COLOR_OBS)
        ax.hist(df_sim[df_sim['result']=='GS']['rt'], bins=bins, density=True, 
                alpha=ALPHA_SIM, label='Sim', color=COLOR_SIM)
        ax.set_xlim(1, 41)
        ax.set_xticks([1, 10, 20, 30, 40])
        ax.set_xlabel("Time Steps", fontsize=10)
        if col_idx == 0: ax.set_ylabel("GS RT Density", fontsize=12)
        _clean_spines(ax)

        # Row 2: Stop Error RT (Histogram)
        ax = axs[2, col_idx]
        ax.hist(df_obs[df_obs['result']=='SE']['rt'], bins=bins, density=True, 
                alpha=ALPHA_OBS, label='Obs', color=COLOR_OBS)
        ax.hist(df_sim[df_sim['result']=='SE']['rt'], bins=bins, density=True, 
                alpha=ALPHA_SIM, label='Sim', color=COLOR_SIM)
        ax.set_xlim(1, 41)
        ax.set_xticks([1, 10, 20, 30, 40])
        ax.set_xlabel("Time Steps", fontsize=10)
        if col_idx == 0: ax.set_ylabel("SE RT Density", fontsize=12)
        _clean_spines(ax)

        # Row 3: SSD Scatter
        ax = axs[3, col_idx]
        # Plot Obs
        ax.scatter(df_obs.index + 1, df_obs['ssd'], label='Obs', 
                   color=COLOR_OBS, s=15, alpha=ALPHA_OBS, marker='o', edgecolors='none')
        # Plot Sim (Overlay)
        sim_trials = len(df_sim) // n_repeat
        current_len = min(sim_trials, len(df_obs))
        ssd_segment = df_sim.iloc[:current_len]['ssd']
        ax.scatter(range(1, current_len + 1), ssd_segment, label='Sim', 
                   color=COLOR_SIM, s=10, alpha=ALPHA_SIM, marker='o', edgecolors='none')

        ax.set_xlim(1, 360)
        ax.set_xticks([1, 120, 240, 360])
        ax.set_xlabel("Trials", fontsize=10)
        if col_idx == 0: ax.set_ylabel("SSD", fontsize=12)
        _clean_spines(ax)
        ax.set_ylim(0, 37)

        # Row 4: Inhibition Function
        ax = axs[4, col_idx]
        prob_obs = _get_prob_ss(df_obs)
        prob_sim = _get_prob_ss(df_sim)
        
        ax.plot(prob_obs.index, prob_obs.values, 'o-', label='Obs', 
                color=COLOR_OBS, alpha=ALPHA_OBS, markersize=4)
        ax.plot(prob_sim.index, prob_sim.values, 'o-', label='Sim', 
                color=COLOR_SIM, alpha=ALPHA_SIM, markersize=4)
        
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 36)
        ax.set_xticks([0, 10, 20, 30, 36])
        ax.set_xlabel("SSD")
        if col_idx == 0: ax.set_ylabel("SS Rates over SSD", fontsize=12)
        _clean_spines(ax)

    # Global Legend
    legend_elements = [
        Patch(facecolor=COLOR_OBS, alpha=ALPHA_OBS, label='Observation'),
        Patch(facecolor=COLOR_SIM, alpha=ALPHA_SIM, label='Simulation'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False, fontsize=12)

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


# --- Helper Functions ---

def _clean_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def _get_prob_ss(df):
    """Calculates probability of Stop Success at each SSD."""
    stop_df = df[df['result'].isin(['SS', 'SE'])].copy()
    if stop_df.empty:
        return pd.Series(dtype=float)
    stop_df['is_ss'] = (stop_df['result'] == 'SS').astype(int)
    return stop_df.groupby('ssd')['is_ss'].mean()


# --- Main Execution ---

def main():
    try:
        # 1. Select Representative Subjects
        subjects = select_representative_subjects(PPC_METRICS_CSV, FILTER_CSV)
        
        # 2. Generate Plots
        plot_model_fit(subjects, RAW_DATA_DIR, PARAMS_CSV, OUTPUT_IMAGE)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()