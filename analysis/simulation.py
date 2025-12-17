import pandas as pd
from pathlib import Path

# --- Custom Imports ---
from model.pomdp import POMDP

# =============================================================================
# CONFIGURATION
# =============================================================================

# Define base paths (Adjust these to match your environment)
BASE_DIR = Path('/kyb/agpd/wwang/data/')
OUT_DIR = Path('/kyb/agpd/wwang/sst-pomdp/outputs/')

# Input Files
PARAMS_CSV = OUT_DIR / "posterior_summary.csv"
STATS_CSV = OUT_DIR / "stats_beh_full.csv"

# Parameter Selection Method: 'mode', 'mean', or 'median'
METHOD = 'mode'

# Default parameters (fallback if files are missing)
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

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_parameters(posterior_path, stats_path, method='mode'):
    """
    Loads posterior summaries and demographic stats to calculate 
    parameters for the baseline year using the specified method.
    
    Args:
        posterior_path (Path): Path to the posterior summary CSV.
        stats_path (Path): Path to the stats CSV.
        method (str): Method to aggregate parameters ('mode', 'mean', 'median').
        
    Returns:
        dict: A dictionary of parameter values.
    """
    if not posterior_path.exists() or not stats_path.exists():
        print(f"[WARN] Data files not found. Using default hardcoded parameters.")
        return DEFAULT_PARAMS

    print(f"Loading data to calculate parameters using method: '{method}'...")
    
    # 1. Load and Pivot Posterior Data
    df_param = pd.read_csv(posterior_path)
    df_param = df_param.rename(columns={"subject_year": "year"})
    
    # Reshape: one row per subject/year, columns for each parameter mean
    df_param_pivot = df_param.pivot(
        index=['subject_id', 'year'], 
        columns='param', 
        values='mean'
    ).reset_index()
    df_param_pivot.columns.name = None

    # 2. Load Stats Data and Merge
    df_stats = pd.read_csv(stats_path)
    df = pd.merge(df_stats, df_param_pivot, on=['subject_id', 'year'], how='inner')
    df = df.dropna()

    # 3. Filter for Baseline
    df_baseline = df[df['year'] == 'baseline']

    if df_baseline.empty:
        raise ValueError("Merged dataframe is empty after filtering for 'baseline'.")

    # 4. Extract Parameters based on Method
    param_keys = list(DEFAULT_PARAMS.keys())
    
    if method == 'mode':
        # Mode returns a DataFrame; take the first row.
        # Rounding to 3 decimals helps find a meaningful mode.
        params = df_baseline[param_keys].round(3).mode().to_dict(orient='records')[0]
    
    elif method == 'mean':
        # Mean returns a Series, convertible directly to dict
        params = df_baseline[param_keys].mean().to_dict()
        
    elif method == 'median':
        # Median returns a Series, convertible directly to dict
        params = df_baseline[param_keys].median().to_dict()
        
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'mode', 'mean', or 'median'.")

    return params


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # 1. Ensure output directory exists
    if not OUT_DIR.exists():
        print(f"Creating output directory: {OUT_DIR}")
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load Parameters
    try:
        params = load_parameters(PARAMS_CSV, STATS_CSV, method=METHOD)
    except Exception as e:
        print(f"[ERROR] Failed to load parameters: {e}")
        return

    # 3. Initialize and Solve POMDP
    print("Initializing POMDP model...")
    pomdp = POMDP(**params)
    
    print("Running Value Iteration (this may take a moment)...")
    pomdp.value_iteration_tensor()

    # 4. Generate Plots
    print(f"Generating policy plots in {OUT_DIR}...")
    
    # Plot 1: Simple Policy (Selected Timesteps)
    save_path_simple = OUT_DIR / 'policy_selected.png'
    pomdp.plot_policy_simple(
        save_path_simple,
        dpi=300, 
        selected_timesteps=[0, 16, 38, 39]
    )
    print(f"Saved: {save_path_simple}")

    # Plot 2: Full Policy
    save_path_full = OUT_DIR / 'policy_full.png'
    pomdp.plot_policy(
        save_path_full, 
        dpi=300
    )
    print(f"Saved: {save_path_full}")
    
    print("Done.")

if __name__ == "__main__":
    main()