import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional

# Custom Modules (Ensure these are in the same folder)
from utils.preprocessing import preprocessing
from model.pomdp import POMDP
from utils.metrics import get_distance

# --- Configuration ---

# ==========================================
# OPTION 1: Local / Mac Configuration (Active)
# ==========================================
BASE_DIR = Path('/kyb/agpd/wwang/data/')
OUT_DIR = Path('/kyb/agpd/wwang/sst-pomdp/outputs/')

RAW_DATA_DIR = BASE_DIR / "sst_valid_raw"
VALID_FILES_LIST = BASE_DIR / "sst_valid_filenames.txt"
POSTERIOR_SUMMARY_CSV = OUT_DIR / "posterior_summary.csv"
OUTPUT_PPC_CSV = OUT_DIR / "ppc_metrics.csv"


# ==========================================
# OPTION 2: HPC Configuration (Uncomment to use)
# ==========================================
# # Base paths
# RAW_DATA_DIR = Path("/home/wwang/data/sst_valid_raw")
# WORK_DIR = Path("/home/wwang/823879") # Or your specific output folder

# # Input Files
# # Note: This script supports both .txt (list of filenames) and .csv (dataframe with 'filename' col)
# VALID_FILES_LIST = Path("/home/wwang/sst_valid_filenames.csv") 
# POSTERIOR_SUMMARY_CSV = Path("/home/wwang/ptmp/823879/outputs/posterior_summary.csv")

# # Output File
# OUTPUT_PPC_CSV = Path("/home/wwang/823879/ppc_metrics.csv")
# ==========================================


# Model Constants
FIXED_PARAMS = {
    "cost_go_error": 1.0, 
    "cost_go_missing": 1.0, 
    "inv_temp": 50.0,
}
PARAM_ORDER = ["q_d_n", "q_d", "q_s_n", "q_s", "cost_stop_error", "cost_time"]


@dataclass
class SimConfig:
    """Configuration for PPC Simulation."""
    n_rep: int = 1         # Single point estimate simulation
    rate_stop_trial: float = 1.0/6.0
    seed: int = 2025


# --- Core Helper Functions ---

def load_posterior_mean(summary_csv: Path, subject_id: str) -> Dict[str, float]:
    """
    Loads the MAP (Mean) parameter estimates for a specific subject.
    """
    # Load entire summary (cached by OS usually, or optimized if passed as df)
    # For efficiency in a loop, it's better to load this once outside, but here is per-call safe.
    df = pd.read_csv(summary_csv)
    
    # Filter for subject
    subset = df[df["subject_id"] == subject_id]
    
    if subset.empty:
        raise ValueError(f"No parameters found for subject {subject_id}")

    theta = {}
    for param in PARAM_ORDER:
        row = subset[subset["param"] == param]
        if not row.empty:
            theta[param] = float(row["mean"].iloc[0])
            
    # Add fixed parameters
    theta.update(FIXED_PARAMS)
    return theta


def simulate_task(params: Dict[str, float], config: SimConfig, seed: int) -> Optional[pd.DataFrame]:
    """
    Runs one simulation of the POMDP task with the given parameters.
    """
    try:
        np.random.seed(seed)
        
        # Initialize and Run Model
        agent = POMDP(**params, rate_stop_trial=config.rate_stop_trial)
        agent.value_iteration_tensor()
        sim_data = agent.simu_task()
        
        # Convert to DataFrame
        # Columns must match what 'metrics.py' expects: 'result', 'rt', 'ssd'
        return pd.DataFrame(sim_data, columns=["result", "rt", "ssd"])
        
    except Exception as e:
        print(f"Simulation failed for params {params}: {e}")
        return None


def run_ppc_single_subject(subject_id: str, filename: str, config: SimConfig) -> dict:
    """
    Worker function: Loads observed data, simulates model, and computes discrepancy metrics.
    """
    try:
        file_path = RAW_DATA_DIR / filename
        
        # 1. Load MAP Parameters
        theta = load_posterior_mean(POSTERIOR_SUMMARY_CSV, subject_id)

        # 2. Load Observed Data
        if not file_path.exists():
            return {"subject_id": subject_id, "error": "Raw file not found"}
            
        df_obs = preprocessing(str(file_path))
        
        # 3. Simulate Data (PPC)
        # Create a deterministic seed based on subject ID to ensure reproducibility
        sid_hash = (abs(hash(subject_id)) % 1_000_000)
        sim_seed = config.seed + sid_hash
        
        df_sim = simulate_task(theta, config, seed=sim_seed)
        
        if df_sim is None or df_sim.empty:
            return {"subject_id": subject_id, "error": "Simulation returned empty"}

        # 4. Compute Metrics (Distance between Obs and Sim)
        # Unpack the tuple returned by get_distance
        (
            d_p_gs, d_p_ge, d_p_gm, d_p_ss,
            d_ws_gs, d_ws_ge, d_ws_se,
            d_ks_gs, d_ks_se,
            d_ssd_mean
        ) = get_distance(df_obs, df_sim)

        return {
            "subject_id": subject_id,
            "filename": filename,
            "dis_perc_gs": d_p_gs,
            "dis_perc_ge": d_p_ge,
            "dis_perc_gm": d_p_gm,
            "dis_perc_ss": d_p_ss,
            "dis_ws_rt_gs": d_ws_gs,
            "dis_ws_rt_ge": d_ws_ge,
            "dis_ws_rt_se": d_ws_se,
            "dis_ks_rt_gs": d_ks_gs,
            "dis_ks_rt_se": d_ks_se,
            "dis_ssd_mean": d_ssd_mean,
            "error": None
        }

    except Exception as e:
        return {"subject_id": subject_id, "filename": filename, "error": str(e)}


# --- Main Execution ---

def main():
    # 1. Setup
    config = SimConfig()
    
    # Load list of valid files to process
    if not VALID_FILES_LIST.exists():
        print(f"Error: File list not found at {VALID_FILES_LIST}")
        return

    with open(VALID_FILES_LIST, 'r') as f:
        filenames = [line.strip() for line in f if line.strip()]

    print(f"Starting PPC for {len(filenames)} subjects...")
    print(f"Using Posterior Summary: {POSTERIOR_SUMMARY_CSV}")

    results = []
    
    # 2. Parallel Processing
    # Adjust max_workers based on your CPU cores
    with ProcessPoolExecutor() as executor:
        futures = {}
        for fname in filenames:
            # Extract ID from filename: "NDAR_INV123_baseline..." -> "INV123"
            parts = fname.split('_')
            if len(parts) > 1:
                sid = parts[1]
                futures[executor.submit(run_ppc_single_subject, sid, fname, config)] = sid

        # 3. Collect Results
        for future in as_completed(futures):
            sid = futures[future]
            try:
                res = future.result()
                if res.get("error"):
                    print(f"[{sid}] Failed: {res['error']}")
                else:
                    # Optional: Print a dot or small log to show progress
                    pass 
                results.append(res)
            except Exception as e:
                print(f"[{sid}] Critical Exception: {e}")

    # 4. Save Output
    df_results = pd.DataFrame(results)
    
    # Move 'error' column to the end or drop rows with errors
    if 'error' in df_results.columns:
        df_clean = df_results[df_results['error'].isna()].drop(columns=['error'])
        print(f"Simulation completed. Success: {len(df_clean)}/{len(filenames)}")
    else:
        df_clean = df_results
        
    OUTPUT_PPC_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_PPC_CSV, index=False)
    print(f"Saved PPC metrics to: {OUTPUT_PPC_CSV}")


if __name__ == "__main__":
    main()