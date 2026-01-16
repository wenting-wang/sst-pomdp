import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional

# Custom Modules (Ensure these are in the same folder)
from utils.preprocessing import preprocessing
from utils.metrics import get_distance
from core.models import POMDP
from core import simulation

# --- Configuration ---

# ==========================================
# OPTION 1: Local / Mac Configuration (Local Test)
# ==========================================
# BASE_DIR = Path('/kyb/agpd/wwang/data/')
# OUT_DIR = Path('/kyb/agpd/wwang/sst-pomdp/outputs/')

# RAW_DATA_DIR = BASE_DIR / "sst_valid_base"
# POSTERIOR_SUMMARY_CSV = OUT_DIR / "posterior_summary.csv"
# OUTPUT_PPC_CSV = OUT_DIR / "ppc_metrics.csv"

# ==========================================
# OPTION 2: HPC Configuration
# ==========================================
# Base paths
RAW_DATA_DIR = Path("/home/wwang/data/sst_valid_base")
WORK_DIR = Path("/home/wwang/831633") 

# Input: Posterior Summary
POSTERIOR_SUMMARY_CSV = Path("/home/wwang/ptmp/831633/outputs/posterior_summary.csv")

# Output: PPC Metrics
OUTPUT_PPC_CSV = Path("/home/wwang/831633/ppc_metrics.csv")
# ==========================================


# Model Constants
FIXED_PARAMS = {
    "cost_go_error":   1.0,
    "cost_go_missing": 1.0,
    "inv_temp":        50.0,
    "rate_stop_trial": 1.0 / 6.0,
}

LINEAR_PARAMS = ["q_d_n", "q_d", "q_s_n", "q_s"]
LOG_PARAMS = ["cost_stop_error", "cost_time", "ndt"]
PARAM_ORDER = LINEAR_PARAMS + LOG_PARAMS


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
    # Load entire summary
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
        agent = POMDP(**params)
        agent.value_iteration_tensor()
        sim_data = simulation.simu_task(agent)
        
        # Convert to DataFrame
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
    
    # Check Directory
    if not RAW_DATA_DIR.exists():
        print(f"Error: Data directory not found at {RAW_DATA_DIR}")
        return

    # 2. Get All Files directly from Directory
    # We iterate over the directory and grab all non-hidden files
    file_paths = [p for p in RAW_DATA_DIR.iterdir() if p.is_file() and not p.name.startswith('.')]
    
    # Extract just the filenames (strings) as the worker expects
    filenames = [p.name for p in file_paths]

    print(f"Found {len(filenames)} files in {RAW_DATA_DIR}")
    print(f"Using Posterior Summary: {POSTERIOR_SUMMARY_CSV}")

    results = []
    
    # 3. Parallel Processing
    with ProcessPoolExecutor() as executor:
        futures = {}
        for fname in filenames:
            # Extract ID from filename: "NDAR_INV123_baseline..." -> "INV123"
            parts = fname.split('_')
            
            # Safety check: Ensure filename format is correct before processing
            if len(parts) > 1:
                sid = parts[1]
                futures[executor.submit(run_ppc_single_subject, sid, fname, config)] = sid
            else:
                print(f"Skipping file with unexpected format: {fname}")

        # 4. Collect Results
        for future in as_completed(futures):
            sid = futures[future]
            try:
                res = future.result()
                if res.get("error"):
                    print(f"[{sid}] Failed: {res['error']}")
                else:
                    # Success
                    pass 
                results.append(res)
            except Exception as e:
                print(f"[{sid}] Critical Exception: {e}")

    # 5. Save Output
    if not results:
        print("No results generated. Check input directory.")
        return

    df_results = pd.DataFrame(results)
    
    # Clean errors
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