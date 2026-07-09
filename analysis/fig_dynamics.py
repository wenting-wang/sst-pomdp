import numpy as np
import pandas as pd
import sys
from pathlib import Path
import argparse

# --- Path Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

sys.path.append(str(PROJECT_ROOT))

from core.models import POMDP
from core import simulation
from core import plotting

# =============================================================================
# 0. ARGUMENT PARSING (DEMO vs PROD MODE)
# =============================================================================
parser = argparse.ArgumentParser(description="Run POMDP Dynamics Simulation")
parser.add_argument(
    '--prod', 
    action='store_true', 
    help='Run in PROD mode using local raw data. Usage: run `python script.py --prod`. Defaults to DEMO mode using example data if omitted.'
)
args, unknown = parser.parse_known_args()

DEMO_MODE = not args.prod

# =============================================================================
# 1. CONFIGURATION & PATHS
# =============================================================================

if DEMO_MODE:
    print(">>> RUNNING IN DEMO MODE (Using example data) <<<")
    DATA_DIR = PROJECT_ROOT / "data"
    OUT_DIR = PROJECT_ROOT / "outputs_example"
    PARAMS_CSV = OUT_DIR / "tesbi" / "model_posteriors" / "example_posteriors_6p_v1.csv"
    SUBJECT_ID = 'EXAMPLE_SUB_001'
    
    # Placeholders to prevent NameError
    RAW_DATA_DIR = DATA_DIR / 'example_processed_data'
    USE_PREPROCESSING = False
else:
    print(">>> RUNNING IN PROD MODE (Using local raw data) <<<")
    DATA_DIR = PROJECT_ROOT / "data"
    OUT_DIR = PROJECT_ROOT / "outputs"
    PARAMS_CSV = Path("/Users/w/sst-pomdp/outputs/tesbi/model_posteriors/params_posteriors_6p_v1.csv")
    SUBJECT_ID = 'NDAR_INVRUVFBX1L'
    
    # Processing settings for local data
    RAW_DATA_DIR = Path("/Users/w/Desktop/data/sst_valid_base")
    USE_PREPROCESSING = True 

OUT_DIR.mkdir(parents=True, exist_ok=True) 

# =============================================================================
# 2. MODEL PARAMETERS
# =============================================================================
# Fixed parameters (Constants that do not vary by subject)
FIXED_PARAMS = {
    "inv_temp": 44.6534,
    "q_s_n": 0.0787,
    "rate_stop_trial": 1.0 / 6.0,
    "q_d_n": 0.0798,
    "cost_go_missing": 15.7821,
}

# Parameters to extract from the CSV
DYNAMIC_PARAMS = [
    "q_d",
    "q_s",
    "tau",
    "cost_stop_error",
    "cost_go_error",
    "cost_time"
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_subject_params(subject_id, csv_path, fixed_params=FIXED_PARAMS):
    """
    Loads parameters for a specific subject from a CSV and merges them 
    with fixed constants.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {csv_path}\n"
                                f"Please ensure you have placed the example data in the correct directory.")

    # Load data
    df = pd.read_csv(csv_path)

    # Filter for the specific subject
    subject_df = df[df['subject_id'] == subject_id]

    if subject_df.empty:
        raise ValueError(f"Subject ID {subject_id} not found in {csv_path}")

    # Convert the subject's rows to a dictionary: {param_name: mean_value}
    param_dict = subject_df.set_index('index')['mean'].to_dict()
    clean_params = {k: param_dict[k]
                    for k in DYNAMIC_PARAMS if k in param_dict}

    # Merge with fixed parameters
    final_params = {**clean_params, **fixed_params}

    return final_params

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print(f"Loading parameters for Subject: {SUBJECT_ID}...")

    # 1. Load Parameters
    try:
        params = get_subject_params(SUBJECT_ID, PARAMS_CSV)
        print('tau before:', params['tau'])
        # params['tau'] = int(params['tau']) 
        params['tau'] = 0
        print("Parameters loaded successfully:")
        for k, v in params.items():
            print(f"  {k}: {v}")

    except Exception as e:
        print(f"Error loading parameters: {e}")
        return

    # 2. Initialize POMDP
    print("\nInitializing POMDP Model...")

    pomdp = POMDP(**params)
    pomdp.value_iteration_tensor()

    # 3. Run Simulations
    # print("Running Simulations...")
    # Go Trials
    out_go = simulation.simu_trial_batch(
        model=pomdp,
        true_go_state='right',
        true_stop_state='nonstop',
        ssd=None,
        # batch_size=6000,
        batch_size=600,
        verbose=True
    )
    
    # gm_trials = [trial for trial in out_go if trial['result'] == 'GM']
    # if len(gm_trials) >= 4:
    #     print(f"Successfully found {len(gm_trials)} GM trials in the batch.")
    # else:
    #     print(f"Warning: Only found {len(gm_trials)} GM trials. The model's GM probability is too low.")

    # Stop Trials
    out_stop = simulation.simu_trial_batch(
        model=pomdp,
        true_go_state='right',
        true_stop_state='stop',
        ssd=7,
        # batch_size=1200,
        batch_size=120,
        verbose=True
    )

    # 4. Plotting
    print("Generating Plots...")
    plotting.plot_belief(
        out_go, out_stop, show_cnt=4, saveto=OUT_DIR / 'fig_belief_states.png'
    )

    plotting.plot_action_value(
        out_go, out_stop, show_cnt=4, saveto=OUT_DIR / 'fig_action_values.png'
    )

    plotting.plot_stop_prior_sensitivity(
        model=pomdp,
        params=params,
        rates=(0.05, 0.10, 1/6, 0.20, 0.25, 0.30),
        ssds=np.arange(2, 35, 4),
        # n_session=600,
        n_session=20,
        saveto=OUT_DIR / "fig_stop_prior_sensitivity.png", 
    )
    
    print("Done.")


if __name__ == "__main__":
    main()