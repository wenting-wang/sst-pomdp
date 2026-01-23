import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent)) 
from core.models import POMDP
from core import simulation
from core import plotting
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Define base paths
BASE_DIR = Path('/kyb/agpd/wwang/data/')
OUT_DIR = Path('/kyb/agpd/wwang/sst-pomdp/outputs/')

# Input Files
# Update this path to point to your specific summary file (e.g., posterior_summary.csv)
PARAMS_CSV = OUT_DIR / "posterior_summary.csv" 

# Simulation Settings
SUBJECT_ID = 'INVJTCXNL4V'

# Fixed parameters (Constants that do not vary by subject)
FIXED_PARAMS = {
    'cost_go_error': 1.0,
    'cost_go_missing': 1.0,
    'inv_temp': 50.0,
    'rate_stop_trial': 1/6
}

# Parameters to extract from the CSV
DYNAMIC_PARAMS = [
    'q_d_n', 
    'q_d', 
    'q_s_n', 
    'q_s', 
    'cost_stop_error', 
    'cost_time'
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_subject_params(subject_id, csv_path, fixed_params):
    """
    Loads parameters for a specific subject from a CSV and merges them 
    with fixed constants.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {csv_path}")

    # Load data
    df = pd.read_csv(csv_path)
    
    # Filter for the specific subject
    subject_df = df[df['subject_id'] == subject_id]
    
    if subject_df.empty:
        raise ValueError(f"Subject ID {subject_id} not found in {csv_path}")

    # Convert the subject's rows to a dictionary: {param_name: mean_value}
    param_dict = subject_df.set_index('param')['mean'].to_dict()
    clean_params = {k: param_dict[k] for k in DYNAMIC_PARAMS if k in param_dict}

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
        params = get_subject_params(SUBJECT_ID, PARAMS_CSV, FIXED_PARAMS)
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

    # 4. Run Simulations
    print("Running Simulations...")
    # Go Trials
    out_go = simulation.simu_trial_batch(
        model=pomdp,
        true_go_state='right', 
        true_stop_state='nonstop', 
        ssd=None, 
        batch_size=1000, 
        verbose=True
    )
    
    # Stop Trials
    out_stop = simulation.simu_trial_batch(
        model=pomdp,
        true_go_state='right', 
        true_stop_state='stop', 
        ssd=10, 
        batch_size=1000, 
        verbose=True
    )

    # 5. Plotting
    print("Generating Plots...")
    # plotting.plot_simu_trial_batch_belief_5conditions(
    #     out_go, out_stop, show_cnt=5, saveto=OUT_DIR / 'belief.png'
    # )
    
    # plotting.plot_simu_trial_batch_action_value_5conditions(
    #     out_go, out_stop, show_cnt=5, saveto=OUT_DIR / 'action_value.png'
    # )

    plotting.plot_rate_timecourses(
                                model=pomdp,
                                params=params,
                                rates=(0.05, 1/6, 0.35, 0.55, 0.80),
                                ssd=10,
                                n_batch=500, 
                                trial_type="stop",
                                saveto=OUT_DIR / "rate_timecourses_stop.png")

    # # NONSTOP trials (go-only), SSD ignored
    # plotting.plot_rate_timecourses(
    #                             model=pomdp,
    #                             params=params,
    #                             rates=(0.05, 1/6, 0.35, 0.55, 0.80),
    #                             ssd=None,
    #                             n_batch=500,
    #                             trial_type="nonstop",
    #                             saveto=OUT_DIR /"rate_timecourses_go.png")

    # plotting.plot_rate_behavior(
    #     model=pomdp,
    #     params=params,
    #     rates=(0.05, 1/6, 0.35, 0.55, 0.80),
    #     ssd=10,
    #     n_session_trials=500,
    #     saveto=OUT_DIR / "rate_behavior.png"
    # )

    print("Done.")

if __name__ == "__main__":
    main()
