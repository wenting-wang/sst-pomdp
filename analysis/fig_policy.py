import sys
import pandas as pd
from pathlib import Path
import traceback
import argparse

# =============================================================================
# 0. ARGUMENT PARSING (DEMO vs PROD MODE)
# =============================================================================
parser = argparse.ArgumentParser(description="Generate Policy Plots")
parser.add_argument(
    '--prod', 
    action='store_true', 
    help='Run in PROD mode using local raw data. Usage: run `python script.py --prod`. Defaults to DEMO mode using example data if omitted.'
)
args, unknown = parser.parse_known_args()

DEMO_MODE = not args.prod

# =============================================================================
# 1. DIRECTORY SETUP & CORE IMPORTS
# =============================================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from core.models import POMDP
from core import simulation
from core import plotting

# =============================================================================
# 2. CONFIGURATION & PATHS
# =============================================================================
if DEMO_MODE:
    print(">>> RUNNING IN DEMO MODE (Using example data) <<<")
    OUT_DIR = PROJECT_ROOT / "outputs_example"
    PARAMS_CSV = PROJECT_ROOT / "outputs_example" / "tesbi" / "model_posteriors" / "example_posteriors_6p_v1.csv"
    SUBJECT_ID = 'EXAMPLE_SUB_001'
else:
    print(">>> RUNNING IN PROD MODE (Using local raw data) <<<")
    OUT_DIR = PROJECT_ROOT / "outputs"
    PARAMS_CSV = Path("/Users/w/sst-pomdp/outputs/tesbi/model_posteriors/params_posteriors_6p_v1.csv")
    SUBJECT_ID = 'XXXXXXXXXXX'

OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 3. MODEL PARAMETERS
# =============================================================================
def get_subject_params(target_id, path):
    """
    Loads posterior parameters for a specific subject and merges them with fixed parameters.
    Includes a fallback for mock datasets where the exact target_id might not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}. Please place your dataset in the 'data' folder."
        )
        
    df = pd.read_csv(path)
    
    # Fallback logic for example/mock datasets
    if target_id not in df['subject_id'].values:
        available_id = df['subject_id'].iloc[0]
        print(f"Warning: Subject '{target_id}' not found. Using '{available_id}' instead.")
        target_id = available_id
        
    sub_df = df[df['subject_id'] == target_id]
    
    # Handle different possible column names for the parameter index
    if 'index' in sub_df.columns:
        p_dict = sub_df.set_index('index')['mean'].to_dict()
    elif 'param' in sub_df.columns:
        p_dict = sub_df.set_index('param')['mean'].to_dict()
    else:
        raise KeyError("Could not find 'index' or 'param' column in posterior data.")

    # Extract dynamic params and merge with fixed
    params = {k: p_dict[k] for k in p_dict}
    merged_params = {**params}
    
    return merged_params, target_id


# --- Execution ---

def main():
    try:
        print(f"Loading params for {SUBJECT_ID}...")
        params, final_subject_id = get_subject_params(SUBJECT_ID, PARAMS_CSV)
        params['tau'] = int(params['tau']) 
        print(f"Original tau: {params['tau']}")
        
        params['tau'] = 0
        
        print(f"\nParameters loaded for {final_subject_id}:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        
        print("\nInitializing POMDP and running Value Iteration...")
        pomdp = POMDP(**params)
        pomdp.value_iteration_tensor()
        
        print("\nRunning simulations...")

        # 1. Run simulations
        out_go = simulation.simu_trial_batch(
            model=pomdp, true_go_state='right', true_stop_state='nonstop',
            ssd=None, batch_size=3000, verbose=True
        )
        
        out_stop = simulation.simu_trial_batch(
            model=pomdp, true_go_state='right', true_stop_state='stop',
            ssd=10, batch_size=600, verbose=True
        )

        # 2. Define mapping for outcome names, data sources, and result codes
        outcome_map = {
            "Go Success":   (out_go, "GS"),
            "Go Error":     (out_go, "GE"),
            "Go Missing":   (out_go, "GM"),
            "Stop Success": (out_stop, "SS"),
            "Stop Error":   (out_stop, "SE")
        }

        # 3. Generate the outcomes dictionary using a comprehension
        all_outcomes = {
            name: [trial for trial in data if trial['result'] == code]
            for name, (data, code) in outcome_map.items()
        }

        print("\nGenerating Plots...")

        # 4. Iterate and plot (full time steps for appendix)
        for outcome_name, (_, code) in outcome_map.items():
            save_path = OUT_DIR / f"fig_policy_{code}.png"
            plotting.plot_policy(
                model=pomdp, 
                out_dict=all_outcomes,
                outcome_name=outcome_name,
                saveto=save_path, 
                dpi=600,
                gamma=0.05
            )
            print(f"  Saved full policy plot: {save_path.name}")
        
        # 5. Plot summary for selected time steps (main text)
        summary_path = OUT_DIR / "fig_policy_summary.png"
        plotting.plot_policy_summary(
            model=pomdp,
            out_dict=all_outcomes,
            selected_timesteps=[3, 17, 33, 39],
            saveto=summary_path,
            dpi=600,
            gamma=0.05
        )
        print(f"  Saved summary policy plot: {summary_path.name}")
        
        print("\nDone.")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()