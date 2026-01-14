import sys
import pandas as pd
from pathlib import Path

# Fix path to allow importing 'model'
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from core.models import POMDP
from core import plotting

# --- Configuration ---
OUT_DIR = Path('/kyb/agpd/wwang/sst-pomdp/outputs/')
PARAMS_CSV = OUT_DIR / "posterior_summary.csv"
SUBJECT_ID = 'INVJTCXNL4V'

FIXED_PARAMS = {
    'cost_go_error': 1.0,
    'cost_go_missing': 1.0,
    'inv_temp': 50.0,
    'ndt': 0,
    'rate_stop_trial': 1/6
}

# Params to extract from CSV
DYNAMIC_KEYS = ['q_d_n', 'q_d', 'q_s_n', 'q_s', 'cost_stop_error', 'cost_time']

# --- Helper ---
def get_subject_params(subject_id, path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    df = pd.read_csv(path)
    sub_df = df[df['subject_id'] == subject_id]
    
    if sub_df.empty:
        raise ValueError(f"Subject {subject_id} not found in {path}")

    # Extract dynamic params and merge with fixed
    p_dict = sub_df.set_index('param')['mean'].to_dict()
    dynamic = {k: p_dict[k] for k in DYNAMIC_KEYS if k in p_dict}
    return {**dynamic, **FIXED_PARAMS}

# --- Execution ---
if __name__ == "__main__":
    print(f"Loading params for {SUBJECT_ID}...")
    params = get_subject_params(SUBJECT_ID, PARAMS_CSV)

    print("Parameters loaded:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    print("Running Value Iteration...")
    pomdp = POMDP(**params)
    pomdp.value_iteration_tensor()

    print("Generating Plots...")
    plotting.plot_policy_simple(pomdp, OUT_DIR / 'policy_selected.png', dpi=300, selected_timesteps=[0, 37, 38, 39])
    plotting.plot_policy(pomdp, OUT_DIR / 'policy_full.png', dpi=300)
    print("Done.")