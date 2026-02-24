import pandas as pd
import numpy as np
import os
import io
import sys
import random
import torch
from typing import Dict, Optional
from pathlib import Path

# ==========================================
# Setup Paths & Imports
# ==========================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# Add project root to sys.path to import local modules
sys.path.append(str(PROJECT_ROOT)) 

# Import your POMDP model and simulation logic from the core module
from core.models import POMDP
from core import simulation

# Input paths
emb_path = DATA_DIR / 'embeddings.csv'
stats_path = DATA_DIR / 'clinical_behavior.csv'
ppc_path = DATA_DIR / 'ppc_metrics.csv'
rec_path = DATA_DIR / 'params_recovery.csv'
post_path = DATA_DIR / 'params_posteriors.csv'

# Output paths for the generated datasets
fake_emb_output = DATA_DIR / 'example_embeddings.csv'
fake_stats_output = DATA_DIR / 'example_clinical_behavior.csv'
fake_post_output = DATA_DIR / 'example_params_posteriors.csv'
fake_ppc_output = DATA_DIR / 'example_ppc_metrics.csv'
fake_rec_output = DATA_DIR / 'example_params_recovery.csv'

# Output directory for the 100 trial-by-trial CSVs
processed_data_dir = DATA_DIR / 'example_processed_data'
os.makedirs(processed_data_dir, exist_ok=True)

# Load the real datasets using Pathlib's .exists()
emb = pd.read_csv(emb_path) if emb_path.exists() else pd.DataFrame()
stats = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
ppc = pd.read_csv(ppc_path) if ppc_path.exists() else pd.DataFrame()
rec = pd.read_csv(rec_path) if rec_path.exists() else pd.DataFrame()
post = pd.read_csv(post_path) if post_path.exists() else pd.DataFrame()

# N = 100 Mock Subjects
N = 100
fake_subjects = [f'EXAMPLE_SUB_{i:03d}' for i in range(1, N+1)]
year = 'baseline'
fake_filenames = [f'{sub}.csv' for sub in fake_subjects]

# ==========================================
# 1-2. Generate Embeddings & Clinical Behavior
# ==========================================
if not emb.empty:
    emb_num = emb.select_dtypes(include=[np.number]).columns
    emb_samples = np.random.multivariate_normal(emb[emb_num].mean(), emb[emb_num].cov(), size=N)
    fake_emb_df = pd.DataFrame(emb_samples, columns=emb_num)
    fake_emb_df.insert(0, 'subject_id', fake_subjects)
    fake_emb_df.insert(1, 'year', [year]*N)
    fake_emb_df.to_csv(fake_emb_output, index=False)
    print(f"Saved {fake_emb_output.name}")

if not stats.empty:
    stats_num = stats.select_dtypes(include=[np.number]).columns
    fake_stats_data = {'subject_id': fake_subjects, 'year': [year]*N, 'filename': fake_filenames}
    non_num = [c for c in stats.columns if c not in stats_num and c not in ['subject_id', 'year', 'filename']]
    for col in non_num:
        fake_stats_data[col] = np.random.choice(stats[col].dropna().unique(), size=N)
    stats_samples = np.random.multivariate_normal(stats[stats_num].mean(), stats[stats_num].cov(), size=N)
    fake_stats_df = pd.DataFrame(stats_samples, columns=stats_num)
    for col in stats_num:
        if 'perc' in col or 'rate' in col:
            fake_stats_df[col] = fake_stats_df[col].clip(0, 1)
        elif 'rt' in col or 'mrt' in col or 'ssrt' in col:
            fake_stats_df[col] = fake_stats_df[col].clip(1e-6, None) # Strictly > 0
    for key, val in fake_stats_data.items():
        fake_stats_df[key] = val
    if 'adhd' in fake_stats_df.columns:
        fake_stats_df['adhd'] = np.random.randint(1, 15, size=N)
    if 'adhd_med_flag' in fake_stats_df.columns:
        fake_stats_df['adhd_med_flag'] = np.random.randint(0, 2, size=N)
    fake_stats_df = fake_stats_df[stats.columns]
    fake_stats_df.to_csv(fake_stats_output, index=False)
    print(f"Saved {fake_stats_output.name}")

# ==========================================
# 3. Simulate Trial Data & Params Posteriors 
# ==========================================
def simulate_data(params: Dict[str, float], seed: Optional[int] = None) -> pd.DataFrame:
    """Runs the POMDP simulator for a single parameter set."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    
    pomdp = POMDP(**params)
    pomdp.value_iteration_tensor()
    arr = simulation.simu_task(pomdp, verbose=True)
    return pd.DataFrame(arr, columns=["true_go_state", "true_stop_state", "ssd", "result", "rt"])

print(f"\nGenerating Params Posteriors and Trial-by-Trial Processed CSVs in {processed_data_dir.name}/ ...")

target_data_csv = """index,mean,std,5%,50%,95%,subject_id,subject_year
q_d_n,0.5182775412737392,0.21021129100844552,0.12152421623468398,0.5575239956378937,0.8309199392795562,EXAMPLE_SUB_001,baseline
q_d,0.7255604705810547,0.054398157303961414,0.6588087648153305,0.7155599594116211,0.8373038649559021,EXAMPLE_SUB_001,baseline
q_s_n,0.01852275689443104,0.016815355518417827,0.0015628402004949748,0.013593575917184353,0.05104220621287822,EXAMPLE_SUB_001,baseline
q_s,0.8391608062088489,0.06945543332067991,0.7337219953536988,0.8332814276218414,0.9547415614128113,EXAMPLE_SUB_001,baseline
inv_temp,36.17459666824341,10.188783578038375,25.87865686416626,33.67682456970215,55.620153617858875,EXAMPLE_SUB_001,baseline
cost_stop_error,1.6490949155036496,0.260155391235878,1.1248021343503272,1.708300738172946,1.966143764511714,EXAMPLE_SUB_001,baseline
cost_time,0.0010000000000000005,4.337893298859799e-19,0.001,0.001,0.001,EXAMPLE_SUB_001,baseline
cost_go_error,1.0,0.0,1.0,1.0,1.0,EXAMPLE_SUB_001,baseline
cost_go_missing,1.0,0.0,1.0,1.0,1.0,EXAMPLE_SUB_001,baseline
rate_stop_trial,0.16666666666666663,0.0,0.16666666666666666,0.16666666666666666,0.16666666666666666,EXAMPLE_SUB_001,baseline"""

expected_keys = [
    'q_d_n', 'q_d', 'q_s_n', 'q_s', 
    'cost_go_error', 'cost_go_missing', 'cost_stop_error', 'cost_time', 
    'inv_temp', 'rate_stop_trial'
]

# Process target subject directly
target_df = pd.read_csv(io.StringIO(target_data_csv))
target_params = {k: target_df.set_index('index')['mean'].to_dict().get(k, 1e-6) for k in expected_keys}
mock_processed_df = simulate_data(target_params)
mock_processed_df.to_csv(processed_data_dir / fake_filenames[0], index=False)

fake_post_data = []

# Prepare statistics to generate remaining subjects
if not post.empty:
    unique_indices = post['index'].unique()
    post_num_cols = ['mean', 'std', '5%', '50%', '95%']
    post_grouped = post.groupby('index')[post_num_cols].agg(['mean', 'std'])

for sub, fname in zip(fake_subjects[1:], fake_filenames[1:]):
    success = False
    
    # Loop until we successfully generate a valid subject
    while not success:
        temp_rows = []
        proposed_params = {}
        
        if not post.empty:
            for idx in unique_indices:
                new_row = {'index': idx, 'subject_id': sub, 'subject_year': year}
                for col in post_num_cols:
                    c_mean = post_grouped.loc[idx, (col, 'mean')]
                    c_std = post_grouped.loc[idx, (col, 'std')]
                    if pd.isna(c_std) or c_std == 0:
                        val = c_mean
                    else:
                        val = np.random.normal(loc=c_mean, scale=c_std)
                    
                    val = max(1e-6, val) # Strictly > 0
                    if 'q_' in idx or 'rate_' in idx:
                        val = min(0.999999, val) # Prevent Probability > 1
                        
                    new_row[col] = val
                    if col == 'mean':
                        proposed_params[idx] = val
                temp_rows.append(new_row)
        else:
            for _, row in target_df.iterrows():
                idx = row['index']
                new_row = {'index': idx, 'subject_id': sub, 'subject_year': year}
                for col in ['mean', 'std', '5%', '50%', '95%']:
                    val = row[col]
                    if row['std'] > 0 and 'cost_go' not in row['index'] and 'cost_time' not in row['index']:
                        val = np.random.normal(loc=val, scale=max(row['std'] * 0.1, 0.001))
                    
                    val = max(1e-6, val) # Strictly > 0
                    if 'q_' in idx or 'rate_' in idx:
                        val = min(0.999999, val) # Prevent Probability > 1
                        
                    new_row[col] = val
                    if col == 'mean':
                        proposed_params[idx] = val
                temp_rows.append(new_row)
        
        # Test the generated parameters in the simulator
        sim_params = {k: proposed_params.get(k, 1e-6) for k in expected_keys}
        
        try:
            mock_processed_df = simulate_data(sim_params)
            
            # If no errors were thrown, append the parameter data and save the CSV
            fake_post_data.extend(temp_rows)
            mock_processed_df.to_csv(processed_data_dir / fname, index=False)
            success = True
            
        except Exception as e:
            # Print error and retry for this specific subject
            print(f"[{sub}] Invalid parameters generated ({e}). Resampling...")

fake_post_df = pd.concat([target_df, pd.DataFrame(fake_post_data)], ignore_index=True)
fake_post_df = fake_post_df[['index', 'mean', 'std', '5%', '50%', '95%', 'subject_id', 'subject_year']]
fake_post_df.to_csv(fake_post_output, index=False)
print(f"Saved {fake_post_output.name}")
print(f"Successfully generated {N} processed simulated data files.")

# ==========================================
# 4-5. Fake PPC Metrics & Params Recovery
# ==========================================
if not ppc.empty:
    ppc_num_cols = ppc.select_dtypes(include=[np.number]).columns
    ppc_samples = np.random.multivariate_normal(ppc[ppc_num_cols].mean(), ppc[ppc_num_cols].cov(), size=N)
    fake_ppc_df = pd.DataFrame(ppc_samples, columns=ppc_num_cols).clip(lower=1e-6)
    fake_ppc_df.insert(0, 'subject_id', fake_subjects)
    fake_ppc_df.insert(1, 'filename', fake_filenames)
    fake_ppc_df.to_csv(fake_ppc_output, index=False)
    print(f"Saved {fake_ppc_output.name}")

if not rec.empty:
    hit_cols = [c for c in rec.columns if 'hit90' in c]
    cont_cols = [c for c in rec.columns if c not in hit_cols and c != 'case']
    rec_samples = np.random.multivariate_normal(rec[cont_cols].mean(), rec[cont_cols].cov(), size=N)
    
    fake_rec_df = pd.DataFrame(rec_samples, columns=cont_cols).clip(lower=1e-6)
    for col in cont_cols:
        if 'q_' in col:
            fake_rec_df[col] = fake_rec_df[col].clip(upper=1.0)
            
    for hit_col in hit_cols:
        prob = rec[hit_col].mean()
        fake_rec_df[hit_col] = np.random.choice([0.0, 1.0], size=N, p=[1-prob, prob])
        
    fake_rec_df.insert(0, 'case', range(N))
    fake_rec_df = fake_rec_df[rec.columns]
    fake_rec_df.to_csv(fake_rec_output, index=False)
    print(f"Saved {fake_rec_output.name}")