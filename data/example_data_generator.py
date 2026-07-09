import pandas as pd
import numpy as np
from pathlib import Path
import random
import torch
from typing import Dict, Optional
import sys

# --- Path Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from core.models import POMDP
from core import simulation

# --- Real Data Paths (Ensure these point to your actual local data) ---
REAL_DATA_DIR = Path("/Users/w/sst-pomdp/data")
REAL_OUT_DIR = Path("/Users/w/sst-pomdp/outputs/tesbi")

# --- Mock Data Output Paths ---
MOCK_DATA_DIR = PROJECT_ROOT / "data"
MOCK_OUT_DIR = PROJECT_ROOT / "outputs_example" / "tesbi"
PROCESSED_DATA_DIR = MOCK_DATA_DIR / "example_processed_data"

MOCK_OUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# MODELS = ['10p_v1', '8p_v2', '7p_v3', '6p_v1']
MODELS = ['10p_v1', 
          '8p_v1', '8p_v2', '8p_v3', '8p_v4', '8p_v5',
          '7p_v1', '7p_v2', '7p_v3', '7p_v4', '7p_v5', '7p_v6', 
          '6p_v1', '6p_v2', '6p_v3']
BEST_MODEL = '6p_v1'

N_SUBJECTS = 10

# =============================================================================
# 2. Helper Functions
# =============================================================================
def generate_mvn_samples(df, num_cols, n_samples):
    """Generate multivariate normal mock data based on real covariance."""
    # 1. Force convert all target columns to numeric, setting invalid ones to NaN
    df_num = df[num_cols].copy().apply(pd.to_numeric, errors='coerce')
    
    # 2. Drop columns that are completely NaN
    df_num = df_num.dropna(axis=1, how='all')
    valid_cols = df_num.columns.tolist()
    
    if len(df_num.dropna()) <= 1:
        # If there are not enough complete rows, fill NaNs with column means first
        df_num = df_num.fillna(df_num.mean())
        
    means = df_num.mean()
    cov = df_num.cov()
    
    # 3. Fill any remaining NaNs in covariance (e.g., from zero-variance columns)
    cov = cov.fillna(0)
    
    # Add a tiny diagonal perturbation to ensure the matrix is positive semi-definite
    cov += np.eye(len(valid_cols)) * 1e-6 
    
    # Generate samples
    samples = np.random.multivariate_normal(means, cov, size=n_samples)
    
    # Reconstruct dataframe and clip lower bound
    res_df = pd.DataFrame(samples, columns=valid_cols).clip(lower=1e-6)
    
    # 4. Restore any columns that were completely empty (fill with 0.0) to keep schema intact
    for c in num_cols:
        if c not in res_df.columns:
            res_df[c] = 0.0
            
    return res_df[num_cols]


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

# =============================================================================
# 3. Execution
# =============================================================================
def main():
    print(f"Starting Example Data Generation (N={N_SUBJECTS} subjects)...")
    
    # Generate fake subject IDs
    fake_subjects = [f"EXAMPLE_SUB_{str(i).zfill(3)}" for i in range(1, N_SUBJECTS + 1)]
    
    # =========================================================================
    # A. Joint Generation for BEST_MODEL and Clinical Data
    # =========================================================================
    print(f"Joining core datasets for {BEST_MODEL} to preserve covariance...")
    
    # 1. Load and prep Clinical Behavior
    real_clin_path = REAL_DATA_DIR / 'clinical_behavior.csv'
    df_clin = pd.read_csv(real_clin_path) if real_clin_path.exists() else pd.DataFrame()
    clin_cols = []
    if not df_clin.empty:
        clin_cols = df_clin.select_dtypes(include=[np.number]).columns.drop('adhd', errors='ignore').tolist()
        df_clin = df_clin[['subject_id'] + clin_cols]
        df_clin.columns = ['subject_id'] + [f"clin_{c}" for c in clin_cols]

    # (Embeddings loading removed)

    # 2. Load and prep PPC Metrics
    real_ppc_path = REAL_OUT_DIR / f'ppc_metrics_{BEST_MODEL}.csv'
    df_ppc = pd.read_csv(real_ppc_path) if real_ppc_path.exists() else pd.DataFrame()
    ppc_cols = []
    if not df_ppc.empty:
        ppc_cols = df_ppc.select_dtypes(include=[np.number]).columns.drop('subject_id', errors='ignore').tolist()
        df_ppc = df_ppc[['subject_id'] + ppc_cols]
        df_ppc.columns = ['subject_id'] + [f"ppc_{c}" for c in ppc_cols]

    # 3. Load and prep Parameter Posteriors
    real_post_path = REAL_OUT_DIR / f'params_posteriors_{BEST_MODEL}.csv'
    df_post = pd.read_csv(real_post_path) if real_post_path.exists() else pd.DataFrame()
    post_cols = []
    if not df_post.empty:
        df_post_wide = df_post.pivot_table(index='subject_id', columns='index', values='mean').reset_index()
        post_cols = [c for c in df_post_wide.columns if c != 'subject_id']
        df_post_wide.columns = ['subject_id'] + [f"post_{c}" for c in post_cols]
    else:
        df_post_wide = pd.DataFrame()

    # Merge all available dataframes (df_emb removed from the list)
    dfs_to_merge = [df for df in [df_clin, df_ppc, df_post_wide] if not df.empty]
    if dfs_to_merge:
        for df in dfs_to_merge:
            df['subject_id'] = df['subject_id'].astype(str).str.strip()

        merged_df = dfs_to_merge[0]
        for df in dfs_to_merge[1:]:
            merged_df = merged_df.merge(df, on='subject_id', how='outer')
            
        # MVN Sampling on joined dataframe
        all_num_cols = [c for c in merged_df.columns if c != 'subject_id']
        fake_merged = generate_mvn_samples(merged_df, all_num_cols, N_SUBJECTS)
        fake_merged.insert(0, 'subject_id', fake_subjects)
        
        
        # --- Split and Save Clinical Data ---
        if clin_cols:
            fake_clin = fake_merged[['subject_id'] + [f"clin_{c}" for c in clin_cols]].copy()
            fake_clin.columns = ['subject_id'] + clin_cols
            fake_clin['year'] = 'baseline'
            fake_clin['sex'] = np.random.choice(['Female', 'Male'], N_SUBJECTS)
            fake_clin['adhd'] = np.random.poisson(lam=2.5, size=N_SUBJECTS)
            fake_clin['adhd_med_flag'] = np.random.choice([0, 1], N_SUBJECTS, p=[0.9, 0.1])
            
            # Add filename column based on subject_id
            fake_clin['filename'] = fake_clin['subject_id'].astype(str) + ".csv"
            
            # Reorder columns to put 'filename' right after 'subject_id'
            cols = fake_clin.columns.tolist()
            cols.remove('filename')
            cols.insert(1, 'filename')
            fake_clin = fake_clin[cols]
            
            out_clin = MOCK_DATA_DIR / 'example_clinical_behavior.csv'
            fake_clin.to_csv(out_clin, index=False)
            print(f"-> Generated {out_clin.name}")

        # (Embeddings saving removed)
            
        # --- Split and Save PPC Metrics ---
        if ppc_cols:
            fake_ppc_res = fake_merged[['subject_id'] + [f"ppc_{c}" for c in ppc_cols]].copy()
            fake_ppc_res.columns = ['subject_id'] + ppc_cols
            fake_ppc_res['year'] = 'baseline'
            out_ppc = MOCK_OUT_DIR / f'example_ppc_metrics_{BEST_MODEL}.csv'
            fake_ppc_res.to_csv(out_ppc, index=False)
            print(f"-> Generated {out_ppc.name}")
            
        # --- Split, Save, and Simulate Best Model Posteriors ---
        if post_cols:
            fake_post_wide = fake_merged[['subject_id'] + [f"post_{c}" for c in post_cols]].copy()
            fake_post_wide.columns = ['subject_id'] + post_cols
            
            # Melt back to long format for saving
            fake_post_long = fake_post_wide.melt(id_vars='subject_id', var_name='index', value_name='mean')
            fake_post_long['std'] = df_post['std'].mean() if 'std' in df_post.columns else 0.1
            fake_post_long['5%'] = fake_post_long['mean'] * 0.85
            fake_post_long['50%'] = fake_post_long['mean']
            fake_post_long['95%'] = fake_post_long['mean'] * 1.15
            fake_post_long['subject_year'] = 'baseline'
            fake_post_long = fake_post_long[['index', 'mean', 'std', '5%', '50%', '95%', 'subject_id', 'subject_year']]
            
            out_post = MOCK_OUT_DIR / f'example_posteriors_{BEST_MODEL}.csv'
            fake_post_long.to_csv(out_post, index=False)
            print(f"-> Generated {out_post.name}")
            
            # Simulate trial-level data using the extracted best model parameters
            print(f"-> Generating trial-by-trial data using parameters from {BEST_MODEL}...")
            for _, row in fake_post_wide.iterrows():
                subject = row['subject_id']
                try:
                    subj_params = {k: float(v) for k, v in row.drop('subject_id').to_dict().items()}
                    df_trial = simulate_data(subj_params)
                    trial_out_path = PROCESSED_DATA_DIR / f"{subject}.csv"
                    df_trial.to_csv(trial_out_path, index=False)
                except Exception as e:
                    print(f"   [Warning] Skipped {subject} due to simulation error: {repr(e)}")
            print(f"-> Finished generating trial data in {PROCESSED_DATA_DIR.name}/")

    # =========================================================================
    # B. Generate Multi-Model Data (Excluding Best Model Posteriors already done)
    # =========================================================================
    for model in MODELS:
        print(f"\nProcessing Model Variant: {model}...")
        
        (MOCK_OUT_DIR / "model_cv").mkdir(parents=True, exist_ok=True)
        (MOCK_OUT_DIR / "model_ppc").mkdir(parents=True, exist_ok=True)
        (MOCK_OUT_DIR / "model_recovery").mkdir(parents=True, exist_ok=True)

        # 1. Parameter Posteriors (Skip if BEST_MODEL, already handled jointly)
        if model != BEST_MODEL:
            real_post_path = REAL_OUT_DIR / f'params_posteriors_{model}.csv'
            if real_post_path.exists():
                df_post_loop = pd.read_csv(real_post_path)
                
                df_wide = df_post_loop.pivot_table(index='subject_id', columns='index', values='mean')
                fake_wide = generate_mvn_samples(df_wide, df_wide.columns, N_SUBJECTS)
                fake_wide.insert(0, 'subject_id', fake_subjects)
                
                fake_long = fake_wide.melt(id_vars='subject_id', var_name='index', value_name='mean')
                fake_long['std'] = df_post_loop['std'].mean() if 'std' in df_post_loop else 0.1
                fake_long['5%'] = fake_long['mean'] * 0.85
                fake_long['50%'] = fake_long['mean']
                fake_long['95%'] = fake_long['mean'] * 1.15
                fake_long['subject_year'] = 'baseline'
                
                fake_long = fake_long[['index', 'mean', 'std', '5%', '50%', '95%', 'subject_id', 'subject_year']]
                
                out_post_loop = MOCK_OUT_DIR / f'example_posteriors_{model}.csv'
                fake_long.to_csv(out_post_loop, index=False)
                print(f"  -> Generated {out_post_loop.name}")
        
        # 2. CV Summary
        real_cv_path = REAL_OUT_DIR / "model_cv" / f'cv_summary_{model}.csv'
        if real_cv_path.exists():
            df_cv = pd.read_csv(real_cv_path)
            df_cv['n_subjects'] = N_SUBJECTS 
            out_cv = MOCK_OUT_DIR / "model_cv" / f'cv_{model}.csv'
            df_cv.to_csv(out_cv, index=False)
            print(f"  -> Generated model_cv/{out_cv.name}")

        # 3. PPC Summary
        real_ppc_sum_path = REAL_OUT_DIR / "model_ppc" / f'ppc_model_summary_{model}.csv'
        if real_ppc_sum_path.exists():
            df_ppc_sum = pd.read_csv(real_ppc_sum_path)
            df_ppc_sum['n_subjects'] = N_SUBJECTS 
            out_ppc_sum = MOCK_OUT_DIR / "model_ppc" / f'ppc_{model}.csv'
            df_ppc_sum.to_csv(out_ppc_sum, index=False)
            print(f"  -> Generated model_ppc/{out_ppc_sum.name}")

        # 4. Parameter Recovery
        real_rec_path = REAL_OUT_DIR / "model_recovery" / f'params_recovery_{model}.csv'
        if real_rec_path.exists():
            df_rec = pd.read_csv(real_rec_path)
            hit_cols = [c for c in df_rec.columns if 'hit' in c]
            num_cols_rec = [c for c in df_rec.columns if c not in hit_cols and c != 'case']
            
            fake_rec = generate_mvn_samples(df_rec, num_cols_rec, N_SUBJECTS)
            fake_rec.insert(0, 'case', range(N_SUBJECTS))
            
            for hcol in hit_cols:
                fake_rec[hcol] = np.random.choice([0.0, 1.0], size=N_SUBJECTS, p=[0.05, 0.95])
            
            out_rec = MOCK_OUT_DIR / "model_recovery" / f'recovery_{model}.csv'
            fake_rec.to_csv(out_rec, index=False)
            print(f"  -> Generated model_recovery/{out_rec.name}")
            
    print("\nAll mock data generated successfully.")

if __name__ == "__main__":
    main()