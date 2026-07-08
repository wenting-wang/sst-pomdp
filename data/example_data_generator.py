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

MODELS = ['10p_v1', '8p_v2', '7p_v3', '6p_v1']
BEST_MODEL = '6p_v1'

N_SUBJECTS = 110

# =============================================================================
# 2. Helper Functions
# =============================================================================
def generate_mvn_samples(df, num_cols, n_samples):
    """Generate multivariate normal mock data based on real covariance."""
    means = df[num_cols].mean()
    cov = df[num_cols].cov()
    # Add a tiny diagonal perturbation to ensure the matrix is positive semi-definite
    cov += np.eye(len(num_cols)) * 1e-6 
    samples = np.random.multivariate_normal(means, cov, size=n_samples)
    return pd.DataFrame(samples, columns=num_cols).clip(lower=1e-6)

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
    
    # ---------------------------------------------------------
    # A. Generate Base Data (Saved to data/ folder)
    # ---------------------------------------------------------
    # 1. Clinical Behavior Data
    real_clin_path = REAL_DATA_DIR / 'clinical_behavior.csv'
    if real_clin_path.exists():
        df_clin = pd.read_csv(real_clin_path)
        num_cols = df_clin.select_dtypes(include=[np.number]).columns.drop('adhd', errors='ignore')
        
        fake_clin = generate_mvn_samples(df_clin, num_cols, N_SUBJECTS)
        fake_clin.insert(0, 'subject_id', fake_subjects)
        fake_clin['year'] = 'baseline'
        fake_clin['sex'] = np.random.choice(['F', 'M'], N_SUBJECTS)
        fake_clin['adhd'] = np.random.poisson(lam=2.5, size=N_SUBJECTS)
        fake_clin['adhd_med_flag'] = np.random.choice([0, 1], N_SUBJECTS, p=[0.9, 0.1])
        
        out_clin = MOCK_DATA_DIR / 'example_clinical_behavior.csv'
        fake_clin.to_csv(out_clin, index=False)
        print(f"-> Generated {out_clin.name}")

    # 2. Embeddings (Best Model Only)
    real_emb_path = REAL_OUT_DIR / f'embeddings_e2e_tesbi_e2e_{BEST_MODEL}.csv'
    if real_emb_path.exists():
        df_emb = pd.read_csv(real_emb_path)
        emb_cols = [c for c in df_emb.columns if c.startswith('emb_')]
        fake_emb = generate_mvn_samples(df_emb, emb_cols, N_SUBJECTS)
        fake_emb.insert(0, 'subject_id', fake_subjects)
        fake_emb['year'] = 'baseline'
        
        out_emb = MOCK_OUT_DIR / f'example_embeddings_{BEST_MODEL}.csv'
        fake_emb.to_csv(out_emb, index=False)
        print(f"-> Generated {out_emb.name}")
        
    # 3. PPC Metrics (Best Model Only)
    real_ppc_metrics_path = REAL_OUT_DIR / f'ppc_metrics_{BEST_MODEL}.csv'
    if real_ppc_metrics_path.exists():
        df_ppc = pd.read_csv(real_ppc_metrics_path)
        num_cols = df_ppc.select_dtypes(include=[np.number]).columns
        fake_ppc = generate_mvn_samples(df_ppc, num_cols, N_SUBJECTS)
        fake_ppc.insert(0, 'subject_id', fake_subjects)
        fake_ppc['year'] = 'baseline'
        
        out_ppc = MOCK_OUT_DIR / f'example_ppc_metrics_{BEST_MODEL}.csv'
        fake_ppc.to_csv(out_ppc, index=False)
        print(f"-> Generated {out_ppc.name}")

    # ---------------------------------------------------------
    # B. Generate Multi-Model Data (Saved to outputs_example/tesbi/)
    # ---------------------------------------------------------
    for model in MODELS:
        print(f"\nProcessing Model Variant: {model}...")
        
        (MOCK_OUT_DIR / "model_cv").mkdir(parents=True, exist_ok=True)
        (MOCK_OUT_DIR / "model_ppc").mkdir(parents=True, exist_ok=True)
        (MOCK_OUT_DIR / "model_recovery").mkdir(parents=True, exist_ok=True)

        # 1. Parameter Posteriors
        real_post_path = REAL_OUT_DIR / f'params_posteriors_{model}.csv'
        if real_post_path.exists():
            df_post = pd.read_csv(real_post_path)
            
            # Pivot to wide format for MVN sampling
            df_wide = df_post.pivot_table(index='subject_id', columns='index', values='mean')
            fake_wide = generate_mvn_samples(df_wide, df_wide.columns, N_SUBJECTS)
            fake_wide.insert(0, 'subject_id', fake_subjects)
            
            # Melt back to long format
            fake_long = fake_wide.melt(id_vars='subject_id', var_name='index', value_name='mean')
            fake_long['std'] = df_post['std'].mean() if 'std' in df_post else 0.1
            fake_long['5%'] = fake_long['mean'] * 0.85
            fake_long['50%'] = fake_long['mean']
            fake_long['95%'] = fake_long['mean'] * 1.15
            fake_long['subject_year'] = 'baseline'
            
            # Reorder columns to align with original
            fake_long = fake_long[['index', 'mean', 'std', '5%', '50%', '95%', 'subject_id', 'subject_year']]
            
            out_post = MOCK_OUT_DIR / f'example_posteriors_{model}.csv'
            fake_long.to_csv(out_post, index=False)
            print(f"  -> Generated {out_post.name}")
            
            # Generate trial data directly from fake_wide if it's the best model
            if model == BEST_MODEL:
                print(f"  -> Generating trial-by-trial data using parameters from {model}...")
                
                for _, row in fake_wide.iterrows():
                    subject = row['subject_id']
                    try:
                        # Direct extraction from fake_wide and convert values to native float
                        subj_params = {k: float(v) for k, v in row.drop('subject_id').to_dict().items()}
                        
                        # Simulate and save trial data
                        df_trial = simulate_data(subj_params)
                        trial_out_path = PROCESSED_DATA_DIR / f"{subject}.csv"
                        df_trial.to_csv(trial_out_path, index=False)
                        
                    except Exception as e:
                        print(f"     [Warning] Skipped {subject} due to simulation error: {repr(e)}")
                        
                print(f"  -> Finished generating trial data in {PROCESSED_DATA_DIR.name}/")
        
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
            num_cols = [c for c in df_rec.columns if c not in hit_cols and c != 'case']
            
            fake_rec = generate_mvn_samples(df_rec, num_cols, N_SUBJECTS)
            fake_rec.insert(0, 'case', range(N_SUBJECTS))
            
            for hcol in hit_cols:
                fake_rec[hcol] = np.random.choice([0.0, 1.0], size=N_SUBJECTS, p=[0.05, 0.95])
            
            out_rec = MOCK_OUT_DIR / "model_recovery" / f'recovery_{model}.csv'
            fake_rec.to_csv(out_rec, index=False)
            print(f"  -> Generated model_recovery/{out_rec.name}")
            
    print("\nAll mock data generated successfully.")

if __name__ == "__main__":
    main()