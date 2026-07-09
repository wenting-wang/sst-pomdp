"""
embedding.py 

Generates embeddings for individual subjects using the trained End-to-End TeSBI model.
"""

import os
import argparse
import sys
import warnings
import importlib
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# =============================================================================
# 0. ARGUMENT PARSING (DEMO vs PROD MODE)
# =============================================================================
parser = argparse.ArgumentParser(description="Generate Embeddings using End-to-End Model")
parser.add_argument(
    '--prod', 
    action='store_true', 
    help='Run in PROD mode using local raw data. Defaults to DEMO mode if omitted.'
)
args, unknown = parser.parse_known_args()

DEMO_MODE = not args.prod

# =============================================================================
# 1. CONFIGURATION & PATHS
# =============================================================================
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
sys.path.append(str(REPO_ROOT))

# Ensure utils can be imported
try:
    from utils.preprocessing import preprocessing
except ImportError as e:
    print(f"Import error: {e}. Please ensure you are running from the correct directory.")
    sys.exit(1)

if DEMO_MODE:
    print(">>> RUNNING IN DEMO MODE (Using example data) <<<")
    DATA_DIR = REPO_ROOT / "data" / "example_processed_data"
    META_CSV = REPO_ROOT / "data" / "example_clinical_behavior.csv"
    MODEL_MODULE = "outputs_example.tesbi.model_spec.tesbi_e2e_6p_v1"
    MODEL_PATH = REPO_ROOT / "outputs_example" / "tesbi" / "amortized_inference_net_6p_v1.pth"
    OUT_CSV = REPO_ROOT / "outputs_example" / "tesbi" / "example_embeddings_6p_v1.csv"
    USE_PREPROCESSING = False
else:
    print(">>> RUNNING IN PROD MODE (Using local raw data) <<<")
    DATA_DIR = Path("/Users/w/Desktop/data/sst_valid_base")
    META_CSV = Path("/Users/w/sst-pomdp/data/clinical_behavior.csv")
    MODEL_MODULE = "tesbi_e2e_6p_v1" 
    MODEL_PATH = Path("/Users/w/sst-pomdp/outputs/tesbi/amortized_inference_net_6p_v1.pth")
    OUT_CSV = Path("/Users/w/sst-pomdp/outputs/tesbi/model_posteriors/embeddings_e2e_tesbi_e2e_6p_v1.csv")
    USE_PREPROCESSING = True

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
RESULT_LEVELS = ["GS", "GE", "GM", "SS", "SE"]

# ==============================================================================
# EMBEDDING EXTRACTOR (The Wrapper)
# ==============================================================================
class EmbeddingExtractor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x_seq):
        h = self.base_model.proj(x_seq)
        h = self.base_model.pe(h)
        h = self.base_model.encoder(h) 
        h_std_latents = h.std(dim=1)   
        h_mean = h.mean(dim=1)
        context = self.base_model.context_head(h_mean) 
        
        return context, h_std_latents

# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
def build_per_trial_features(df: pd.DataFrame, use_index=True) -> np.ndarray:
    N = len(df)
    idx_map = {k: i for i, k in enumerate(RESULT_LEVELS)}

    oh = np.zeros((N, len(RESULT_LEVELS)), dtype=np.float32)
    res = df["result"].astype(str).values
    for t, r in enumerate(res):
        if r in idx_map:
            oh[t, idx_map[r]] = 1.0

    rt = pd.to_numeric(df["rt"], errors="coerce").values.astype(np.float32)
    ssd = pd.to_numeric(df["ssd"], errors="coerce").values.astype(np.float32)
    
    rt_nan = np.isnan(rt).astype(np.float32)
    ssd_nan = np.isnan(ssd).astype(np.float32)
    
    rt = np.nan_to_num(rt, nan=0.0)
    ssd = np.nan_to_num(ssd, nan=0.0)

    prev_oh = np.roll(oh, 1, axis=0); prev_oh[0, :] = 0.0
    prev_rt = np.roll(rt, 1); prev_rt[0] = 0.0
    prev_ssd = np.roll(ssd, 1); prev_ssd[0] = 0.0
    delta_ssd = ssd - prev_ssd

    cols = [
        oh, rt[:, None], ssd[:, None], rt_nan[:, None], ssd_nan[:, None],
        prev_oh, prev_rt[:, None], prev_ssd[:, None], delta_ssd[:, None]
    ]
    
    if use_index:
        t_idx = np.linspace(-1.0, 1.0, N, dtype=np.float32)[:, None]
        cols.append(t_idx)

    return np.concatenate(cols, axis=1).astype(np.float32)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    warnings.filterwarnings("ignore")
    
    # --- 1. Dynamic Module Import ---
    print(f"[Init] Loading model specification from: {MODEL_MODULE}")
    m = importlib.import_module(MODEL_MODULE)
    EndToEndTeSBI = m.EndToEndTeSBI
    PARAM_ORDER = m.PARAM_ORDER

    # --- 2. Load Subject List ---
    if not META_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {META_CSV}")
    
    df_meta = pd.read_csv(META_CSV)
    target_files = set(df_meta["filename"].dropna().unique())
    print(f"Found {len(target_files)} target files. Preprocessing Enabled: {USE_PREPROCESSING}")

    # --- 3. Initialize Model & Device ---
    device = torch.device("cpu")
    extractor = None 
    results = []
    
    count_ok, count_miss, count_err = 0, 0, 0

    for fname in target_files:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            count_miss += 1; continue

        try:
            df_subj = preprocessing(str(fpath)) if USE_PREPROCESSING else pd.read_csv(fpath)
            X = build_per_trial_features(df_subj)
            
            if extractor is None:
                in_dim = X.shape[1]
                param_dim = len(PARAM_ORDER) 
                
                base_model = EndToEndTeSBI(in_dim=in_dim, param_dim=param_dim)
                
                if not MODEL_PATH.exists():
                     raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")
                
                base_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
                base_model.eval()
                
                extractor = EmbeddingExtractor(base_model).to(device)
                print(f"Extractor ready. Loaded E2E model from {MODEL_PATH.name}")

            X_t = torch.tensor(X[None, ...], dtype=torch.float32).to(device) 
            
            with torch.no_grad():
                z_mean, h_std = extractor(X_t)
                z_vec = z_mean.squeeze(0).cpu().numpy()      
                h_std_vec = h_std.squeeze(0).cpu().numpy()   
            
            meta_row = df_meta[df_meta["filename"] == fname].iloc[0]
            sid = meta_row.get("subject_id", "unknown")
            year = meta_row.get("year", "unknown")

            row = {
                "subject_id": sid,
                "year": year,
                "filename": fname,
                "embedding_vec_mean": float(np.mean(z_vec)),
                "embedding_vec_std": float(np.std(z_vec)),
                "trial_latents_mean_std": float(np.mean(h_std_vec)) 
            }
            
            for i, val in enumerate(z_vec):
                row[f"emb_{i}"] = val
            
            results.append(row)
            count_ok += 1
            
            if count_ok % 100 == 0:
                print(f"Processed {count_ok} files...")

        except Exception as e:
            print(f"[Error] {fname}: {e}")
            count_err += 1

    # --- 4. Save Results ---
    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(OUT_CSV, index=False)
        print(f"\nDone. Saved 128-dim embeddings for {len(df_out)} subjects to {OUT_CSV.name}")
        print(f"Stats: OK={count_ok}, Missing={count_miss}, Error={count_err}")
    else:
        print("\nWarning: No results generated.")

if __name__ == "__main__":
    main()