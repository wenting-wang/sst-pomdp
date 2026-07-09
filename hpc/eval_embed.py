"""
embedding.py 

Generates embeddings for individual subjects using the trained End-to-End TeSBI model.
This script extracts the 128-dimensional embedding (context) and the trial-to-trial 
variability (std of latents) without running the final Normalizing Flow (Zuko) part.

================================================================================
USAGE & CONFIGURATION
================================================================================
Ensure this script is in the same directory level as `tesbi_e2e_6p_v1.py` so it 
can import the model architecture correctly.
"""

import os
import argparse
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
RESULT_LEVELS = ["GS", "GE", "GM", "SS", "SE"]

# --- OPTION A: HPC Production Configuration (Recommended) ---
# from tesbi_e2e_6p_v1 import EndToEndTeSBI, PARAM_ORDER 
# from utils.preprocessing import preprocessing

# BASE_DIR = Path('/u/wenwang/data/')
# OUT_DIR = Path('/u/wenwang/outputs/')
# DEFAULT_DATA_DIR = BASE_DIR / "sst_valid_base"
# DEFAULT_META_CSV = BASE_DIR / "clinical_behavior.csv"
# # DEFAULT_MODEL_PATH = OUT_DIR / "amortized_inference_net_6p_v1.pth"
# DEFAULT_MODEL_PATH="/u/wenwang/my_ptmp/10331248/outputs/amortized_inference_net_6p_v1.pth"
# DEFAULT_OUT_CSV = OUT_DIR / "embeddings_e2e_6p_v1.csv"
# USE_PREPROCESSING = True  


# --- OPTION B: Local Testing Configuration ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from outputs_example.tesbi.model_spec.tesbi_e2e_6p_v1 import EndToEndTeSBI, PARAM_ORDER 
from utils.preprocessing import preprocessing

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "example_processed_data"
DEFAULT_META_CSV = PROJECT_ROOT / "data" / "example_clinical_behavior.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs_example" / "tesbi" / "amortized_inference_net_6p_v1.pth"
DEFAULT_OUT_CSV = PROJECT_ROOT / "outputs_example" / "tesbi" / "example_embeddings_6p_v1.csv"
USE_PREPROCESSING = False 


# ==============================================================================
# EMBEDDING EXTRACTOR (The Wrapper)
# ==============================================================================
class EmbeddingExtractor(nn.Module):
    """
    Wraps the trained EndToEndTeSBI model to intercept the forward pass 
    right before the Zuko flow. It extracts:
    1. context (z_mean): The 128-dim subject-level embedding.
    2. h_std: The trial-to-trial variability across the latent dimension.
    """
    def __init__(self, base_model: EndToEndTeSBI):
        super().__init__()
        self.base_model = base_model

    def forward(self, x_seq):
        # 1. Linear projection and positional encoding
        h = self.base_model.proj(x_seq)
        h = self.base_model.pe(h)
        
        # 2. Transformer encoder (Extract sequence features)
        h = self.base_model.encoder(h) # Shape: (Batch, Trials, 128)
        
        # 3. Extract trial-to-trial variability (Trial-level Std)
        h_std_latents = h.std(dim=1)   # Shape: (Batch, 128)
        
        # 4. Temporal mean pooling and Context Head (Subject-level Embedding)
        h_mean = h.mean(dim=1)
        context = self.base_model.context_head(h_mean) # Shape: (Batch, 128)
        
        return context, h_std_latents


# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
def build_per_trial_features(df: pd.DataFrame, use_index=True) -> np.ndarray:
    """Constructs per-trial feature vectors (N_trials x Features)."""
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
    
    parser = argparse.ArgumentParser(description="Generate Embeddings using End-to-End Model")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_META_CSV), help="Input CSV with metadata")
    parser.add_argument("--folder", type=str, default=str(DEFAULT_DATA_DIR), help="Folder containing subject data")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to trained .pth")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_CSV), help="Output CSV filename")
    args = parser.parse_args()

    # --- 1. Load Subject List ---
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
    
    df_meta = pd.read_csv(csv_path)
    target_files = set(df_meta["filename"].dropna().unique())
    print(f"Found {len(target_files)} target files. Preprocessing Enabled: {USE_PREPROCESSING}")

    # --- 2. Initialize Model & Device ---
    device = torch.device("cpu")
    extractor = None 
    results = []
    
    data_folder = Path(args.folder)
    count_ok, count_miss, count_err = 0, 0, 0

    for fname in target_files:
        fpath = data_folder / fname
        if not fpath.exists():
            count_miss += 1; continue

        try:
            # Preprocess data
            df_subj = preprocessing(str(fpath)) if USE_PREPROCESSING else pd.read_csv(fpath)
            X = build_per_trial_features(df_subj)
            
            # Dynamically initialize the model (first iteration only)
            if extractor is None:
                in_dim = X.shape[1]
                param_dim = len(PARAM_ORDER) # Dynamically get parameter dimension
                
                # Initialize the full base model
                base_model = EndToEndTeSBI(in_dim=in_dim, param_dim=param_dim)
                model_file = Path(args.model_path)
                
                if not model_file.exists():
                     raise FileNotFoundError(f"Model weights not found at {model_file}")
                
                # Load weights
                base_model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
                base_model.eval()
                
                # Wrap it as a feature extractor
                extractor = EmbeddingExtractor(base_model).to(device)
                print(f"Extractor ready. Loaded E2E model from {model_file.name}")

            # --- Core Inference Step ---
            X_t = torch.tensor(X[None, ...], dtype=torch.float32).to(device) # (1, T, F)
            
            with torch.no_grad():
                z_mean, h_std = extractor(X_t)
                z_vec = z_mean.squeeze(0).cpu().numpy()      # 128-dim Context Embedding
                h_std_vec = h_std.squeeze(0).cpu().numpy()   # 128-dim Latent Std
            
            # --- Compile Results ---
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
            
            # Save the 128-dimensional features
            for i, val in enumerate(z_vec):
                row[f"emb_{i}"] = val
            
            results.append(row)
            count_ok += 1
            
            if count_ok % 100 == 0:
                print(f"Processed {count_ok} files...")

        except Exception as e:
            print(f"[Error] {fname}: {e}")
            count_err += 1

    # --- 3. Save Results ---
    if results:
        df_out = pd.DataFrame(results)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        print(f"\nDone. Saved 128-dim embeddings for {len(df_out)} subjects to {out_path.name}")
        print(f"Stats: OK={count_ok}, Missing={count_miss}, Error={count_err}")
    else:
        print("\nWarning: No results generated.")

if __name__ == "__main__":
    main()