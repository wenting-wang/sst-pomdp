"""
embedding.py 
(run on HPC cluster using run_embedding.sh)

Generates embeddings for individual subjects using a trained TrialTransformer encoder.
Calculates the embedding vector (mean-pooled representation) and the standard deviation 
of the trial-level latents for each subject.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import warnings

# Try to import preprocessing, identical to pipeline
try:
    from preprocessing import preprocessing
except ImportError:
    print("[Warning] 'preprocessing' module not found. Please ensure preprocessing.py is in the directory.")
    # Dummy placeholder if needed, but likely will fail at runtime if real data is needed
    def preprocessing(fp): raise NotImplementedError("preprocessing module missing")

# -----------------------------
# Configuration & Constants
# -----------------------------
RESULT_LEVELS = ["GS", "GE", "GM", "SS", "SE"]
ART_DIR = "artifacts_sbi_txf"
ENC_PATH = os.path.join(ART_DIR, "encoder.pt")

# -----------------------------
# Model Definitions (Copied from pipeline)
# -----------------------------
class SinusoidalPE(nn.Module):
    """Sinusoidal Positional Encoding."""
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        N = x.size(1)
        return x + self.pe[:N].unsqueeze(0)

class TrialTransformer(nn.Module):
    """
    A lightweight Transformer encoder + mean pooling to generate simulation embeddings.
    """
    def __init__(self, in_dim: int, model_dim: int = 64, nhead: int = 4, nlayers: int = 2,
                 dropout: float = 0.1, out_dim: int = 32, use_pos_enc: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_dim, model_dim)
        self.use_pos = use_pos_enc
        if use_pos_enc:
            self.pe = SinusoidalPE(model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=model_dim*4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, out_dim)
        )

    def forward_with_stats(self, x):
        """
        Custom forward pass to return both Mean (Embedding) and Std of trial latents.
        """
        # x shape: (Batch, Trials, Features)
        h = self.proj(x)
        if self.use_pos:
            h = self.pe(h)
        h = self.encoder(h) # (Batch, Trials, ModelDim)
        
        # 1. Mean Pooling (Standard Embedding)
        h_mean = h.mean(dim=1)  
        z_mean = self.head(h_mean) # (Batch, OutDim)
        
        # 2. Std Pooling (Extra Statistic)
        # We calculate std of the latents across trials
        h_std_latents = h.std(dim=1) # (Batch, ModelDim)
        # We cannot pass h_std through self.head because self.head includes LayerNorm
        # which expects the distribution of the inputs to match training. 
        # However, for "std" stats, we can just return the raw latent std or average it.
        # To provide a scalar/vector summary, we'll return the mean of the std vector
        # or the raw vector. Let's return the vector of stds.
        
        return z_mean, h_std_latents

    def forward(self, x):
        z, _ = self.forward_with_stats(x)
        return z

# -----------------------------
# Feature Engineering
# -----------------------------
def build_per_trial_features(df: pd.DataFrame, use_index=True) -> np.ndarray:
    """
    Constructs per-trial feature vectors (N_trials x Features).
    """
    N = len(df)
    idx_map = {k: i for i, k in enumerate(RESULT_LEVELS)}

    # Current trial features
    oh = np.zeros((N, len(RESULT_LEVELS)), dtype=np.float32)
    res = df["result"].astype(str).values
    for t, r in enumerate(res):
        if r in idx_map:
            oh[t, idx_map[r]] = 1.0

    rt = pd.to_numeric(df["rt"], errors="coerce").values.astype(np.float32)
    ssd = pd.to_numeric(df["ssd"], errors="coerce").values.astype(np.float32)
    
    # Missing value indicators
    rt_nan = np.isnan(rt).astype(np.float32)
    ssd_nan = np.isnan(ssd).astype(np.float32)
    
    # Fill NaNs
    rt = np.nan_to_num(rt, nan=0.0)
    ssd = np.nan_to_num(ssd, nan=0.0)

    # Lag-1 features
    prev_oh = np.roll(oh, 1, axis=0)
    prev_oh[0, :] = 0.0
    prev_rt = np.roll(rt, 1)
    prev_rt[0] = 0.0
    prev_ssd = np.roll(ssd, 1)
    prev_ssd[0] = 0.0
    delta_ssd = ssd - prev_ssd

    cols = [
        oh, rt[:, None], ssd[:, None], rt_nan[:, None], ssd_nan[:, None],
        prev_oh, prev_rt[:, None], prev_ssd[:, None], delta_ssd[:, None]
    ]
    
    if use_index:
        t_idx = np.linspace(-1.0, 1.0, N, dtype=np.float32)[:, None]
        cols.append(t_idx)

    X = np.concatenate(cols, axis=1).astype(np.float32)  # Shape: (N, F)
    return X

# -----------------------------
# Main Execution
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate Embeddings for Valid Subjects")
    parser.add_argument("--csv", type=str, default="stats_all_cbcl_demo.csv", help="Input CSV with filenames")
    parser.add_argument("--folder", type=str, default="/home/wwang/data/all_valid", help="Folder with data files")
    parser.add_argument("--model_path", type=str, default=ENC_PATH, help="Path to trained encoder.pt")
    parser.add_argument("--out", type=str, default="embeddings_output.csv", help="Output CSV filename")
    args = parser.parse_args()

    # 1. Load Subject List
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    
    df_meta = pd.read_csv(args.csv)
    if "filename" not in df_meta.columns:
        raise ValueError(f"Column 'filename' missing from {args.csv}")
    
    target_files = set(df_meta["filename"].dropna().unique())
    print(f"Found {len(target_files)} unique target filenames in CSV.")

    # 2. Initialize Model
    # Note: We need to know 'in_dim' to init model. We'll verify it from the first file.
    device = torch.device("cpu") # Use CPU for inference on list
    encoder = None 

    results = []
    
    print(f"Processing files from {args.folder}...")
    
    count_ok = 0
    count_miss = 0
    count_err = 0

    for fname in target_files:
        fpath = os.path.join(args.folder, fname)
        
        # Check existence
        if not os.path.exists(fpath):
            count_miss += 1
            continue

        try:
            # Load and Preprocess
            df_subj = preprocessing(fpath)
            X = build_per_trial_features(df_subj)
            
            # Init model once we know input dim
            if encoder is None:
                in_dim = X.shape[1]
                # Default params from pipeline
                encoder = TrialTransformer(in_dim, model_dim=64, nhead=4, nlayers=2, out_dim=64) # Assuming out_dim=64 based on pipeline
                
                # Load weights
                if not os.path.exists(args.model_path):
                     raise FileNotFoundError(f"Model not found at {args.model_path}")
                
                state_dict = torch.load(args.model_path, map_location=device)
                encoder.load_state_dict(state_dict)
                encoder.to(device)
                encoder.eval()
                print(f"Model loaded from {args.model_path} (Input Dim: {in_dim})")

            # Inference
            X_t = torch.tensor(X[None, ...], dtype=torch.float32).to(device) # (1, T, F)
            
            with torch.no_grad():
                # Get Embedding (Mean) and Latent Std
                z_mean, h_std = encoder.forward_with_stats(X_t)
                
                z_vec = z_mean.squeeze(0).cpu().numpy()      # The primary embedding
                h_std_vec = h_std.squeeze(0).cpu().numpy()   # The std of trials
            
            # Prepare Row
            # Identify subject_id/year from meta df for convenience
            meta_row = df_meta[df_meta["filename"] == fname].iloc[0]
            sid = meta_row.get("subject_id", "unknown")
            year = meta_row.get("year", "unknown")

            row = {
                "subject_id": sid,
                "year": year,
                "filename": fname,
                # Scalar summaries of the embedding vector itself
                "embedding_vec_mean": float(np.mean(z_vec)),
                "embedding_vec_std": float(np.std(z_vec)),
                # Scalar summary of trial variability
                "trial_latents_mean_std": float(np.mean(h_std_vec)) 
            }
            
            # Save full embedding vector elements
            for i, val in enumerate(z_vec):
                row[f"emb_{i}"] = val
            
            results.append(row)
            count_ok += 1
            
            if count_ok % 100 == 0:
                print(f"Processed {count_ok} files...")

        except Exception as e:
            print(f"[Error] {fname}: {e}")
            count_err += 1

    # 3. Save
    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(args.out, index=False)
        print(f"Done. Saved embeddings for {len(df_out)} subjects to {args.out}")
        print(f"Stats: OK={count_ok}, Missing={count_miss}, Error={count_err}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()