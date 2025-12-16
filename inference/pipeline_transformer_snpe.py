"""
pipeline_transformer_snpe.py

End-to-end SBI with a Transformer-based summary for the POMDP-SST simulator.

Workflow:
1) Round-1: Simulate pairs of (omega, trials).
2) Pretrain: Train a small Transformer encoder to map trials -> embedding.
3) Freeze Encoder: Generate embeddings and fit a z-score standardizer.
4) Train SNPE: Train Neural Spline Flow (NSF) posterior on (omega, embedding_z).
5) Round-2 (Active Learning): Sample from proposal, simulate, and retrain.
6) Inference: Condition on observed data to sample the posterior.

================================================================================
USAGE EXAMPLES
================================================================================

--- A. SMOKE TEST (Fast Debugging) ---
Use these commands to verify the pipeline works without waiting hours.

# 1. Pretrain Encoder (Quick: 500 sims, 2 epochs)
python pipeline_transformer_snpe.py --stage pretrain --n1_pre 500 --epochs 2

# 2. Train SNPE (Light: MAF density, small N)
python pipeline_transformer_snpe.py --stage snpe --n1_pre 500 --n2 200 --density maf

# 3. Recovery Check (5 test cases)
python pipeline_transformer_snpe.py --stage recover --K 5 --num_post 500


--- B. REAL RUN (Production) ---
Use these commands for final analysis. WARNING: Heavy computation.

# 1. Pretrain Encoder (Deep: 30k sims, 8 epochs)
python pipeline_transformer_snpe.py --stage pretrain --n1_pre 30000 --epochs 8

# 2. Train SNPE (Heavy: NSF density, 50k total sims)
python pipeline_transformer_snpe.py --stage snpe --n1_pre 30000 --n2 20000 --density nsf

# 3. Full Recovery Sweep (200 test cases)
python pipeline_transformer_snpe.py --stage recover --K 200 --num_post 4000

# 4. Inference on Real Data (All subjects)
python pipeline_transformer_snpe.py --stage posterior \
  --sst_folder "/home/wwang/data/sst_valid_raw" \
  --glob_pat "NDAR_*.csv.zip" \
  --num_samples 10000

================================================================================
"""

import os
import math
import random
import pickle
import argparse
import warnings
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sbi.inference import SNPE
from sbi.utils import BoxUniform

# Local modules
try:
    from utils.preprocessing import preprocessing
    from model.pomdp import POMDP
except ImportError:
    # Placeholder for linting/testing without local modules
    def preprocessing(fp): raise NotImplementedError("preprocessing module missing")
    class POMDP: pass 


# --- Configuration ---

# Parameter Ranges
PARAM_RANGES = {
    "q_d_n": (0.0, 1.0),
    "q_d":   (0.5, 1.0),
    "q_s_n": (0.0, 1.0),
    "q_s":   (0.5, 1.0),
    "cost_stop_error": (3e-5, 2.0),
    "cost_time":       (3e-5, 0.003),
}

LINEAR_PARAMS = ["q_d_n", "q_d", "q_s_n", "q_s"]
LOG_PARAMS = ["cost_stop_error", "cost_time"]
PARAM_ORDER = LINEAR_PARAMS + LOG_PARAMS

FIXED_PARAMS = {
    "cost_go_error":   1.0,
    "cost_go_missing": 1.0,
    "inv_temp":        50.0,
}

RESULT_LEVELS = ["GS", "GE", "GM", "SS", "SE"]

# Artifact Paths
ART_DIR = Path("outputs")
ENC_PATH = ART_DIR / "encoder.pt"
STD_MEAN_PATH = ART_DIR / "embeds_mean.npy"
STD_STD_PATH = ART_DIR / "embeds_std.npy"
POST1_PATH = ART_DIR / "posterior_round1.pkl"
POSTF_PATH = ART_DIR / "posterior_final.pkl"
RECOVERY_CSV = ART_DIR / "recovery_details.csv"
POST_SUMMARY_CSV = ART_DIR / "posterior_summary_all.csv"


# --- 1. Simulator Wrapper ---

def simulate_data(params: Dict[str, float], rate_stop_trial=1/6, seed: Optional[int] = None) -> pd.DataFrame:
    """Runs the POMDP simulator for a single parameter set."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    
    pomdp = POMDP(**params, rate_stop_trial=rate_stop_trial)
    pomdp.value_iteration_tensor()
    arr = pomdp.simu_task()
    return pd.DataFrame(arr, columns=["result", "rt", "ssd"])


# --- 2. Parameter Transforms ---

def make_box_prior() -> Tuple[BoxUniform, torch.Tensor, torch.Tensor]:
    """Constructs the SBI BoxUniform prior based on configured ranges."""
    low, high = [], []
    for k in PARAM_ORDER:
        lo, hi = PARAM_RANGES[k]
        if k in LOG_PARAMS:
            lo, hi = math.log(lo), math.log(hi)
        low.append(lo)
        high.append(hi)
    
    low = torch.tensor(low, dtype=torch.float32)
    high = torch.tensor(high, dtype=torch.float32)
    prior = BoxUniform(low=low, high=high)
    return prior, low, high


def untransform(omega_vec: torch.Tensor) -> Dict[str, float]:
    """Converts transformed (log) parameters back to original space."""
    vals = omega_vec.detach().cpu().numpy().astype(float)
    out = {}
    for i, k in enumerate(PARAM_ORDER):
        v = vals[i]
        if k in LOG_PARAMS:
            v = float(np.exp(v))
        out[k] = float(v)
    out.update(FIXED_PARAMS)
    return out


# --- 3. Feature Engineering ---

def build_per_trial_features(df: pd.DataFrame, use_index=True) -> np.ndarray:
    """
    Constructs per-trial feature vectors (N_trials x Features).
    Features: One-hot result, RT, SSD, missing flags, lag-1 history, and trial index.
    """
    N = len(df)
    idx_map = {k: i for i, k in enumerate(RESULT_LEVELS)}

    # Current trial features (One-Hot)
    oh = np.zeros((N, len(RESULT_LEVELS)), dtype=np.float32)
    res = df["result"].astype(str).values
    for t, r in enumerate(res):
        if r in idx_map:
            oh[t, idx_map[r]] = 1.0

    rt = pd.to_numeric(df["rt"], errors="coerce").values.astype(np.float32)
    ssd = pd.to_numeric(df["ssd"], errors="coerce").values.astype(np.float32)
    
    # Missing value indicators & fill
    rt_nan = np.isnan(rt).astype(np.float32)
    ssd_nan = np.isnan(ssd).astype(np.float32)
    rt = np.nan_to_num(rt, nan=0.0)
    ssd = np.nan_to_num(ssd, nan=0.0)

    # Lag-1 features
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


# --- 4. Transformer Encoder ---

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
    """Lightweight Transformer encoder + mean pooling."""
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

    def forward(self, x):
        h = self.proj(x)
        if self.use_pos:
            h = self.pe(h)
        h = self.encoder(h)
        h = h.mean(dim=1)  # Mean pool
        return self.head(h)


# --- 5. Training Utils ---

class OmegaDataset(Dataset):
    def __init__(self, X_list: List[np.ndarray], omega_list: List[np.ndarray]):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X_list]
        self.Y = [torch.tensor(y, dtype=torch.float32) for y in omega_list]

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


def train_encoder_on_simulations(X_list, omega_list, in_dim, omega_dim, 
                                 epochs=5, batch_size=64, lr=1e-3, seed=137):
    """Supervised regression: Trials -> Embedding -> Predicted Omega."""
    torch.manual_seed(seed)
    
    encoder = TrialTransformer(in_dim, out_dim=64, use_pos_enc=True)
    reg_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, omega_dim))

    params = list(encoder.parameters()) + list(reg_head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    # Normalize targets
    Y = np.stack(omega_list, axis=0).astype(np.float32)
    y_mean = Y.mean(axis=0, keepdims=True)
    y_std = Y.std(axis=0, keepdims=True)
    y_std[y_std == 0.0] = 1.0
    omega_scaled = [(Y[i] - y_mean[0]) / y_std[0] for i in range(len(Y))]

    ds = OmegaDataset(X_list, omega_scaled)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    encoder.train(); reg_head.train()
    print(f"[Encoder] Starting pretraining for {epochs} epochs...")
    
    for ep in range(epochs):
        total_loss = 0.0
        for Xb, Yb in dl:
            pred = reg_head(encoder(Xb))
            loss = loss_fn(pred, Yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * Xb.size(0)
        print(f"[Encoder] Epoch {ep+1}/{epochs} MSE={total_loss/len(ds):.4f}")

    # Freeze encoder
    for p in encoder.parameters(): p.requires_grad = False
    encoder.eval()
    return encoder, reg_head


class SummaryStandardizer:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self
    def transform(self, X): return (X - self.mean_) / self.std_
    def fit_transform(self, X): return self.fit(X).transform(X)


# --- 6. Pipeline Stages ---

@dataclass
class SimBlock:
    omegas: torch.Tensor
    embeds: np.ndarray

def simulate_round(prior_sampler, N, encoder, seed_offset=0):
    omegas = prior_sampler((N,))
    embeds = []
    for i in range(N):
        params = untransform(omegas[i])
        df = simulate_data(params, seed=seed_offset + i)
        X = build_per_trial_features(df)
        with torch.no_grad():
            z = encoder(torch.tensor(X[None, ...], dtype=torch.float32))
        embeds.append(z.squeeze(0).numpy())
    return SimBlock(omegas=omegas, embeds=np.vstack(embeds))


def run_pretrain(args, prior, D):
    print(f"[Pretrain] Simulating {args.n1_pre} pairs...")
    omegas = prior.sample((args.n1_pre,))
    X_list, Omega_list = [], []
    
    for i in range(args.n1_pre):
        params = untransform(omegas[i])
        df = simulate_data(params, seed=i)
        X_list.append(build_per_trial_features(df))
        Omega_list.append(omegas[i].numpy())
    
    encoder, _ = train_encoder_on_simulations(
        X_list, Omega_list, in_dim=X_list[0].shape[1], omega_dim=D,
        epochs=args.epochs
    )
    torch.save(encoder.state_dict(), ENC_PATH)
    print(f"[Pretrain] Encoder saved to {ENC_PATH}")


def run_snpe(args, prior, encoder):
    # Round 1
    print(f"[Round1] Simulating {args.n1_pre} pairs...")
    block1 = simulate_round(prior.sample, args.n1_pre, encoder, seed_offset=0)
    
    stdzr = SummaryStandardizer()
    embeds_1_z = stdzr.fit_transform(block1.embeds)
    
    # Save standardizer using numpy
    np.save(STD_MEAN_PATH, stdzr.mean_)
    np.save(STD_STD_PATH, stdzr.std_)

    print(f"[Round1] Training SNPE ({args.density})...")
    inference = SNPE(prior=prior, density_estimator=args.density)
    inference.append_simulations(block1.omegas, torch.tensor(embeds_1_z, dtype=torch.float32))
    density = inference.train()
    posterior_1 = inference.build_posterior(density)
    
    with open(POST1_PATH, "wb") as f: pickle.dump(posterior_1, f)

    # Round 2 (Optional)
    if args.n2 > 0:
        print(f"[Round2] Active Learning with {args.n2} simulations...")
        # Mixture Proposal
        def proposal_sampler(shape):
            n = shape[0]; n_prior = int(args.mix_prior_frac * n); n_post = n - n_prior
            return torch.cat([prior.sample((n_prior,)), posterior_1.sample((n_post,))], dim=0)

        block2 = simulate_round(proposal_sampler, args.n2, encoder, seed_offset=1_000_000)
        embeds_2_z = stdzr.transform(block2.embeds)

        omegas_all = torch.cat([block1.omegas, block2.omegas], 0)
        embeds_all = torch.cat([torch.tensor(embeds_1_z), torch.tensor(embeds_2_z)], 0)

        print("[Merged] Training Final Posterior...")
        inference = SNPE(prior=prior, density_estimator=args.density)
        inference.append_simulations(omegas_all, embeds_all.float())
        density = inference.train()
        posterior_final = inference.build_posterior(density)
    else:
        posterior_final = posterior_1

    with open(POSTF_PATH, "wb") as f: pickle.dump(posterior_final, f)


def run_recovery(args, prior, posterior, encoder, standardizer):
    print(f"[Recovery] Checking {args.K} ground-truth cases...")
    omegas_true = prior.sample((args.K,))
    details = []
    
    for i in range(args.K):
        gt_params = untransform(omegas_true[i])
        df_obs = simulate_data(gt_params, seed=4242 + i)
        
        # Inference
        X = build_per_trial_features(df_obs)
        with torch.no_grad():
            z = encoder(torch.tensor(X[None, ...], dtype=torch.float32)).numpy()
        z_z = standardizer.transform(z)
        samples = posterior.sample((args.num_post,), x=torch.tensor(z_z, dtype=torch.float32))
        
        # Stats
        row = {"case": i}
        samples_np = samples.numpy()
        for k_idx, k in enumerate(PARAM_ORDER):
            # Transform back to original space for stats
            vals = [untransform(torch.tensor(s))[k] for s in samples_np]
            mu, lo, hi = np.mean(vals), np.percentile(vals, 5), np.percentile(vals, 95)
            gt_val = gt_params[k]
            row[f"gt_{k}"] = gt_val
            row[f"mu_{k}"] = mu
            row[f"hit90_{k}"] = 1.0 if lo <= gt_val <= hi else 0.0
        details.append(row)

    pd.DataFrame(details).to_csv(RECOVERY_CSV, index=False)
    print(f"[Recovery] Saved to {RECOVERY_CSV}")


def run_inference(args, posterior, encoder, standardizer):
    out_root = ART_DIR / "subjects"
    out_root.mkdir(parents=True, exist_ok=True)
    
    files = []
    if args.real_csv: files.append(Path(args.real_csv))
    if args.sst_folder: files.extend(Path(args.sst_folder).glob(args.glob_pat))
    files = sorted(files)
    
    print(f"[Inference] Processing {len(files)} files...")
    summaries = []

    for fp in files:
        try:
            # Parse filename (Assumes NDAR_ID_YEAR_...)
            parts = fp.name.split('_')
            sid = parts[1] if len(parts) > 1 else fp.stem
            year = parts[2] if len(parts) > 2 else "unknown"

            df_obs = preprocessing(str(fp))
            X = build_per_trial_features(df_obs)
            with torch.no_grad():
                z = encoder(torch.tensor(X[None, ...], dtype=torch.float32)).numpy()
            z_z = standardizer.transform(z)
            
            samples = posterior.sample((args.num_samples,), x=torch.tensor(z_z, dtype=torch.float32))
            
            # Save Results
            rows = [untransform(s) for s in samples]
            post_df = pd.DataFrame(rows)
            summ = post_df.describe(percentiles=[0.05, 0.5, 0.95]).T[["mean", "std", "5%", "50%", "95%"]]
            
            subj_dir = out_root / f"{sid}_{year}"
            subj_dir.mkdir(exist_ok=True)
            post_df.to_csv(subj_dir / "posterior_samples.csv", index=False)
            summ.to_csv(subj_dir / "posterior_summary.csv")
            
            s_row = summ.copy(); s_row["subject_id"] = sid; s_row["subject_year"] = year
            summaries.append(s_row.reset_index())

        except Exception as e:
            print(f"[Error] {fp.name}: {e}")

    if summaries:
        pd.concat(summaries).to_csv(POST_SUMMARY_CSV, index=False)
        print(f"[Inference] All summaries saved to {POST_SUMMARY_CSV}")


# --- 7. Main ---

def main():
    warnings.filterwarnings("ignore")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    
    # Global Seed
    SEED = 137
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

    parser = argparse.ArgumentParser(description="Transformer-based SNPE Pipeline")
    parser.add_argument("--stage", choices=["all", "pretrain", "snpe", "recover", "posterior"], default="all")
    parser.add_argument("--n1_pre", type=int, default=20000, help="Simulations for pretraining/Round 1")
    parser.add_argument("--epochs", type=int, default=5, help="Pretraining epochs")
    parser.add_argument("--n2", type=int, default=15000, help="Round 2 simulations")
    parser.add_argument("--mix_prior_frac", type=float, default=0.2, help="Prior fraction in proposal")
    parser.add_argument("--density", choices=["nsf", "maf", "mdn"], default="nsf", help="Density estimator")
    
    # Recovery/Inference args
    parser.add_argument("--K", type=int, default=20, help="Recovery test cases")
    parser.add_argument("--num_post", type=int, default=1000, help="Samples per recovery posterior")
    parser.add_argument("--real_csv", type=str, default=None, help="Single real CSV")
    parser.add_argument("--sst_folder", type=str, default=None, help="Folder of real CSVs")
    parser.add_argument("--glob_pat", type=str, default="NDAR_*.csv*", help="File pattern")
    parser.add_argument("--num_samples", type=int, default=20000, help="Posterior samples per subject")

    args = parser.parse_args()
    ART_DIR.mkdir(exist_ok=True)

    prior, _, _ = make_box_prior()
    D = len(PARAM_ORDER)

    # 1. Pretrain
    if args.stage in ["all", "pretrain"]:
        run_pretrain(args, prior, D)
        if args.stage == "pretrain": return

    # Load Encoder
    # Note: Logic to determine input dim for loading is simplified here for brevity
    # In production, we'd save/load config args.
    dummy_df = simulate_data(untransform(prior.sample((1,))[0]), seed=0)
    in_dim = build_per_trial_features(dummy_df).shape[1]
    
    encoder = TrialTransformer(in_dim, out_dim=64, use_pos_enc=True)
    encoder.load_state_dict(torch.load(ENC_PATH))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False

    # 2. SNPE
    if args.stage in ["all", "snpe"]:
        run_snpe(args, prior, encoder)
    
    # Load Posterior & Standardizer for next steps
    if args.stage in ["recover", "posterior"] or args.stage == "all":
        with open(POSTF_PATH, "rb") as f: posterior = pickle.load(f)
        stdzr = SummaryStandardizer()
        stdzr.mean_ = np.load(STD_MEAN_PATH)
        stdzr.std_ = np.load(STD_STD_PATH)

    # 3. Recovery
    if args.stage in ["all", "recover"]:
        run_recovery(args, prior, posterior, encoder, stdzr)

    # 4. Inference
    if args.stage in ["all", "posterior"]:
        run_inference(args, posterior, encoder, stdzr)

if __name__ == "__main__":
    main()