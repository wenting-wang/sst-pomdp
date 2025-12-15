"""
pipeline_transformer_snpe.py

End-to-end SBI with a Transformer-based summary for the POMDP-SST simulator.

Workflow:
1) Round-1: Simulate pairs of (omega, trials).
2) Pretrain: Train a small Transformer encoder to map trials -> embedding,
   using supervised regression to predict transformed-omega (MSE).
3) Freeze Encoder: Generate embeddings for Round-1 simulations and fit a z-score standardizer.
4) Train SNPE: Train Neural Spline Flow (NSF) posterior on (omega, embedding_z).
5) Round-2 (Optional): Sample from proposal (mixture of prior & posterior_1), simulate,
   encode, standardize, merge with Round-1, and retrain final posterior.
6) Inference: Condition on observed data to sample the posterior.

Requirements:
  pip install sbi torch numpy pandas
Assumes:
  from pomdp import POMDP
  simulate_data(params) returns a DataFrame with columns ["result","rt","ssd"].
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sbi.inference import SNPE
from sbi.utils import BoxUniform

# Assumes a local preprocessing module exists
try:
    from preprocessing import preprocessing
except ImportError:
    # Fallback or placeholder if module is missing during linting
    def preprocessing(fp): raise NotImplementedError("preprocessing module missing")

# -----------------------------
# 0) Model-specific Configuration
# -----------------------------

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
ART_DIR = "artifacts_sbi_txf"
ENC_PATH = os.path.join(ART_DIR, "encoder.pt")
STD_MEAN_PATH = os.path.join(ART_DIR, "embeds_mean.npy")
STD_STD_PATH = os.path.join(ART_DIR, "embeds_std.npy")
POST1_PATH = os.path.join(ART_DIR, "posterior_round1.pkl")
POSTF_PATH = os.path.join(ART_DIR, "posterior_final.pkl")


# -----------------------------
# 1) Simulator Wrapper
# -----------------------------

def simulate_data(params: Dict[str, float], rate_stop_trial=1/6, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Runs the POMDP simulator for a single parameter set.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
    
    from pomdp import POMDP  # Local import to keep file standalone
    
    pomdp = POMDP(**params, rate_stop_trial=rate_stop_trial)
    pomdp.value_iteration_tensor()
    arr = pomdp.simu_task()
    return pd.DataFrame(arr, columns=["result", "rt", "ssd"])


# -----------------------------
# 2) Parameter Transforms (Omega)
# -----------------------------

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
    """
    Converts a tensor in the transformed (log) space back to the original dictionary parameter space.
    """
    vals = omega_vec.detach().cpu().numpy().astype(float)
    out = {}
    for i, k in enumerate(PARAM_ORDER):
        v = vals[i]
        if k in LOG_PARAMS:
            v = float(np.exp(v))
        out[k] = float(v)
    out.update(FIXED_PARAMS)
    return out


# -----------------------------
# 3) Feature Engineering
# -----------------------------

def build_per_trial_features(df: pd.DataFrame, use_index=True) -> np.ndarray:
    """
    Constructs per-trial feature vectors (N_trials x Features).
    Features: One-hot result, RT, SSD, missing flags, lag-1 history, and trial index.
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
# 4) Transformer Encoder
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

    def forward(self, x):
        # x shape: (Batch, Trials, Features)
        h = self.proj(x)
        if self.use_pos:
            h = self.pe(h)
        h = self.encoder(h)
        h = h.mean(dim=1)  # Mean pool over trials
        z = self.head(h)   # Embedding
        return z


# -----------------------------
# 5) Encoder Training (Supervised)
# -----------------------------

class OmegaDataset(Dataset):
    def __init__(self, X_list: List[np.ndarray], omega_list: List[np.ndarray]):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X_list]
        self.Y = [torch.tensor(y, dtype=torch.float32) for y in omega_list]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def train_encoder_on_simulations(X_list: List[np.ndarray],
                                 omega_list: List[np.ndarray],
                                 in_dim: int,
                                 omega_dim: int,
                                 model_dim: int = 64,
                                 out_dim: int = 64,
                                 nhead: int = 4,
                                 nlayers: int = 2,
                                 dropout: float = 0.1,
                                 batch_size: int = 64,
                                 epochs: int = 5,
                                 lr: float = 1e-3,
                                 seed: int = 137) -> Tuple[TrialTransformer, nn.Module]:
    """
    Trains the encoder using supervised regression: Trials -> Embedding -> Predicted Omega.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    encoder = TrialTransformer(in_dim, model_dim=model_dim, nhead=nhead,
                               nlayers=nlayers, dropout=dropout, out_dim=out_dim,
                               use_pos_enc=True)
    
    # Auxiliary regression head for training
    reg_head = nn.Sequential(
        nn.Linear(out_dim, out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, omega_dim)
    )

    params = list(encoder.parameters()) + list(reg_head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    # Normalize targets for stable MSE training
    Y = np.stack(omega_list, axis=0).astype(np.float32)
    y_mean = Y.mean(axis=0, keepdims=True)
    y_std = Y.std(axis=0, keepdims=True)
    y_std[y_std == 0.0] = 1.0
    Y_scaled = (Y - y_mean) / y_std
    omega_list_scaled = [Y_scaled[i] for i in range(Y_scaled.shape[0])]

    ds = OmegaDataset(X_list, omega_list_scaled)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    encoder.train()
    reg_head.train()
    
    print(f"[Encoder] Starting pretraining for {epochs} epochs...")
    for ep in range(epochs):
        total_loss = 0.0
        for Xb, Yb in dl:
            Zb = encoder(Xb)
            pred = reg_head(Zb)
            loss = loss_fn(pred, Yb)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * Xb.size(0)
        
        avg_loss = total_loss / len(ds)
        print(f"[Encoder] Epoch {ep+1}/{epochs} MSE={avg_loss:.4f}")

    # Freeze encoder after training
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    
    return encoder, reg_head


# -----------------------------
# 6) Summary Standardizer
# -----------------------------

class SummaryStandardizer:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# -----------------------------
# 7) Simulation Blocks
# -----------------------------

@dataclass
class SimBlock:
    omegas: torch.Tensor      # (N, D) in transformed space
    embeds: np.ndarray        # (N, E) raw embeddings (before z-score)


def simulate_round(prior_sampler: Callable[[Tuple[int]], torch.Tensor],
                   N: int,
                   encoder: TrialTransformer,
                   seed_offset: int = 0) -> SimBlock:
    """
    Samples parameters (omegas), runs the simulator, and computes embeddings.
    """
    omegas = prior_sampler((N,))
    embeds = []
    
    for i in range(N):
        params = untransform(omegas[i])
        df = simulate_data(params, seed=seed_offset + i)
        X = build_per_trial_features(df)
        X_t = torch.tensor(X[None, ...], dtype=torch.float32)
        
        with torch.no_grad():
            z = encoder(X_t).squeeze(0).cpu().numpy().astype(np.float32)
        embeds.append(z)
        
    return SimBlock(omegas=omegas, embeds=np.vstack(embeds))


# -----------------------------
# 8) SNPE Training
# -----------------------------

def train_snpe_from_arrays(omegas: torch.Tensor,
                           embeds_z: torch.Tensor,
                           prior: BoxUniform,
                           density_estimator: str = "nsf"):
    """
    Trains the SNPE posterior density estimator.
    """
    inference = SNPE(prior=prior, density_estimator=density_estimator)
    inference = inference.append_simulations(omegas, embeds_z)
    density = inference.train()
    posterior = inference.build_posterior(density)
    return posterior


# -----------------------------
# 9) Inference & Sampling
# -----------------------------

def condition_and_sample(posterior, df_obs: pd.DataFrame,
                         encoder: TrialTransformer,
                         standardizer: SummaryStandardizer,
                         num_samples: int = 20000) -> torch.Tensor:
    """
    Given observed data (df_obs), encode it, standardize it, and sample from the posterior.
    """
    X = build_per_trial_features(df_obs)
    X_t = torch.tensor(X[None, ...], dtype=torch.float32)
    
    with torch.no_grad():
        z = encoder(X_t).squeeze(0).cpu().numpy().astype(np.float32)[None, :]
    
    z_z = standardizer.transform(z)[0]
    z_t = torch.tensor(z_z, dtype=torch.float32)
    
    samples = posterior.sample((num_samples,), x=z_t)
    return samples


# -----------------------------
# 10) Utilities (IO, Recovery)
# -----------------------------

def save_object(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_object(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_encoder(in_dim, model_dim=64, out_dim=64, nhead=4, nlayers=2, dropout=0.1):
    enc = TrialTransformer(in_dim, model_dim=model_dim, nhead=nhead,
                           nlayers=nlayers, dropout=dropout, out_dim=out_dim, use_pos_enc=True)
    enc.load_state_dict(torch.load(ENC_PATH, map_location="cpu"))
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc

def save_standardizer(stdzr: SummaryStandardizer):
    os.makedirs(ART_DIR, exist_ok=True)
    np.save(STD_MEAN_PATH, stdzr.mean_)
    np.save(STD_STD_PATH,  stdzr.std_)

def load_standardizer():
    stdzr = SummaryStandardizer()
    stdzr.mean_ = np.load(STD_MEAN_PATH)
    stdzr.std_ = np.load(STD_STD_PATH)
    return stdzr

def posterior_stats_original(samples_t: torch.Tensor):
    """
    Computes statistics (mean, median, 90% CI) in the original parameter space.
    """
    samples_np = samples_t.detach().cpu().numpy()
    rows = [untransform(torch.tensor(s)) for s in samples_np]
    M = pd.DataFrame(rows)
    M = M[PARAM_ORDER + list(FIXED_PARAMS.keys())]
    
    stats = {}
    for k in PARAM_ORDER:
        stats[k] = {
            "mean":   M[k].mean(),
            "median": M[k].median(),
            "p05":    M[k].quantile(0.05),
            "p95":    M[k].quantile(0.95),
        }
    return stats, M

def recovery_sweep(prior: BoxUniform,
                   posterior,
                   encoder: nn.Module,
                   standardizer,
                   K: int = 100,
                   num_post: int = 4000,
                   seed: int = 2024):
    """
    Performs a recovery test:
    1. Sample K ground-truth omegas.
    2. Simulate data for each.
    3. Infer posterior and check if ground truth is within the 90% CI.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    abs_err = {k: [] for k in PARAM_ORDER}
    sq_err = {k: [] for k in PARAM_ORDER}
    hit90 = {k: [] for k in PARAM_ORDER}
    details = []

    omegas_true = prior.sample((K,))

    for i in range(K):
        omega_star = omegas_true[i]
        omega_true_orig = untransform(omega_star)

        # Simulate observed dataset
        df_obs = simulate_data(omega_true_orig, seed=seed + i)

        # Infer
        samples_t = condition_and_sample(
            posterior, df_obs, encoder, standardizer, num_samples=num_post)
        stats, _ = posterior_stats_original(samples_t)

        rec = {"case": i}
        for k in PARAM_ORDER:
            mu = stats[k]["mean"]
            lo = stats[k]["p05"]
            hi = stats[k]["p95"]
            gt = omega_true_orig[k]

            abs_err[k].append(abs(mu - gt))
            sq_err[k].append((mu - gt) ** 2)
            hit90[k].append(1.0 if lo <= gt <= hi else 0.0)

            rec[f"gt_{k}"] = gt
            rec[f"mu_{k}"] = mu
            rec[f"lo90_{k}"] = lo
            rec[f"hi90_{k}"] = hi
            rec[f"hit90_{k}"] = hit90[k][-1]
        details.append(rec)

    rows = []
    for k in PARAM_ORDER:
        mae = float(np.mean(abs_err[k]))
        rmse = float(np.sqrt(np.mean(sq_err[k])))
        cov = float(np.mean(hit90[k]))
        rows.append({"param": k, "MAE": mae, "RMSE": rmse, "coverage90": cov})
    metrics = pd.DataFrame(rows).set_index("param")

    return metrics, pd.DataFrame(details)


def parse_id_year_from_filename(filename: str):
    """
    Example filename: NDAR_INV0A4P0LWM_baseline_year_1_arm_1_sst.csv.zip
    Returns: ("INV0A4P0LWM", "baseline")
    """
    base = os.path.basename(filename)
    parts = base.split("_")
    sid = parts[1] if len(parts) > 1 else base
    year = parts[2] if len(parts) > 2 else "unknown"
    return sid, year


# -----------------------------
# 11) Main Execution
# -----------------------------

def main():
    warnings.filterwarnings("ignore")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    
    # Global Seed
    SEED = 137
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    parser = argparse.ArgumentParser(description="Transformer-based SNPE Pipeline for POMDP-SST")
    parser.add_argument("--stage", choices=["all", "pretrain", "snpe", "recover", "posterior"], default="all")
    parser.add_argument("--n1_pre", type=int, default=20000, help="Simulations for pretraining/Round 1")
    parser.add_argument("--epochs", type=int, default=5, help="Pretraining epochs")
    parser.add_argument("--n2", type=int, default=15000, help="Round 2 simulations")
    parser.add_argument("--mix_prior_frac", type=float, default=0.2, help="Fraction of prior samples in proposal")
    parser.add_argument("--density", choices=["nsf", "maf", "mdn"], default="nsf", help="SBI density estimator")
    
    # Recovery arguments
    parser.add_argument("--K", type=int, default=20, help="Number of recovery test cases")
    parser.add_argument("--num_post", type=int, default=1000, help="Samples per posterior in recovery")
    
    # Real data arguments
    parser.add_argument("--real_csv", type=str, default=None, help="Path to single real data CSV")
    parser.add_argument("--sst_folder", type=str, default=None, help="Folder containing real data CSVs")
    parser.add_argument("--glob_pat", type=str, default="NDAR_*.csv*", help="File pattern for sst_folder")
    parser.add_argument("--num_samples", type=int, default=20000, help="Posterior samples per subject")

    args = parser.parse_args()

    prior, _, _ = make_box_prior()
    D = len(PARAM_ORDER)

    os.makedirs(ART_DIR, exist_ok=True)

    # ========================
    # Stage: Pretrain Encoder
    # ========================
    if args.stage in ["all", "pretrain"]:
        N1_pre = args.n1_pre
        print(f"[Pretrain] Simulating {N1_pre} pairs for encoder training...")
        
        omegas_1 = prior.sample((N1_pre,))
        X_list, Omega_list = [], []
        
        for i in range(N1_pre):
            params = untransform(omegas_1[i])
            df = simulate_data(params, seed=i)
            X = build_per_trial_features(df)
            X_list.append(X)
            Omega_list.append(omegas_1[i].numpy())
        
        in_dim = X_list[0].shape[1]

        encoder, _ = train_encoder_on_simulations(
            X_list, Omega_list, in_dim=in_dim, omega_dim=D,
            epochs=args.epochs, batch_size=64
        )
        torch.save(encoder.state_dict(), ENC_PATH)
        print(f"[Pretrain] Encoder saved to {ENC_PATH}")
        
        if args.stage == "pretrain":
            print("[Pretrain] Done.")
            return

    else:
        # Load existing encoder; determine input dimension from a dummy simulation
        dummy_df = simulate_data(untransform(prior.sample((1,))[0]), seed=0)
        in_dim = build_per_trial_features(dummy_df).shape[1]
        encoder = load_encoder(in_dim)

    # ========================
    # Stage: SNPE Training
    # ========================
    if args.stage in ["all", "snpe"]:
        # If running only SNPE stage, regenerate Round 1 data
        if args.stage == "snpe":
            N1_pre = args.n1_pre
            print(f"[Round1] Regenerating {N1_pre} simulations for embeddings...")
            omegas_1 = prior.sample((N1_pre,))
            X_list = []
            for i in range(N1_pre):
                params = untransform(omegas_1[i])
                df = simulate_data(params, seed=i)
                X_list.append(build_per_trial_features(df))

        print(f"[Round1] generating embeddings for {len(X_list)} simulations...")
        embeds_1 = []
        for X in X_list:
            X_t = torch.tensor(X[None, ...], dtype=torch.float32)
            with torch.no_grad():
                z = encoder(X_t).squeeze(0).cpu().numpy().astype(np.float32)
            embeds_1.append(z)
        embeds_1 = np.vstack(embeds_1)

        stdzr = SummaryStandardizer()
        embeds_1_z = stdzr.fit_transform(embeds_1)
        save_standardizer(stdzr)

        print(f"[Round1] Training SNPE ({args.density}) posterior...")
        posterior_1 = train_snpe_from_arrays(
            omegas_1, torch.tensor(embeds_1_z, dtype=torch.float32), prior, density_estimator=args.density
        )
        save_object(posterior_1, POST1_PATH)

        # --- Round 2: Active Learning ---
        N2 = args.n2
        if N2 > 0:
            print(f"[Round2] Simulating {N2} using proposal distribution...")
            
            # Select anchor points for mixture proposal
            M_anchors = min(8, embeds_1_z.shape[0])
            anchor_idx = np.random.choice(embeds_1_z.shape[0], size=M_anchors, replace=False)
            pilot_x_list = [torch.tensor(embeds_1_z[j], dtype=torch.float32) for j in anchor_idx]

            def proposal_sampler(shape):
                n = shape[0]
                n_prior = int(args.mix_prior_frac * n)
                n_post = n - n_prior
                
                # Sample from Prior
                th_prior = prior.sample((n_prior,))
                
                # Sample from Round-1 Posterior (mixture of anchors)
                per_anchor = max(1, n_post // M_anchors)
                th_posts = []
                for px in pilot_x_list:
                    th_posts.append(posterior_1.sample((per_anchor,), x=px))
                th_post = torch.cat(th_posts, dim=0)
                
                # Padding if needed
                if th_post.shape[0] < n_post:
                    extra = prior.sample((n_post - th_post.shape[0],))
                    th_post = torch.cat([th_post, extra], dim=0)
                    
                return torch.cat([th_prior, th_post[:n_post]], dim=0)

            block2 = simulate_round(proposal_sampler, N2, encoder, seed_offset=1_000_000)
            embeds_2_z = stdzr.transform(block2.embeds)
            
            # Merge Datasets
            omegas_all = torch.cat([omegas_1, block2.omegas], dim=0)
            embeds_all = torch.cat([
                torch.tensor(embeds_1_z, dtype=torch.float32),
                torch.tensor(embeds_2_z, dtype=torch.float32)
            ], dim=0)

            print("[Merged] Training final SNPE posterior...")
            posterior_final = train_snpe_from_arrays(
                omegas_all, embeds_all, prior, density_estimator=args.density
            )
        else:
            posterior_final = posterior_1

        save_object(posterior_final, POSTF_PATH)

    elif args.stage in ["recover", "posterior"]:
        stdzr = load_standardizer()
        posterior_final = load_object(POSTF_PATH)

    # ========================
    # Stage: Recovery Check
    # ========================
    if args.stage in ["all", "recover"]:
        print(f"[Recovery] Running recovery sweep (K={args.K})...")
        metrics, details = recovery_sweep(
            prior=prior,
            posterior=posterior_final,
            encoder=encoder,
            standardizer=stdzr,
            K=args.K,
            num_post=args.num_post,
            seed=4242
        )
        print(metrics)
        details.to_csv(os.path.join(ART_DIR, "recovery_details.csv"), index=False)

    # ========================
    # Stage: Inference on Data
    # ========================
    if args.stage in ["all", "posterior"]:
        out_root = ART_DIR
        
        # 1. Folder Mode
        if args.sst_folder is not None:
            files = sorted(glob.glob(os.path.join(args.sst_folder, args.glob_pat)))
            print(f"[Posterior] Processing {len(files)} files in {args.sst_folder}...")

            os.makedirs(os.path.join(out_root, "subjects"), exist_ok=True)
            summaries = []
            n_ok, n_err = 0, 0

            for fp in files:
                sid, year = parse_id_year_from_filename(fp)
                try:
                    df_obs = preprocessing(fp)
                    samples = condition_and_sample(
                        posterior_final, df_obs, encoder, stdzr, num_samples=args.num_samples
                    )

                    # Stats
                    samples_np = samples.detach().cpu().numpy()
                    rows = [untransform(torch.tensor(s)) for s in samples_np]
                    post_df = pd.DataFrame(rows)
                    summary = post_df.describe(percentiles=[0.05, 0.5, 0.95]).T[[
                        "mean", "std", "5%", "50%", "95%"]]
                    summary.index.name = "param"

                    # Save per subject
                    subj_dir = os.path.join(out_root, "subjects", f"{sid}_{year}")
                    os.makedirs(subj_dir, exist_ok=True)
                    post_df.to_csv(os.path.join(subj_dir, "posterior_samples.csv"), index=False)
                    summary.to_csv(os.path.join(subj_dir, "posterior_summary.csv"))

                    # Aggregate
                    s_sum = summary.copy()
                    s_sum.insert(0, "subject_year", year)
                    s_sum.insert(0, "subject_id", sid)
                    summaries.append(s_sum.reset_index())
                    n_ok += 1
                except Exception as e:
                    print(f"[ERROR] {os.path.basename(fp)}: {e}")
                    n_err += 1

            if summaries:
                big = pd.concat(summaries, axis=0, ignore_index=True)
                big.to_csv(os.path.join(out_root, "posterior_summary_all.csv"), index=False)
            print(f"[Posterior] Finished. OK={n_ok}, Errors={n_err}")

        # 2. Single File Mode
        elif args.real_csv is not None:
            sid, year = parse_id_year_from_filename(args.real_csv)
            print(f"[Posterior] Single file: {args.real_csv} (ID={sid})")

            df_obs = preprocessing(args.real_csv)
            samples = condition_and_sample(
                posterior_final, df_obs, encoder, stdzr, num_samples=args.num_samples
            )
            
            samples_np = samples.detach().cpu().numpy()
            rows = [untransform(torch.tensor(s)) for s in samples_np]
            post_df = pd.DataFrame(rows)
            summary = post_df.describe(percentiles=[0.05, 0.5, 0.95]).T[[
                "mean", "std", "5%", "50%", "95%"]]
            
            # Output
            single_dir = os.path.join(out_root, "subjects", f"{sid}_{year}")
            os.makedirs(single_dir, exist_ok=True)
            post_df.to_csv(os.path.join(single_dir, "posterior_samples.csv"), index=False)
            summary.to_csv(os.path.join(single_dir, "posterior_summary.csv"))
            print(summary)

if __name__ == "__main__":
    main()

# -----------------------------
# Example Usage:
# -----------------------------
# A. Smoke Test (Fast Debugging)
# 1. Pretrain Encoder (very quick)
# python pipeline_transformer_snpe.py \
#   --stage pretrain \
#   --n1_pre 500 \
#   --epochs 2

# # 2. Train SNPE (Round 1 & 2)
# python pipeline_transformer_snpe.py \
#   --stage snpe \
#   --n1_pre 500 \
#   --n2 200 \
#   --density maf

# # 3. Quick Recovery Check (5 test cases)
# python pipeline_transformer_snpe.py \
#   --stage recover \
#   --K 5 \
#   --num_post 500
# -----------------------------
# B. Real Run (Production / High Fidelity)
# 1. Pretrain Encoder (longer training for better embeddings)
# python pipeline_transformer_snpe.py \
#   --stage pretrain \
#   --n1_pre 30000 \
#   --epochs 8

# # 2. Train SNPE (Heavy computation)
# python pipeline_transformer_snpe.py \
#   --stage snpe \
#   --n1_pre 30000 \
#   --n2 20000 \
#   --density nsf

# # 3. Full Recovery Sweep (200 test cases)
# python pipeline_transformer_snpe.py \
#   --stage recover \
#   --K 200 \
#   --num_post 4000

# # 4. Inference on Real Data Folder
# python pipeline_transformer_snpe.py \
#   --stage posterior \
#   --sst_folder "/home/wwang/data/all_valid" \
#   --glob_pat "NDAR_*.csv.zip" \
#   --num_samples 14988