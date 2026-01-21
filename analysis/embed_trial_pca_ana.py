"""
gen_trajectory_data.py
用于提取指定被试的 Trial-level embedding (不进行 mean pooling)。
"""
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import glob

# 引入项目路径以导入 utils
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

# 复用 tesbi.py 中的核心工具
from utils.preprocessing import preprocessing

# ================= CONFIGURATION =================
SUBJECT_ID = 'INVJTCXNL4V'
SST_FOLDER = "/kyb/agpd/wwang/data/sst_valid_base" # 请确认这是存放真实数据的路径
MODEL_PATH = "outputs/encoder.pt"
OUT_CSV = f"outputs/trajectory_{SUBJECT_ID}.csv"

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= MODEL DEFINITION (复制自 tesbi.py) =================
# 我们需要在这里重新定义类，并添加一个不池化的方法
class SinusoidalPE(nn.Module):
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
        # 注意：这里不需要 head，因为我们要的是 encoder 输出的 latent representations

    def get_sequence_embeddings(self, x):
        """
        New method: Returns (Batch, SeqLen, HiddenDim) without pooling.
        """
        h = self.proj(x)
        if self.use_pos:
            h = self.pe(h)
        h = self.encoder(h)
        # 这里不进行 h.mean(dim=1)，也不过 self.head
        # 直接返回 Transformer Encoder 的输出 (Batch, Seq, 64)
        return h

# ================= FEATURE ENGINEERING (复制自 tesbi.py) =================
RESULT_LEVELS = ["GS", "GE", "GM", "SS", "SE"]

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

# ================= MAIN EXECUTION =================
def main():
    print(f"Searching for subject {SUBJECT_ID} in {SST_FOLDER}...")
    
    # 1. Find File
    pattern = str(Path(SST_FOLDER) / f"NDAR_{SUBJECT_ID}_*.csv*")
    files = glob.glob(pattern)
    if not files:
        # 尝试不带 NDAR 前缀
        pattern = str(Path(SST_FOLDER) / f"*{SUBJECT_ID}*.csv*")
        files = glob.glob(pattern)
        
    if not files:
        raise FileNotFoundError(f"Could not find data file for {SUBJECT_ID}")
    
    target_file = files[0]
    print(f"Processing file: {target_file}")

    # 2. Prepare Data
    df_obs = preprocessing(target_file)
    X = build_per_trial_features(df_obs)
    in_dim = X.shape[1]
    print(f"Input features shape: {X.shape} (Trials={X.shape[0]}, Features={in_dim})")

    # 3. Load Model
    print(f"Loading encoder from {MODEL_PATH}...")
    # Initialize model with same structure as training
    model = TrialTransformer(in_dim, out_dim=64, use_pos_enc=True).to(device)
    
    # Load weights (需要处理可能的 strict=False 情况，因为我们可能去掉了 head)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    
    # 这是一个小技巧：由于我们的 TrialTransformer 定义和 tesbi.py 基本一致，
    # 只是我们不再需要 head。如果 state_dict 包含 head 的权重，
    # load_state_dict 默认会报错。我们使用 strict=False 来忽略 head 的权重不匹配。
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 4. Inference
    tensor_x = torch.tensor(X[None, ...], dtype=torch.float32).to(device) # (1, Seq, Feat)
    
    with torch.no_grad():
        # 调用我们新写的 get_sequence_embeddings 方法
        # 输出形状: (1, SeqLen, 64)
        seq_emb = model.get_sequence_embeddings(tensor_x)
    
    # Remove batch dim -> (SeqLen, 64)
    embeddings_np = seq_emb.squeeze(0).cpu().numpy()

    # 5. Save
    # 创建 DataFrame，保存 embedding 和一些关键行为特征用于绘图
    res_df = pd.DataFrame(embeddings_np, columns=[f"dim_{i}" for i in range(embeddings_np.shape[1])])
    
    # 添加行为数据方便后续分析
    res_df['trial_index'] = range(1, len(res_df) + 1)
    res_df['result'] = df_obs['result'].values
    res_df['rt'] = df_obs['rt'].values
    res_df['ssd'] = df_obs['ssd'].values
    
    res_df.to_csv(OUT_CSV, index=False)
    print(f"Saved trajectory embeddings to {OUT_CSV}")

if __name__ == "__main__":
    main()