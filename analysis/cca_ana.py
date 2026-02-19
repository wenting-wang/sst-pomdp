import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from pathlib import Path

# ==========================================
# 1. Configuration & Paths
# ==========================================
# Update these paths to match your local environment
OUT_DIR = Path('/Users/w/sst-abcd-adhd-mac/sst-pomdp-main/outputs/')
EMBED_CSV = Path("/Users/w/Desktop/833725/embeddings.csv") 
STATS_CSV = OUT_DIR / "stats_beh_full.csv"
OUTPUT_PLOT = OUT_DIR / "cca.png"

# ==========================================
# 2. Plotting Style (Larger Fonts for PDF Clarity)
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 12          
plt.rcParams['axes.linewidth'] = 1.2    

# ==========================================
# 3. Data Loading & Preprocessing
# ==========================================
print("Loading data...")
df_emb = pd.read_csv(EMBED_CSV)
df_stats = pd.read_csv(STATS_CSV)

df_emb['subject_id'] = df_emb['subject_id'].astype(str)
df_stats['subject_id'] = df_stats['subject_id'].astype(str)

df = pd.merge(df_emb, df_stats, on=['subject_id', 'year', 'filename'], how='inner')
print(f"Data merged. Total subjects: {len(df)}")

emb_cols = [f'emb_{i}' for i in range(64)]
target_stats = [
    'mrt_gs', 'perc_gs', 'perc_gm', 'mrt_ge', 
    'ssrt', 'perc_ss', 'mrt_se',
    'rate_perc_ss_ssd', 'rate_rt_se_ssd',
    'pes', 'pss', 'pges', 'rt_acf_1', 'rt_acf_2', 'rt_slope'
]

available_stats = [c for c in target_stats if c in df.columns]

X = df[emb_cols].values
Y = df[available_stats].values

mask = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
X_clean = X[mask]
Y_clean = Y[mask]

std_y = Y_clean.std(axis=0)
valid_cols_idx = np.where(std_y > 1e-6)[0]
Y_clean = Y_clean[:, valid_cols_idx]
final_stats_names = [available_stats[i] for i in valid_cols_idx]

X_std = StandardScaler().fit_transform(X_clean)
Y_std = StandardScaler().fit_transform(Y_clean)

# ==========================================
# 4. Run CCA
# ==========================================
print("Running CCA...")
cca = CCA(n_components=1)
cca.fit(X_std, Y_std)
X_c, Y_c = cca.transform(X_std, Y_std)

corr, p_val = pearsonr(X_c[:, 0], Y_c[:, 0])

print("\n" + "="*50)
print(f"CCA RESULTS FOR MANUSCRIPT:")
print(f"Canonical Correlation (r): {corr:.4f}")
print(f"P-value: {p_val:.4e}")
print("="*50 + "\n")

# ==========================================
# 5. Visualization
# ==========================================
raw_loadings = [np.corrcoef(Y_std[:, i], Y_c[:, 0])[0, 1] for i in range(Y_std.shape[1])]

name_map = {
    'mrt_gs': 'Mean Go Success RT',
    'mrt_se': 'Mean Stop Error RT',
    'mrt_ge': 'Mean Go Error RT',
    'perc_gs': 'Go Success Rate',
    'perc_gm': 'Go Missing Rate',
    'perc_ss': 'Stop Success Rate',
    'ssrt': 'SSRT',
    'rate_perc_ss_ssd': 'Stop Success Slope over SSD',
    'rate_rt_se_ssd': 'Stop Error RT Slope over SSD',
    'pes': 'Post-Stop Error Slowing',
    'pss': 'Post-Stop Success Slowing',
    'pges': 'Post-Go Error Slowing',
    'rt_acf_1': 'RT Autocorrelation (Lag-1)',
    'rt_acf_2': 'RT Autocorrelation (Lag-2)',
    'rt_slope': 'Fatigue Slope (RTs over Trials)'
}

pretty_names = [name_map.get(n, n) for n in final_stats_names]

df_plot = pd.DataFrame({
    'Feature': pretty_names,
    'Loading': raw_loadings,
    'Abs_Loading': np.abs(raw_loadings)
})

df_sorted = df_plot.sort_values(by='Abs_Loading', ascending=True)

fig, ax = plt.subplots(figsize=(7, 6)) 

COLOR_POS = '#1b5e63'  # Zhu-Qing (Bamboo Green)
COLOR_NEG = '#a05d2c'  # Li-Hong (Chestnut/Lacquer Red)

colors = [COLOR_POS if x > 0 else COLOR_NEG for x in df_sorted['Loading']]

bars = ax.barh(np.arange(len(df_sorted)), df_sorted['Loading'], color=colors, height=0.75)

ax.set_yticks(np.arange(len(df_sorted)))
ax.set_yticklabels(df_sorted['Feature'])

ax.set_xlabel('Correlation with Canonical Variate', weight='regular')
ax.set_xlim(-0.8, 1.0)

# Minimalist Style
ax.grid(False) 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# Add value annotations
for i, v in enumerate(df_sorted['Loading']):
    offset = 0.02 if v > 0 else -0.02
    ha = 'left' if v > 0 else 'right'
    ax.text(v + offset, i, f"{v:.3f}", va='center', ha=ha, fontsize=11)

ax.axvline(0, color='black', linewidth=1.0, linestyle='-')

plt.tight_layout()

print(f"Saving figure to: {OUTPUT_PLOT}")
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
plt.show()