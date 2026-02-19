import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from pathlib import Path
from matplotlib.gridspec import GridSpec

# ==========================================
# 1. Configuration & Paths
# ==========================================
# Update these paths to match your local environment
OUT_DIR = Path('/Users/w/sst-abcd-adhd-mac/sst-pomdp-main/outputs/')
EMBED_CSV = Path("/Users/w/Desktop/833725/embeddings.csv") 
STATS_CSV = OUT_DIR / "stats_beh_full.csv"
OUTPUT_PLOT = OUT_DIR / "cca.png"

# ==========================================
# 2. Plotting Style (PLOS CB Rules)
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 9           # PLOS CB requires 8-12pt font
plt.rcParams['axes.linewidth'] = 1.0    

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
# 5. Visualization (PLOS CB Style 3-Panel)
# ==========================================
print("Generating final publication-quality plot...")

# ---------------------------------------------------------
# 1. 计算 Cross-Loadings
# ---------------------------------------------------------
y_variate = Y_c[:, 0]
x_variate = X_c[:, 0]

y_cross_loadings = np.array([np.corrcoef(Y_std[:, i], x_variate)[0, 1] for i in range(Y_std.shape[1])])
x_cross_loadings = np.array([np.corrcoef(X_std[:, i], y_variate)[0, 1] for i in range(X_std.shape[1])])

# ---------------------------------------------------------
# 2. 对双侧特征进行排序
# ---------------------------------------------------------
sort_idx_Y = np.argsort(y_cross_loadings)[::-1]
sort_idx_X = np.argsort(x_cross_loadings)[::-1]

y_cross_sorted = y_cross_loadings[sort_idx_Y]
x_cross_sorted = x_cross_loadings[sort_idx_X]

name_map = {
    'mrt_gs': 'Mean Go Success RT', 'mrt_se': 'Mean Stop Error RT', 'mrt_ge': 'Mean Go Error RT',
    'perc_gs': 'Go Success Rate', 'perc_gm': 'Go Missing Rate', 'perc_ss': 'Stop Success Rate',
    'ssrt': 'Stop Signal RT (SSRT)', 'rate_perc_ss_ssd': 'Stop Succ. Rate across SSD', 'rate_rt_se_ssd': 'Stop Error RT across SSD',
    'pes': 'Post-Stop Error Slowing', 'pss': 'Post-Stop Success Slowing', 'pges': 'Post-Go Error Slowing',
    'rt_acf_1': 'RT Autocorrelation (Lag-1)', 'rt_acf_2': 'RT Autocorrelation (Lag-2)', 'rt_slope': 'Fatigue Slope (RTs)'
}
sorted_Y_names = [name_map.get(final_stats_names[i], final_stats_names[i]) for i in sort_idx_Y]

X_std_sorted = X_std[:, sort_idx_X]
Y_std_sorted = Y_std[:, sort_idx_Y]

cross_corr_matrix = np.dot(Y_std_sorted.T, X_std_sorted) / (X_std_sorted.shape[0] - 1)

# ---------------------------------------------------------
# 3. 画图排版 (Top: Heatmap, Bottom: Two Bar Charts)
# ---------------------------------------------------------
# PLOS CB Max Width: 7.5 inches. 
fig = plt.figure(figsize=(7.5, 7.5)) # 稍微增加了整体高度以容纳拉开的间距

# 调大 hspace (从 0.25 -> 0.45) 拉开上下间距，确保不重叠
gs = GridSpec(2, 2, height_ratios=[1.2, 1], width_ratios=[1, 1], hspace=0.35, wspace=0.55)

# Separated color theme for Cross-loadings to distinguish from Pearson correlation
COLOR_BEHAV = '#4d4d4d'  # Dark Gray 
COLOR_EMBED = '#4d4d4d'  # Medium Gray

# ==================== Panel A: Heatmap (横向置顶) ====================
axA = fig.add_subplot(gs[0, :])
cax = axA.imshow(cross_corr_matrix, cmap='BrBG', vmin=-0.6, vmax=0.6, aspect='auto')

axA.set_yticks(np.arange(len(sorted_Y_names)))
axA.set_yticklabels(sorted_Y_names)
# axA.set_ylabel('Model-agnostic Behavioral Metrics', weight='regular')
axA.text(-0.205, 1.03, 'Behavioral Metrics', transform=axA.transAxes, ha='left', va='bottom', fontsize=9, weight='regular')

axA.set_xticks([0, 15, 31, 47, 63])
axA.set_xticklabels(['1', '16', '32', '48', '64'])
axA.set_xlabel('Latent Behavioral Embeddings (Sorted)', weight='regular')

# Colorbar moved to the right. Label placed ABOVE the colorbar.
cbar = fig.colorbar(cax, ax=axA, orientation='vertical', fraction=0.03, pad=0.02)
cbar.ax.set_title('Pearson r', pad=10, weight='regular', fontsize=9) 

# Label A 进一步往左移 (从 -0.15 变为 -0.3)
axA.text(-0.3, 1.05, 'A', transform=axA.transAxes, size=14, weight='bold')

# ==================== Panel B: Behavior Cross-Loadings ====================
axB = fig.add_subplot(gs[1, 0])
y_pos = np.arange(len(sorted_Y_names))

axB.barh(y_pos, y_cross_sorted[::-1], color=COLOR_BEHAV, height=0.6)
axB.set_yticks(y_pos)
axB.set_yticklabels(sorted_Y_names[::-1])
axB.set_xlabel('Cross-Loading with Embeddings', weight='regular')

axB.spines['top'].set_visible(False)
axB.spines['right'].set_visible(False)
axB.axvline(0, color='black', linewidth=1)

# Label B
axB.text(-0.7, 1.05, 'B', transform=axB.transAxes, size=14, weight='bold')

# ==================== Panel C: Embedding Cross-Loadings ====================
axC = fig.add_subplot(gs[1, 1])
x_pos = np.arange(len(x_cross_sorted))

axC.barh(x_pos, x_cross_sorted, color=COLOR_EMBED, height=0.6)
axC.set_xlabel('Cross-Loading with Behavioral Metrics', weight='regular')
axC.set_ylabel('Latent Behavioral Embeddings (Sorted)', weight='regular')
axC.spines['top'].set_visible(False)
axC.spines['right'].set_visible(False)

# Keep specific y-ticks for C to prevent clutter
axC.set_yticks([0, 15, 31, 47, 63])
axC.set_yticklabels(['1', '16', '32', '48', '64'])

# FLIP THE Y-AXIS: Largest positive loading at the top
axC.invert_yaxis()

# Label C 进一步往左移 (从 -0.35 变为 -0.45)
axC.text(-0.45, 1.05, 'C', transform=axC.transAxes, size=14, weight='bold')

# ---------------------------------------------------------
# 4. Final Output & Save
# ---------------------------------------------------------
print(f"Saving PLOS CB compliant figure to: {OUTPUT_PLOT}")
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight', format='png')