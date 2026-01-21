import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ================= CONFIG =================
SUBJECT_ID = 'INVJTCXNL4V'
INPUT_FILE = Path(f"outputs/trajectory_{SUBJECT_ID}.csv")
OUT_IMG = Path(f"outputs/trajectory_split_{SUBJECT_ID}.png")

# 定义顺序
OUTCOME_ORDER = ['GS','GE', 'GM', 'SS', 'SE']
OUTCOME_NAMES = {
    'GS': 'Go Success',
    'GE': 'Go Error',
    'GM': 'Go Missing',
    'SS': 'Stop Success',
    'SE': 'Stop Error',
}

def main():
    if not INPUT_FILE.exists():
        print(f"File not found: {INPUT_FILE}")
        return

    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    
    # 1. PCA 降维 (基于所有点)
    emb_cols = [c for c in df.columns if c.startswith('dim_')]
    X = df[emb_cols].values
    
    # 标准化是必须的
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    
    df['pc1'] = coords[:, 0]
    df['pc2'] = coords[:, 1]
    
    var_exp = pca.explained_variance_ratio_ * 100
    
    # 2. 绘图
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    
    # 1行5列
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharex=True, sharey=True)
    
    # 统一坐标轴范围
    x_min, x_max = df['pc1'].min(), df['pc1'].max()
    y_min, y_max = df['pc2'].min(), df['pc2'].max()
    pad = 1.0
    
    # 设置单色渐变色板 (越深=时间越晚)
    # 可选: 'Blues', 'Greens', 'Oranges', 'Purples', 'Greys'
    CMAP_NAME = 'Blues' 

    for i, outcome in enumerate(OUTCOME_ORDER):
        ax = axes[i]
        
        # --- 背景: 完整的灰色轨迹 ---
        # 这样可以看出当前点在"全局信念空间"的哪个位置
        ax.plot(df['pc1'], df['pc2'], c='lightgrey', lw=0.5, alpha=0.5, zorder=0)
        ax.scatter(df['pc1'], df['pc2'], c='lightgrey', s=5, alpha=0.2, zorder=0)
        
        # --- 前景: 当前 Outcome 的点 ---
        subset = df[df['result'] == outcome]
        
        if len(subset) > 0:
            sc = ax.scatter(
                subset['pc1'], subset['pc2'], 
                c=subset['trial_index'], # 颜色映射到时间
                cmap=CMAP_NAME,
                vmin=0, vmax=360,        # 固定颜色范围
                s=40, 
                alpha=0.9,
                edgecolors='white',      # 加白边增加对比度
                linewidth=0.5,
                zorder=10
            )
        else:
            ax.text(0.5, 0.5, "No Trials", transform=ax.transAxes, 
                    ha='center', color='grey', style='italic')

        # 格式化
        ax.set_title(f"{OUTCOME_NAMES[outcome]} (n={len(subset)})", fontsize=14, fontweight='bold')
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        
        if i == 0:
            ax.set_ylabel(f"PC 2 ({var_exp[1]:.1f}%)")
        else:
            ax.set_ylabel("")
            
        ax.set_xlabel(f"PC 1 ({var_exp[0]:.1f}%)")
        sns.despine(ax=ax)

    # 添加 Colorbar
    fig.subplots_adjust(right=0.92, wspace=0.1)
    cbar_ax = fig.add_axes([0.93, 0.25, 0.008, 0.5]) # [left, bottom, width, height]
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Trial Index (Time)', fontsize=12)
    # cbar.set_ticks([0, 360])
    # cbar.set_ticklabels(['Start', 'End'])
    
    plt.suptitle(f"Subject {SUBJECT_ID}: Behaviour Trajectories by Outcome", fontsize=18, y=1.05)
    
    save_path = OUT_IMG
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved split plot to: {save_path}")

if __name__ == "__main__":
    main()