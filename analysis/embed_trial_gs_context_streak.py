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
OUT_IMG = Path(f"outputs/gs_winning_streak_{SUBJECT_ID}.png")

def main():
    if not INPUT_FILE.exists():
        print(f"File not found: {INPUT_FILE}")
        return

    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)

    # 1. 保持坐标系一致 (Global PCA)
    # 必须用所有点来构建坐标系，这样才能和之前的图对上
    emb_cols = [c for c in df.columns if c.startswith('dim_')]
    X = df[emb_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    df['pc1'] = coords[:, 0]
    df['pc2'] = coords[:, 1]
    var_exp = pca.explained_variance_ratio_ * 100

    # 2. 计算连胜 (Winning Streak)
    # 定义成功: GS (Go Success) 或 SS (Stop Success)
    # 我们关心的是当前 Trial *开始前* 已经连续赢了多少把
    is_success = df['result'].isin(['GS', 'SS']).values.astype(int)
    
    streaks = np.zeros(len(df), dtype=int)
    current_streak = 0
    
    for i in range(len(df)):
        # 记录当前 trial 开始时的 streak (即上一把结束后的状态)
        streaks[i] = current_streak
        
        # 更新 streak 供下一把使用
        if is_success[i] == 1:
            current_streak += 1
        else:
            current_streak = 0 # 犯错 (SE, GE, GM, etc.) -> 重置
            
    df['prev_streak'] = streaks

    # 3. 筛选数据：只看 GS Trials
    df_gs = df[df['result'] == 'GS'].copy()
    
    # 为了绘图的可读性，设置一个视觉上限 (Visual Cap)
    VISUAL_MAX = df['prev_streak'].max()-1
    df_gs['streak_vis'] = df_gs['prev_streak'].clip(upper=VISUAL_MAX)
    
    print(f"Max streak found in data: {df['prev_streak'].max()}")
    
    # ================= PLOTTING =================
    sns.set_context("paper", font_scale=1.5)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # A. 背景：所有点 (浅灰)
    ax.scatter(df['pc1'], df['pc2'], c='lightgrey', s=30, alpha=0.15, label='All Trials')
    
    # B. 前景：GS Trials，颜色映射 streak
    # 使用 'Purples' 单色渐变，颜色越深代表连胜越多
    CMAP = 'Blues' 

    
    sc = ax.scatter(
        df_gs['pc1'], df_gs['pc2'], 
        c=df_gs['streak_vis'], 
        cmap=CMAP,
        s=60, 
        alpha=0.9, 
        edgecolors='white', 
        linewidth=0.5
    )

    sns.kdeplot(
        data=df_gs, x='pc1', y='pc2', 
        ax=ax, color='blue', alpha=0.1, fill=True, thresh=0.1
    )

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Consecutive Successes', fontsize=14)
    
    # 处理 Colorbar 刻度，加上 "+" 号
    cbar.set_ticks(np.linspace(0, VISUAL_MAX, 6).astype(int))
    labels = [str(t) for t in np.linspace(0, VISUAL_MAX, 6).astype(int)]
    cbar.ax.set_yticklabels(labels)

    ax.set_title(f"Subject {SUBJECT_ID}: Winning Effect on GS", fontsize=18)
    ax.set_xlabel(f"PC 1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"PC 2 ({var_exp[1]:.1f}%)")
    
   

    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {OUT_IMG}")

if __name__ == "__main__":
    main()