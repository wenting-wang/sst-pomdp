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
OUT_IMG = Path(f"outputs/gs_context_analysis_{SUBJECT_ID}.png")

def main():
    if not INPUT_FILE.exists():
        print(f"File not found: {INPUT_FILE}")
        return

    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    
    # 1. 保持全局 PCA 坐标一致性
    # 我们必须用所有数据训练 PCA，这样画出来的坐标系才和之前的图一样
    emb_cols = [c for c in df.columns if c.startswith('dim_')]
    X = df[emb_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    
    df['pc1'] = coords[:, 0]
    df['pc2'] = coords[:, 1]
    var_exp = pca.explained_variance_ratio_ * 100

    # 2. 标记上下文 (History Context)
    df['prev_result'] = df['result'].shift(1)
    
    def get_gs_context(row):
        # 如果当前 trial 不是 GS，我们不关心
        if row['result'] != 'GS':
            return 'Non-GS'
        
        prev = row['prev_result']
        if pd.isna(prev):
            return 'Start'
        
        # 定义 Post-Error (犯错后)
        if prev in ['SE', 'GE']:
            return 'Post-Error GS'
        
        # 定义 Post-Success (成功后)
        # 注意：SS (Stop Success) 也是成功，视为一种正向/受控的上下文
        if prev in ['GS', 'SS']:
            return 'Post-Success GS'
            
        return 'Other'

    df['gs_context'] = df.apply(get_gs_context, axis=1)

    # 3. 筛选数据：只看 GS Trials
    df_gs = df[df['gs_context'].isin(['Post-Error GS', 'Post-Success GS'])].copy()
    print(f"Analyzed GS Trials: {len(df_gs)}")
    print(f" - Post-Success: {len(df_gs[df_gs['gs_context']=='Post-Success GS'])}")
    print(f" - Post-Error:   {len(df_gs[df_gs['gs_context']=='Post-Error GS'])}")

    # ================= PLOTTING =================
    sns.set_context("paper", font_scale=1.5)
    # sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
    
    # --- 左图：History Effect (Post-Error vs Post-Success) ---
    ax0 = axes[0]
    # 画背景灰点 (所有轨迹作为参考)
    ax0.scatter(df['pc1'], df['pc2'], c='lightgrey', s=10, alpha=0.1, label='All Trials (Background)')
    
    # 画 Post-Success (蓝色，代表常态)
    subset_succ = df_gs[df_gs['gs_context'] == 'Post-Success GS']
    sns.kdeplot(
        data=subset_succ, x='pc1', y='pc2', 
        ax=ax0, color='royalblue', alpha=0.3, fill=True, thresh=0.05
    )
    ax0.scatter(subset_succ['pc1'], subset_succ['pc2'], 
                c='royalblue', s=40, alpha=0.8, edgecolors='white', label='Post-Success GS')

    # 画 Post-Error (红色，代表调整态)
    subset_err = df_gs[df_gs['gs_context'] == 'Post-Error GS']
    if len(subset_err) > 1: # 需要至少2个点才能画 KDE
        sns.kdeplot(
            data=subset_err, x='pc1', y='pc2', 
            ax=ax0, color='crimson', alpha=0.3, fill=True, thresh=0.05
        )
    ax0.scatter(subset_err['pc1'], subset_err['pc2'], 
                c='crimson', s=60, marker='D', alpha=1.0, edgecolors='white', label='Post-Error GS')

    ax0.set_title("History Effect", fontweight='bold')
    ax0.legend(loc='upper right', frameon=True)
    ax0.set_xlabel(f"PC 1 ({var_exp[0]:.1f}%)")
    ax0.set_ylabel(f"PC 2 ({var_exp[1]:.1f}%)")

    # --- 右图：Time Effect (Early vs Late) ---
    # 用来通过对比排除"这只是时间漂移"的可能性
    ax1 = axes[1]
    ax1.scatter(df['pc1'], df['pc2'], c='lightgrey', s=10, alpha=0.1) # 背景
    
    sc = ax1.scatter(
        df_gs['pc1'], df_gs['pc2'], 
        c=df_gs['trial_index'], 
        cmap='viridis', 
        s=40, alpha=0.9, edgecolors='white', linewidth=0.5
    )
    
    ax1.set_title("GS by Time", fontweight='bold')
    ax1.set_xlabel(f"PC 1 ({var_exp[0]:.1f}%)")
    fig.colorbar(sc, ax=ax1, label="Trial Index (Time)")

    plt.suptitle(f"Subject {SUBJECT_ID}", fontsize=20, y=0.98)
    plt.tight_layout()
    
    save_path = OUT_IMG
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    main()