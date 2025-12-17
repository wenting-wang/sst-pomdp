import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from math import pi
from pathlib import Path

# ==========================================
# 0. Path Definitions & Style
# ==========================================
BASE_DIR = Path('/kyb/agpd/wwang/data/')
OUT_DIR = Path('/kyb/agpd/wwang/sst-pomdp/outputs/')

# Journal Style Settings
sns.set_context("paper", font_scale=1.4)
sns.set_style("white") # Clean background (no grid)
JOURNAL_PALETTE = sns.color_palette("deep", 4) # Aesthetic, low-saturation palette

# LaTeX Display Names for Plots
PARAM_DISPLAY_NAMES = {
    'q_d_n': r"$\chi'$",
    'q_d': r"$\chi$",
    'q_s_n': r"$\sigma'$",
    'q_s': r"$\sigma$",
    'cost_stop_error': r"$c_{\mathrm{se}}$",
    'cost_time': r"$c_t$",
}

# ==========================================
# 1. Data Loading & Filtering
# ==========================================
def load_and_filter_data():
    print(f"Loading data from {OUT_DIR}...")
    
    df_emb = pd.read_csv(OUT_DIR / 'embeddings.csv')
    df_stats = pd.read_csv(OUT_DIR / 'stats_beh_full.csv')
    df_post = pd.read_csv(OUT_DIR / 'posterior_summary.csv')

    df_emb['year'] = df_emb['year'].astype(str)
    df_stats['year'] = df_stats['year'].astype(str)
    df_post['subject_year'] = df_post['subject_year'].astype(str)

    df_post_pivot = df_post.pivot_table(index=['subject_id', 'subject_year'], 
                                        columns='param', 
                                        values='mean').reset_index()
    df_post_pivot.rename(columns={'subject_year': 'year'}, inplace=True)

    df_merged_1 = pd.merge(df_emb, df_stats, on=['subject_id', 'year', 'filename'], how='inner')
    df_final = pd.merge(df_merged_1, df_post_pivot, on=['subject_id', 'year'], how='inner')
    
    print(f"Data Loaded. Valid Subjects: {len(df_final)}")
    return df_final

# ==========================================
# 2. Clustering & PCA
# ==========================================
def run_pca_and_clustering(df, k=4):
    print(f"Running PCA and Clustering (k={k})...")
    
    embedding_cols = [f'emb_{i}' for i in range(64)]
    X = df[embedding_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df['pca_1'] = pca_result[:, 0]
    df['pca_2'] = pca_result[:, 1]
    
    # Run K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled) + 1
    
    return df, pca

# ==========================================
# 3. Visualization: PCA Parameters (Plot 1)
# ==========================================
def plot_pca_param(df, pca_model):
    print("Generating PCA Parameter Plots...")

    params = ['q_d_n', 'q_d', 'q_s_n', 'q_s', 'cost_stop_error', 'cost_time']
    
    # Calculate Variance Ratios
    var_ratios = pca_model.explained_variance_ratio_ * 100
    pc1_label = f"PC 1 ({var_ratios[0]:.1f}%)"
    pc2_label = f"PC 2 ({var_ratios[1]:.1f}%)"
    
    # Share axes to reduce chaos
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, param in enumerate(params):
        if param in df.columns:
            # Scatter with gradient color
            sc = axes[i].scatter(df['pca_1'], df['pca_2'], c=df[param], cmap='mako_r', alpha=0.6, s=8, linewidth=0)
            
            # Use LaTeX Title
            title = PARAM_DISPLAY_NAMES.get(param, param)
            axes[i].set_title(title, fontsize=14)
            
            # Set labels only on outer plots (bottom row and left column)
            if i >= 3: 
                axes[i].set_xlabel(pc1_label, fontsize=12)
            if i % 3 == 0:
                axes[i].set_ylabel(pc2_label, fontsize=12)
            
            # Clean Aesthetics
            sns.despine(ax=axes[i], trim=False)
            
            # Custom Colorbar
            cbar = plt.colorbar(sc, ax=axes[i], pad=0.02)
            cbar.ax.tick_params(labelsize=8)
            cbar.outline.set_visible(False)
        else:
            axes[i].axis('off')
            
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'embed_param.png', dpi=300)
    plt.show()

# ==========================================
# 4. Visualization: PCA Clusters (Plot 2)
# ==========================================
def plot_pca_cluster(df, pca_model):
    print("Generating PCA Cluster Plot...")
    
    # Calculate Variance Ratios
    var_ratios = pca_model.explained_variance_ratio_ * 100
    pc1_label = f"PC 1 ({var_ratios[0]:.1f}%)"
    pc2_label = f"PC 2 ({var_ratios[1]:.1f}%)"
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='pca_1', y='pca_2', hue='cluster', 
                    palette=JOURNAL_PALETTE, alpha=0.7, s=40, legend='full', linewidth=0)
    
    plt.title('PCA Projection (Clusters)', fontsize=16)
    plt.xlabel(pc1_label, fontsize=12)
    plt.ylabel(pc2_label, fontsize=12)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'embed_cluster.png', dpi=300)
    plt.show()

# ==========================================
# 5. Visualization: Dual Radar (Plot 3)
# ==========================================
def plot_dual_radar_profiles(df):
    print("Generating Plot 3: Dual Radar...")
    features_cognitive = ['q_d_n', 'q_d', 'q_s_n', 'q_s', 'cost_stop_error', 'cost_time']
    features_behavior  = ['perc_gs', 'perc_ge', 'perc_gm', 'perc_ss', 'ssrt', 'mrt_gs']
    all_features = features_cognitive + features_behavior

    # Standardization
    df_z = df.copy()
    for col in all_features:
        if col in df_z.columns:
            df_z[col] = (df_z[col] - df_z[col].mean()) / df_z[col].std()
    
    z_means = df_z.groupby('cluster')[all_features].mean()

    def draw_radar(ax, features, title, z_means, display_map=None):
        N = len(features)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        # Labels
        labels = [display_map.get(f, f) if display_map else f for f in features]
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color='black', size=12)
        ax.set_rlabel_position(0)
        
        # Faint grid (essential for reading values, but made subtle)
        ax.yaxis.grid(True, color='grey', alpha=0.2)
        ax.xaxis.grid(True, color='grey', alpha=0.2)
        ax.spines['polar'].set_visible(False) # Remove outer circle
        
        plt.yticks([-1, 0, 1], ["-1", "0", "1"], color="grey", size=8)
        plt.ylim(-1.5, 1.5)
        
        for i, (idx, row) in enumerate(z_means.iterrows()):
            values = row[features].tolist()
            values += values[:1]
            color = JOURNAL_PALETTE[i]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {idx}', color=color)
            ax.fill(angles, values, color=color, alpha=0.1)
        
        ax.set_title(title, size=14, y=1.1, weight='bold')

    fig = plt.figure(figsize=(16, 8))
    
    ax1 = fig.add_subplot(121, polar=True)
    draw_radar(ax1, features_cognitive, "Cognitive Parameters", z_means, PARAM_DISPLAY_NAMES)
    
    ax2 = fig.add_subplot(122, polar=True)
    # Simple formatting for behavioral labels
    beh_display = {f: f.replace('_', ' ').upper() for f in features_behavior}
    draw_radar(ax2, features_behavior, "Behavioral Stats", z_means, beh_display)
    
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title="Cluster", frameon=False)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'embed_radar.png', dpi=300)
    plt.show()

# ==========================================
# 6. Visualization: ADHD Stats (Plot 4)
# ==========================================
def analyze_adhd_stats(df):
    print("Generating Plot 4: ADHD Stats...")
    threshold = 10
    df['high_adhd'] = df['adhd'] >= threshold
    
    summary = df.groupby('cluster').agg({
        'high_adhd': lambda x: np.mean(x) * 100
    }).rename(columns={'high_adhd': 'High ADHD Freq (%)'})
    
    print(summary)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load
    df = load_and_filter_data()
    
    # 2. Run PCA & Clustering
    df, pca_model = run_pca_and_clustering(df, k=4)
    
    # 3. Plot
    plot_pca_param(df, pca_model)
    plot_pca_cluster(df, pca_model)
    plot_dual_radar_profiles(df)
    analyze_adhd_stats(df)
    
    print("Analysis Complete.")