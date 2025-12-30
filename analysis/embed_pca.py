import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent)) 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.offline import plot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. CONFIGURATION & CONSTANTS
# ==========================================
BASE_DIR = Path('/kyb/agpd/wwang/data/')
OUT_DIR = Path('/kyb/agpd/wwang/sst-pomdp/outputs/')

# LaTeX formatted names for Matplotlib (2D)
DISPLAY_NAMES_2D = {
    'q_d_n': r"$\chi'$",
    'q_d': r"$\chi$",
    'q_s_n': r"$\sigma'$",
    'q_s': r"$\sigma$",
    'cost_stop_error': r"$c_{\mathrm{se}}$",
    'cost_time': r"$c_t$",
    'adhd': 'ADHD Score',
    'mrt_gs': 'GSRT',
    'ssrt': 'SSRT',
}

# Descriptive names for Plotly (3D Dropdowns)
DISPLAY_NAMES_3D = {
    'q_d_n': "Chi' (Go Noise)",
    'q_d': "Chi (Go Precision)",
    'q_s_n': "Sigma' (Stop Noise)",
    'q_s': "Sigma (Stop Precision)",
    'cost_stop_error': "Stop Cost (C_se)",
    'cost_time': "Time Cost (C_t)",
    'adhd': 'ADHD Score',
    'mrt_gs': 'GSRT',
    'ssrt': 'SSRT',
}

TARGETS = [
    'q_d_n', 'q_d', 'q_s_n', 'q_s', 
    'cost_stop_error', 'cost_time',
    'adhd', 'mrt_gs', 'ssrt'
]

# ==========================================
# 1. SHARED DATA LOADING
# ==========================================
def load_data():
    """
    Loads embeddings, behavioral stats, and posterior summaries.
    Merges them into a single DataFrame.
    """
    print(f"Loading data from {OUT_DIR}...")
    
    # Load Files
    df_emb = pd.read_csv(OUT_DIR / 'embeddings.csv')
    df_stats = pd.read_csv(OUT_DIR / 'stats_beh_full.csv')
    df_post = pd.read_csv(OUT_DIR / 'posterior_summary.csv')

    # Type Conversion
    df_emb['year'] = df_emb['year'].astype(str)
    df_stats['year'] = df_stats['year'].astype(str)
    df_post['subject_year'] = df_post['subject_year'].astype(str)

    # Pivot Posterior Parameters
    df_post_pivot = df_post.pivot_table(
        index=['subject_id', 'subject_year'], 
        columns='param', 
        values='mean'
    ).reset_index()
    df_post_pivot.rename(columns={'subject_year': 'year'}, inplace=True)

    # Merge Sequence
    # 1. Embeddings + Behavior
    df_merged = pd.merge(df_emb, df_stats, on=['subject_id', 'year', 'filename'], how='inner')
    # 2. + Posterior Parameters
    df_final = pd.merge(df_merged, df_post_pivot, on=['subject_id', 'year'], how='inner')
    
    print(f"Data Loaded. Valid Subjects: {len(df_final)}")
    return df_final

def get_pca_coordinates(df, n_components=3):
    """
    Helper to run PCA on embedding columns and return the transformed data.
    """
    embedding_cols = [f'emb_{i}' for i in range(64)]
    X = df[embedding_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)
    var_ratios = pca.explained_variance_ratio_ * 100
    
    return pca_result, var_ratios

# ==========================================
# 2. PART 1: 2D PCA (Matplotlib/Seaborn)
# ==========================================
def run_pca_2d(df):
    print("Generating 2D PCA Plot...")
    
    # Settings
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("white")
    
    # Run PCA
    pca_result, var_ratios = get_pca_coordinates(df, n_components=2)
    df['pca_1'] = pca_result[:, 0]
    df['pca_2'] = pca_result[:, 1]
    
    pc1_label = f"PC 1 ({var_ratios[0]:.1f}%)"
    pc2_label = f"PC 2 ({var_ratios[1]:.1f}%)"

    # Plotting Grid
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, var in enumerate(TARGETS):
        ax = axes[i]
        if var in df.columns:
            # Robust scaling for color mapping
            vmin, vmax = df[var].quantile([0.01, 0.999])
            
            sc = ax.scatter(
                df['pca_1'], df['pca_2'], 
                c=df[var], cmap='RdYlBu_r', 
                alpha=0.7, s=8, linewidth=0, 
                vmin=vmin, vmax=vmax
            )
            
            # Styling
            title = DISPLAY_NAMES_2D.get(var, var)
            ax.set_title(title, fontsize=14)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Labels (only on edges)
            if i >= 6: ax.set_xlabel(pc1_label, fontsize=12)
            if i % 3 == 0: ax.set_ylabel(pc2_label, fontsize=12)
            
            # Colorbar
            cbar = plt.colorbar(sc, ax=ax, pad=0.02)
            cbar.ax.tick_params(labelsize=9)
            cbar.outline.set_visible(False)
        else:
            ax.text(0.5, 0.5, f"{var} missing", ha='center')
            ax.axis('off')

    plt.tight_layout()
    save_path = OUT_DIR / 'embed_pca_2d.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"2D Plot saved to: {save_path}")
    # plt.show() # Uncomment if you want to see the window

# ==========================================
# 3. PART 2: 3D PCA (Plotly)
# ==========================================
def run_pca_3d(df):
    print("Generating 3D PCA Interactive Plot...")

    # Run PCA
    pca_result, var_ratios = get_pca_coordinates(df, n_components=3)
    df['pca_1'] = pca_result[:, 0]
    df['pca_2'] = pca_result[:, 1]
    df['pca_3'] = pca_result[:, 2]
    
    total_var = sum(var_ratios)

    # Plot Setup
    fig = go.Figure()
    
    # Use first valid target as default
    valid_targets = [t for t in TARGETS if t in df.columns]
    default_var = valid_targets[0] if valid_targets else None

    if not default_var:
        print("No valid targets found for 3D plot.")
        return

    # Add Default Trace
    fig.add_trace(go.Scatter3d(
        x=df['pca_1'], y=df['pca_2'], z=df['pca_3'],
        mode='markers',
        marker=dict(
            size=2,
            color=df[default_var],
            colorscale='RdYlBu_r',
            opacity=0.9,
            colorbar=dict(title=DISPLAY_NAMES_3D.get(default_var, default_var), len=0.6)
        ),
        text=[f"ID: {s}<br>Val: {v:.2f}" for s, v in zip(df['subject_id'], df[default_var])],
        hoverinfo='text'
    ))

    # Create Dropdown Buttons
    buttons = []
    for var in valid_targets:
        label = DISPLAY_NAMES_3D.get(var, var)
        buttons.append(dict(
            label=label,
            method="restyle",
            args=[{
                'marker.color': [df[var]],
                'marker.colorscale': 'RdYlBu_r', 
                'marker.colorbar.title': label,
                'text': [[f"ID: {s}<br>{label}: {v:.2f}" for s, v in zip(df['subject_id'], df[var])]]
            }]
        ))

    # Layout Styles
    no_axis = dict(showgrid=False, zeroline=False, showbackground=False, title='', showticklabels=False)
    
    fig.update_layout(
        title=dict(text=f"3D Latent Policy Space (Total Var: {total_var:.1f}%)", x=0.5, y=0.95, xanchor='center'),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(xaxis=no_axis, yaxis=no_axis, zaxis=no_axis, aspectmode='cube'),
        updatemenus=[dict(
            active=0, 
            buttons=buttons,
            x=0.02, y=0.98,
            xanchor='left', yanchor='top',
            bgcolor='rgba(255,255,255,0.9)'
        )],
        paper_bgcolor='white'
    )

    save_path = OUT_DIR / 'embed_pca_3d.html'
    plot(fig, filename=str(save_path), auto_open=False)
    print(f"3D Plot saved to: {save_path}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load Data once
    df_data = load_data()
    
    # 2. Run Visualization Parts
    run_pca_2d(df_data.copy())
    run_pca_3d(df_data.copy())