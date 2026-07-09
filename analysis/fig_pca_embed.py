import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.offline import plot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse

# =============================================================================
# 0. ARGUMENT PARSING (DEMO vs PROD MODE)
# =============================================================================
parser = argparse.ArgumentParser(description="Generate PCA Embeddings Plot")
parser.add_argument(
    '--prod', 
    action='store_true', 
    help='Run in PROD mode using local raw data. Usage: run `python script.py --prod`. Defaults to DEMO mode using example data if omitted.'
)
args, unknown = parser.parse_known_args()

DEMO_MODE = not args.prod

# =============================================================================
# 1. CONFIGURATION & CONSTANTS
# =============================================================================

# --- Directory Setup ---
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if DEMO_MODE:
    print(">>> RUNNING IN DEMO MODE (Using example data) <<<")
    OUT_DIR = PROJECT_ROOT / "outputs_example"
    FILE_STATS = PROJECT_ROOT / "data" / "example_clinical_behavior.csv"
    FILE_EMBEDDINGS = PROJECT_ROOT / "outputs_example" / "tesbi" / "model_posteriors" / "example_embeddings_6p_v1.csv"
    FILE_POSTERIOR = PROJECT_ROOT / "outputs_example" / "tesbi" / "model_posteriors" / "example_posteriors_6p_v1.csv"
else:
    print(">>> RUNNING IN PROD MODE (Using local raw data) <<<")
    OUT_DIR = PROJECT_ROOT / "outputs"
    FILE_EMBEDDINGS = Path("/Users/w/sst-pomdp/outputs/tesbi/model_posteriors/embeddings_e2e_tesbi_e2e_6p_v1.csv")
    FILE_POSTERIOR = Path("/Users/w/sst-pomdp/outputs/tesbi/model_posteriors/params_posteriors_6p_v1.csv")
    FILE_STATS = Path("/Users/w/sst-pomdp/data/clinical_behavior.csv")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Output Files ---
PCA_2D = OUT_DIR / "fig_pca_embed_2d.png"
PCA_3D = OUT_DIR / "fig_pca_embed_3d.html"

# ==========================================
# 2. PLOT CONFIGURATIONS
# ==========================================
# LaTeX formatted names for Matplotlib (2D)
DISPLAY_NAMES_2D = {
    'q_d': r"$\chi$ (Go Precision)",
    'q_s': r"$\delta$ (Stop Precision)",
    'cost_stop_error': r"$c_{\mathrm{se}}$ (Stop Error Cost)",
    'cost_time': r"$c_{\mathrm{t}}$ (Time Cost)",
    'tau': r"$\tau$ (Non-decision Time)",
    'cost_go_error': r"$c_{\mathrm{ge}}$ (Go Error Cost)",
    'adhd': 'ADHD Score',
    'mrt_gs': 'GSRT',
    'ssrt': 'SSRT',
}

# Descriptive names for Plotly (3D Dropdowns)
DISPLAY_NAMES_3D = {
    'q_d': r"$\chi$ (Go Precision)",
    'q_s': r"$\delta$ (Stop Precision)",
    'cost_stop_error': r"$c_{\mathrm{se}}$ (Stop Error Cost)",
    'cost_time': r"$c_{\mathrm{t}}$ (Time Cost)",
    'tau': r"$\tau$ (Non-decision Time)",
    'cost_go_error': r"$c_{\mathrm{ge}}$ (Go Error Cost)",
    'adhd': 'ADHD Score',
    'mrt_gs': 'GSRT',
    'ssrt': 'SSRT',
}

TARGETS = [
    'q_d', 'q_s','cost_stop_error', 
    'cost_go_error', 'cost_time', 'tau',
    'adhd', 'mrt_gs', 'ssrt'
]

# PLOS CB Style Settings (Matched with param_ana.py)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10         
plt.rcParams['axes.labelsize'] = 10    
plt.rcParams['xtick.labelsize'] = 9   
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8   
plt.rcParams['axes.titlesize'] = 11


# ==========================================
# 1. SHARED DATA LOADING
# ==========================================
def load_data():
    """
    Loads embeddings, behavioral stats, and posterior summaries.
    Merges them into a single DataFrame.
    """
    
    # Verify files exist before loading
    missing_files = []
    for filepath in [FILE_EMBEDDINGS, FILE_STATS, FILE_POSTERIOR]:
        if not filepath.exists():
            missing_files.append(filepath.name)
            
    if missing_files:
        raise FileNotFoundError(f"Missing required data files in the 'data' folder: {', '.join(missing_files)}")

    # Load Files
    df_emb = pd.read_csv(FILE_EMBEDDINGS)
    df_stats = pd.read_csv(FILE_STATS)
    df_post = pd.read_csv(FILE_POSTERIOR)

    df_emb['subject_id'] = df_emb['subject_id'].astype(str).str.replace(r'^NDAR_', '', regex=True)
    df_stats['subject_id'] = df_stats['subject_id'].astype(str).str.replace(r'^NDAR_', '', regex=True)
    df_post['subject_id'] = df_post['subject_id'].astype(str).str.replace(r'^NDAR_', '', regex=True)

    # Type Conversion for merging consistency
    df_emb['year'] = df_emb['year'].astype(str)
    df_stats['year'] = df_stats['year'].astype(str)
    df_post['subject_year'] = df_post['subject_year'].astype(str)

    # Pivot Posterior Parameters
    df_post_pivot = df_post.pivot_table(
        index=['subject_id', 'subject_year'], 
        columns='index', 
        values='mean'
    ).reset_index()
    df_post_pivot.rename(columns={'subject_year': 'year'}, inplace=True)

    # Merge Sequence
    # 1. Embeddings + Behavior
    df_merged = pd.merge(df_emb, df_stats, on=['subject_id', 'year'], how='inner')
    # 2. + Posterior Parameters
    df_final = pd.merge(df_merged, df_post_pivot, on=['subject_id', 'year'], how='inner')
    
    print(f"Data Loaded. Valid Subjects: {len(df_final)}")
    return df_final

def get_pca_coordinates(df, n_components=3):
    """
    Helper to run PCA on embedding columns and return the transformed data.
    """
    # Adjust this if your embeddings have a different naming convention (e.g., emb_0 to emb_63)
    embedding_cols = [f'emb_{i}' for i in range(128)]
    
    # Ensure all embedding columns exist in the DataFrame
    missing_cols = [col for col in embedding_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing embedding columns in the dataset. Expected columns like 'emb_0' to 'emb_63'.")

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
    
    # Run PCA
    pca_result, var_ratios = get_pca_coordinates(df, n_components=2)
    df['pca_1'] = pca_result[:, 0]
    df['pca_2'] = pca_result[:, 1]
    
    pc1_label = f"PC 1 ({var_ratios[0]:.1f}%)"
    pc2_label = f"PC 2 ({var_ratios[1]:.1f}%)"

    # Plotting Grid
    # SIZE CHANGE: 7.5 inches width (PLOS CB standard), 7.5 inches height for 3x3 square-ish
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 6.5), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, var in enumerate(TARGETS):
        ax = axes[i]
        if var in df.columns:
            if var == 'adhd':
                vmin, vmax = 0, 14
            else:
                # Robust scaling for other variables
                vmin, vmax = df[var].quantile([0.01, 0.99])
            
            # SCATTER SIZE CHANGE: s=8 -> s=3 for smaller plot
            sc = ax.scatter(
                df['pca_1'], df['pca_2'], 
                c=df[var], cmap='RdBu_r', 
                alpha=0.8, s=3, linewidth=0, 
                vmin=vmin, vmax=vmax
            )
            
            # Styling
            title = DISPLAY_NAMES_2D.get(var, var)
            ax.set_title(title) # Font size controlled by rcParams
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Labels (only on edges)
            if i >= 6: ax.set_xlabel(pc1_label)
            if i % 3 == 0: ax.set_ylabel(pc2_label)
            
            # Colorbar
            cbar = plt.colorbar(sc, ax=ax, pad=0.03)
            cbar.ax.tick_params(labelsize=8)
            cbar.outline.set_visible(False)
        else:
            ax.text(0.5, 0.5, f"{var} missing", ha='center')
            ax.axis('off')

    plt.tight_layout()
    # Adjust spacing similar to param_ana
    plt.subplots_adjust(wspace=0.3, hspace=0.3) 
    
    save_path = PCA_2D
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"2D Plot saved to: {save_path}")

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

    save_path = PCA_3D
    plot(fig, filename=str(save_path), auto_open=False)
    print(f"3D Plot saved to: {save_path}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        # 1. Load Data once
        df_data = load_data()
        
        # 2. Run Visualization Parts
        run_pca_2d(df_data.copy())
        # Uncomment the line below to also generate the 3D interactive plot
        # run_pca_3d(df_data.copy())
        
    except Exception as e:
        print(f"An error occurred: {e}")