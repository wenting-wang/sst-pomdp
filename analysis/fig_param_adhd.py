import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import argparse

# =============================================================================
# 0. ARGUMENT PARSING (DEMO vs PROD MODE)
# =============================================================================
parser = argparse.ArgumentParser(description="Generate ADHD vs Parameters Ridgeline Plot")
parser.add_argument(
    '--prod', 
    action='store_true', 
    help='Run in PROD mode using local raw data. Usage: run `python script.py --prod`. Defaults to DEMO mode using example data if omitted.'
)
args, unknown = parser.parse_known_args()

DEMO_MODE = not args.prod

# =============================================================================
# 1. CONFIGURATION & PATHS
# =============================================================================
try:
    current_dir = Path(__file__).resolve().parent
except NameError:
    current_dir = Path().resolve()

repo_root = current_dir.parent

if DEMO_MODE:
    print(">>> RUNNING IN DEMO MODE (Using example data) <<<")
    OUT_DIR = repo_root / 'outputs'
    CLIN_CSV = repo_root / 'data' / 'example_clinical_behavior.csv'
    PARAMS_CSV = repo_root / 'data' / 'example_params_posteriors.csv'
else:
    print(">>> RUNNING IN PROD MODE (Using local raw data) <<<")
    OUT_DIR = repo_root / 'outputs'
    CLIN_CSV = Path("/Users/w/sst-hdbm-mac/sst-pomdp-main/data/clinical_behavior.csv")
    PARAMS_CSV = Path("/Users/w/sst-hdbm-mac/sst-pomdp-main/data/params_posteriors_6p_v1.csv")

OUT_DIR.mkdir(parents=True, exist_ok=True) 

# =============================================================================
# 2. DATA LOADING & MERGING
# =============================================================================
df_clin = pd.read_csv(CLIN_CSV)
df_clin['subject_id_full'] = 'NDAR_' + df_clin['subject_id'].astype(str)

df_params = pd.read_csv(PARAMS_CSV)

df_params_wide = df_params.pivot(index='subject_id', columns='index', values='50%').reset_index()

df_merged = pd.merge(df_clin, df_params_wide, left_on='subject_id_full', right_on='subject_id')

# =============================================================================
# # 3. Clinical Bins (Refined for Extreme Tails)
# =============================================================================
bins = [-1, 0, 3, 7, 15]
raw_labels = ['0', '1-3', '4-7', '$\geq$8']
df_merged['adhd_bin_raw'] = pd.cut(df_merged['adhd'], bins=bins, labels=raw_labels)

total_n = len(df_merged['adhd_bin_raw'].dropna())
label_mapping = {}
for l in raw_labels:
    count = sum(df_merged['adhd_bin_raw'] == l)
    pct = (count / total_n) * 100
    label_mapping[l] = f"{l}\n({pct:.1f}%)"

labels = [label_mapping[l] for l in raw_labels]
df_merged['adhd_bin'] = df_merged['adhd_bin_raw'].map(label_mapping)

# =============================================================================
# 4. PLOS Print-Ready Settings (2x3 Layout)
# =============================================================================
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42

params = [
    'q_d', 'cost_stop_error', 'cost_time',   
    'q_s', 'cost_go_error', 'tau'            
]

titles = [
    r'$\chi$ (Go Precision)', 
    r'$c_{\text{se}}$ (Stop Error Cost)', 
    r'$c_t$ (Time Cost)', 
    r'$\delta$ (Stop Precision)', 
    r'$c_{\text{ge}}$ (Go Error Cost)', 
    r'$\tau$ (Non-decision Time)'
]

colors = {'Female': "#7f4189", 'Male': "#ce6802"}

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.5))
axes = axes.flatten()

# =============================================================================
# 5. Ridgeline Generation (Viewport Clipping + MEAN Anchor)
# =============================================================================
for idx, p in enumerate(params):
    ax = axes[idx]
    
    if p in ['q_d', 'q_s']:
        p_min = df_merged[p].min()
        p_max = df_merged[p].max()
        pad = (p_max - p_min) * 0.15
        
        view_min = max(0.0, p_min - pad)
        view_max = min(1.0, p_max + pad)
        
    else:
        p_min = df_merged[p].quantile(0.00)
        p_max = df_merged[p].quantile(0.99)
        pad = (p_max - p_min) * 0.15
        
        view_min = max(0.0, p_min - pad)
        view_max = p_max + pad
    
    x_eval = np.linspace(p_min - pad*2, p_max + pad*2, 1000)
    
    mask = (x_eval >= view_min) & (x_eval <= view_max)
    x_plot = x_eval[mask]
    
    max_h = 0
    kdes = {}
    means = {}
    
    for g in labels:
        kdes[g] = {}
        means[g] = {}
        for s in ['Female', 'Male']:
            data = df_merged[(df_merged['adhd_bin'] == g) & (df_merged['sex'] == s)][p].dropna()
            if len(data) > 2:
                kde = gaussian_kde(data, bw_method=0.4)
                y_plot = kde(x_eval)[mask]
                max_h = max(max_h, y_plot.max())
                kdes[g][s] = y_plot
                means[g][s] = data.mean() 
                
    spacing = max_h * 0.65 
    
    for i, g in enumerate(labels):
        offset = (len(labels) - 1 - i) * spacing  
        z_order = i + 2 
        
        ax.axhline(offset, color='lightgray', linewidth=0.5, zorder=z_order - 0.5)
        
        if idx % 3 == 0:
            text_x = view_min - (view_max - view_min) * 0.05
            ax.text(text_x, offset + spacing * 0.15, g, 
                    ha='right', va='bottom', fontsize=9, color='black', zorder=10)
            
            if i == 0:
                ax.text(text_x, offset + spacing * 1.1, "ADHD Scores", 
                        ha='right', va='bottom', fontsize=9, fontweight='bold', color='black', zorder=10)
        
        y_dict = {}
        for s in ['Female', 'Male']:
            if s in kdes[g]:
                y = kdes[g][s]
                y_dict[s] = y
                y_drawn = y + offset
                ax.fill_between(x_plot, offset, y_drawn, facecolor=colors[s], alpha=0.75, zorder=z_order)
                
        if 'Female' in y_dict and 'Male' in y_dict:
            y_inter = np.minimum(y_dict['Female'], y_dict['Male']) + offset
            ax.fill_between(x_plot, offset, y_inter, facecolor="#dedede", alpha=0.95, zorder=z_order + 0.05)
            
        for s in ['Female', 'Male']:
            if s in kdes[g]:
                y_drawn = kdes[g][s] + offset
                ax.plot(x_plot, y_drawn, color='w', linewidth=0.8, zorder=z_order + 0.1)
                
                mean_val = means[g][s]
                if view_min <= mean_val <= view_max: 
                    mean_idx = np.abs(x_plot - mean_val).argmin()
                    mean_y = y_drawn[mean_idx]
                    if p in ['q_d', 'cost_stop_error', 'cost_go_error', 'cost_time']:
                        ax.plot([mean_val, mean_val], [offset, mean_y], color=colors[s], 
                                linestyle='--', linewidth=0.8, zorder=z_order + 0.2)
                  

    ax.set_title(titles[idx], fontsize=10, fontweight='normal', pad=10)
    ax.set_yticks([]) 
    ax.set_xlabel('') 
    
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('gray')
    
    ax.set_xlim(view_min, view_max)
    
    ax.tick_params(axis='x', labelsize=8, color='gray', width=0.5, length=3, pad=3)
    
    if p in ['q_d', 'q_s']:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    elif p == 'tau':
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# =============================================================================
# 6. Global Legend & Layout Adjustment
# =============================================================================
legend_elements = [
    Patch(facecolor=colors['Female'], alpha=0.75, edgecolor='w', label='Female'),
    Patch(facecolor=colors['Male'], alpha=0.75, edgecolor='w', label='Male'),
    Patch(facecolor='#dedede', alpha=0.95, edgecolor='none', label='Overlap')
]

fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.01), 
           ncol=3, fontsize=9, frameon=False)

plt.subplots_adjust(top=0.90, bottom=0.08, left=0.15, right=0.95, hspace=0.3, wspace=0.20)

output_filepath = OUT_DIR / 'fig_param_adhd.png'
plt.savefig(output_filepath, dpi=300, bbox_inches='tight', transparent=False)
print(f"Plot successfully saved to: {output_filepath}")