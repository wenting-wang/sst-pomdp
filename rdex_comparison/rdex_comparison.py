import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. Configuration & Paths
# ==========================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT)) 

from core.pomdp import POMDP
from core import simulation
from rdex import RDEX

# Paths
PARAMS_CSV = Path('/Users/w/sst-hdbm-mac/sst-pomdp-main/data/params_posteriors_6p_v1.csv')
RAW_DATA_DIR = Path('/Users/w/Desktop/data/sst_valid_base') 

# PLOS CB Aesthetics
COLOR_OBS = '#404040'      # Dark Gray
COLOR_POMDP = '#008080'    # Teal
COLOR_RDEX = '#D55E00'     # Vermilion

# Dimensions & Global Thinner Styles
FIG_SIZE = (7.5, 4.8)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.0
})
STYLE = {"marker_size": 3.5}

# ==========================================
# 1. Data Generation & Loading
# ==========================================
def generate_pomdp_data(n_subjects=20, sessions_per_sub=5):
    print(f"Loading POMDP params and sampling {n_subjects} subjects, {sessions_per_sub} sessions each...")
    df_params = pd.read_csv(PARAMS_CSV)
    unique_subs = df_params['subject_id'].unique()
    sampled_subs = np.random.choice(unique_subs, min(n_subjects, len(unique_subs)), replace=False)
    
    all_trials = []
    for sub in sampled_subs:
        sub_df = df_params[df_params['subject_id'] == sub]
        p_dict = sub_df.set_index('index' if 'index' in sub_df.columns else 'param')['mean'].to_dict()
        
        fixed = {"inv_temp": 59.93, "q_s_n": 0.22, "rate_stop_trial": 1.0 / 6.0, "q_d_n": 0.1370, "cost_go_missing": 2.2687}
        for k, v in fixed.items():
            if k not in p_dict: p_dict[k] = v

        model = POMDP(**p_dict)
        model.value_iteration_tensor()
        
        for sess_id in range(sessions_per_sub):
            sim_id = f"POMDP_{sub}_{sess_id}"
            session_outcomes = simulation.simu_task(model, verbose=False)
            for t, row in enumerate(session_outcomes):
                all_trials.append({
                    'result': row[0], 'rt_steps': row[1], 'ssd_steps': row[2], 
                    'trial': t+1, 'sim_id': sim_id,
                    'subject_id': sub 
                })
            
    df = pd.DataFrame(all_trials)
    df['rt_ms'] = df['rt_steps'] * 25.0
    df['ssd_ms'] = df['ssd_steps'] * 25.0 
    df['model'] = 'POMDP'
    return df, sampled_subs

def generate_rdex_data(n_sessions=100):
    print(f"Simulating {n_sessions} sessions for RDEX...")
    model = RDEX()
    all_trials = []
    for sess_id in range(n_sessions):
        sim_id = f"RDEX_sess_{sess_id}"
        session_outcomes = model.simu_task()
        for t, row in enumerate(session_outcomes):
            all_trials.append({
                'result': row[0], 'rt_ms': row[1], 'ssd_ms': row[2], 
                'trial': t+1, 'sim_id': sim_id,
                'subject_id': sim_id 
            })
            
    df = pd.DataFrame(all_trials)
    df['model'] = 'RDEX'
    return df

def extract_empirical_data(sid, raw_data_dir):
    file_pattern = os.path.join(raw_data_dir, f"*{sid}*.csv*")
    matched_files = glob.glob(file_pattern)
    
    if not matched_files: return None
        
    df_raw = pd.read_csv(matched_files[0], low_memory=False)
    obs_rows = []
    trial_counter = 1
    sim_id = f"OBS_{sid}"
    
    for _, row in df_raw.iterrows():
        expcon = str(row.get('sst_expcon', ''))
        if expcon not in ['GoTrial', 'VariableStopTrial']: continue
            
        ssd_ms_raw = row.get('sst_ssd_dur', np.nan)
        rt_ms_raw = row.get('sst_primaryrt', np.nan)
        if pd.isna(rt_ms_raw) and 'sst_go_rt' in df_raw.columns:
            rt_ms_raw = row.get('sst_go_rt', np.nan)
            
        is_correct = row.get('sst_choiceacc', np.nan)
        is_sst_go_resp = row.get('sst_go_resp', np.nan)
        
        if expcon == 'GoTrial':
            result = 'GM' if pd.isna(is_sst_go_resp) or is_sst_go_resp == 0 else ('GS' if is_correct == 1.0 else 'GE')
        else:
            result = 'SS' if row.get('sst_inhibitacc') == 1.0 else 'SE'
                        
        obs_rows.append({
            'result': result, 'rt_ms': rt_ms_raw, 'ssd_ms': ssd_ms_raw, 
            'trial': trial_counter, 'sim_id': sim_id,
            'subject_id': sid 
        })
        trial_counter += 1
        
    return pd.DataFrame(obs_rows)

def load_observed_data(sampled_subs):
    all_obs = [extract_empirical_data(sid, RAW_DATA_DIR) for sid in sampled_subs]
    all_obs = [df for df in all_obs if df is not None and not df.empty]
    if all_obs:
        df_obs = pd.concat(all_obs, ignore_index=True)
        df_obs['model'] = 'Observation'
        return df_obs
    return None

# ==========================================
# 2. Plotting Functions
# ==========================================
def clean_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_rt_distribution(ax, df, res_type, name, color, bins_ms):
    subs = df[df['result'] == res_type]
    if subs.empty: return
    
    if name == 'Observation':
        ax.hist(subs['rt_ms'].dropna(), bins=bins_ms, density=True, 
                color=color, alpha=0.35, edgecolor='none', zorder=1)
    else:
        densities = [np.histogram(grp['rt_ms'].dropna(), bins=bins_ms, density=True)[0] 
                     for _, grp in subs.groupby('sim_id')]
        if not densities: return
        
        mean_dens = np.mean(densities, axis=0)
        std_dens = np.std(densities, axis=0)
        x_centers = (bins_ms[:-1] + bins_ms[1:]) / 2
        
        smooth_mean = gaussian_filter1d(mean_dens, 1.0)
        smooth_std = gaussian_filter1d(std_dens, 1.0)
        
        ax.plot(x_centers, smooth_mean, color=color, lw=1.0, alpha=0.9, zorder=2)
        ax.fill_between(x_centers, smooth_mean - smooth_std, smooth_mean + smooth_std, 
                        color=color, alpha=0.2, zorder=2, edgecolor='none')

def plot_comparison(df_pomdp, df_rdex, df_obs=None):
    fig, axs = plt.subplots(2, 3, figsize=FIG_SIZE, constrained_layout=True)
    datasets = [('POMDP', df_pomdp, COLOR_POMDP), ('RDEX', df_rdex, COLOR_RDEX)]
    if df_obs is not None:
        datasets.insert(0, ('Observation', df_obs, COLOR_OBS))

    df_all = pd.concat([df for _, df, _ in datasets], ignore_index=True)
    palette = {name: color for name, _, color in datasets}

    # Plot 1: Outcome Proportions
    ax = axs[0, 0]
    outcomes = ['GS', 'GE', 'GM', 'SS', 'SE']
    prop_data = []
    
    for name, df, _ in datasets:
        session_props = []
        for (sim_id, sub_id), grp in df.groupby(['sim_id', 'subject_id']):
            counts = grp['result'].value_counts(normalize=True)
            for o in outcomes:
                session_props.append({'subject_id': sub_id, 'Outcome': o, 'Proportion': counts.get(o, 0)})
        
        df_sess = pd.DataFrame(session_props)
        df_sub = df_sess.groupby(['subject_id', 'Outcome'])['Proportion'].mean().reset_index()
        df_sub['Model'] = name
        prop_data.append(df_sub)
                
    sns.barplot(data=pd.concat(prop_data, ignore_index=True), x='Outcome', y='Proportion', hue='Model', 
                palette=palette, ax=ax, capsize=0.08, errorbar='se', alpha=0.8, 
                err_kws={'linewidth': 0.8}, linewidth=0,
                order=['GS', 'GE', 'GM', 'SS', 'SE'])  # 强制按指定顺序绘图
    
    ax.set_title("Outcome Proportions")
    ax.set_ylabel("Proportions")
    ax.set_xlabel("")
    ax.get_legend().remove()
    clean_spines(ax)

    # Plot 2 & 3: RT Distributions
    bins_ms = np.linspace(0, 1000, 41)
    for col_idx, res_type in enumerate(['GS', 'SE']):
        ax = axs[0, col_idx + 1]
        for name, df, color in datasets:
            plot_rt_distribution(ax, df, res_type, name, color, bins_ms)
            
        # ax.set_title(f"{res_type} RT Distribution")
        ax.set_title(f"{'Go Success' if res_type == 'GS' else 'Stop Error'} RT Distribution")
        ax.set_xlabel("RT (ms)")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1000)
        ax.set_ylim(bottom=0)  # 强制 Y 轴从 0 开始
        clean_spines(ax)

    # Plot 4: SSD Staircase Trajectory
    ax = axs[1, 0]
    if df_obs is not None:
        ax.scatter(df_obs['trial'], df_obs['ssd_ms'], color=COLOR_OBS, s=1.5, alpha=0.3, zorder=1)
        
    for name, df, color in datasets:
        if name == 'Observation': continue
        df_sim = df.copy()
        df_sim['ssd_filled'] = df_sim.groupby('sim_id')['ssd_ms'].ffill()
        for i, (_, sub) in enumerate(df_sim.groupby('sim_id')):
            if i < 3: 
                ax.step(sub['trial'], sub['ssd_filled'], color=color, alpha=0.6, lw=0.8, where='post', zorder=2)
                
    ax.set_title("SSD Staircases")
    ax.set_xlabel("Trial")
    ax.set_ylabel("SSD (ms)")
    ax.set_ylim(-25, 850)
    ax.set_xticks([1, 120, 240, 360])
    clean_spines(ax)

    def prepare_lineplot_data(df_full, filter_col, value_col):
        plot_df = []
        for name in df_full['model'].unique():
            df_model = df_full[df_full['model'] == name].copy()
            df_filtered = df_model.dropna(subset=['ssd_ms', filter_col]).copy()
            df_filtered['ssd_bin'] = (np.round(df_filtered['ssd_ms'] / 25.0) * 25.0)
            
            sub_agg = df_filtered.groupby(['subject_id', 'ssd_bin'])[value_col].mean().reset_index()
            
            bin_counts = sub_agg['ssd_bin'].value_counts()
            valid_bins = bin_counts[bin_counts >= 3].index 
            sub_agg = sub_agg[sub_agg['ssd_bin'].isin(valid_bins)]
            sub_agg['Model'] = name
            plot_df.append(sub_agg)
            
        return pd.concat(plot_df, ignore_index=True) if plot_df else pd.DataFrame()

    # Plot 5: Inhibition Function P(SS)
    ax = axs[1, 1]
    df_stop = df_all[df_all['result'].isin(['SS', 'SE'])].copy()
    df_stop['is_ss'] = (df_stop['result'] == 'SS').astype(float)
    plot_data_inhibition = prepare_lineplot_data(df_stop, 'result', 'is_ss')
    
    sns.lineplot(data=plot_data_inhibition, x='ssd_bin', y='is_ss', hue='Model', 
                 palette=palette, marker='o', errorbar='se', err_style='band', 
                 err_kws={'alpha': 0.2, 'linewidth': 0}, ax=ax, 
                 markersize=STYLE['marker_size'], linewidth=1.0, alpha=0.9)
                 
    ax.set_title("Stop Accuracy across SSD")
    ax.set_xlabel("SSD (ms)")
    ax.set_ylabel("P(Stop Success)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, 200, 400, 600, 800])
    ax.get_legend().remove()
    clean_spines(ax)

    # Plot 6: SE RT across SSD
    ax = axs[1, 2]
    df_se = df_all[df_all['result'] == 'SE'].copy()
    plot_data_se_rt = prepare_lineplot_data(df_se, 'rt_ms', 'rt_ms')
    
    sns.lineplot(data=plot_data_se_rt, x='ssd_bin', y='rt_ms', hue='Model', 
                 palette=palette, marker='o', errorbar='se', err_style='band', 
                 err_kws={'alpha': 0.2, 'linewidth': 0}, ax=ax, 
                 markersize=STYLE['marker_size'], linewidth=1.0, alpha=0.9)
                 
    ax.set_title("Stop Error RT across SSD")
    ax.set_xlabel("SSD (ms)")
    ax.set_ylabel("Mean RT (ms)")
    ax.set_xticks([0, 200, 400, 600, 800])
    ax.get_legend().remove()
    clean_spines(ax)

    # Global Legend
    legend_elements = [Patch(facecolor=color, label=name, alpha=0.8, edgecolor='none') 
                       for name, _, color in datasets]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.06), 
               ncol=3, frameon=False, handlelength=1.2, handleheight=1.2)

    plt.savefig('rdex_comparison.png', dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    print("Success: Plot saved as 'rdex_comparison.png'")

# ==========================================
# Main Execution
# ==========================================
if __name__ == '__main__':
    df_pomdp, sampled_subs = generate_pomdp_data(n_subjects=10, sessions_per_sub=5)
    df_rdex = generate_rdex_data(n_sessions=50)
    df_obs = load_observed_data(sampled_subs)
    
    print("\nGenerating comparative plots...")
    plot_comparison(df_pomdp, df_rdex, df_obs)