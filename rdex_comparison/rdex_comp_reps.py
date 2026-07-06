import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
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
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = Path('/Users/w/Desktop/data/sst_valid_base') 
POMDP_PARAMS_CSV = Path('/Users/w/sst-hdbm-mac/sst-pomdp-main/data/params_posteriors_6p_v1.csv')
RDEX_PARAMS_CSV = CURRENT_DIR / 'fitted_3_subjects_params.csv'

SUBJECTS = [
    "NDAR_INVRUVFBX1L",
    "NDAR_INVPWL6YB2C", 
    "NDAR_INVRH6B02GB", 
]


N_SESSIONS = 5
N_TRACES_PLOT = 3

# PLOS CB Aesthetics
COLOR_OBS = '#404040'      # Dark Gray
COLOR_POMDP = '#008080'    # Teal
COLOR_RDEX = '#D55E00'     # Vermilion

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "tick_fontsize": 8,
    "legend_fontsize": 9,   
    "panel_label_size": 12,
    "marker_size": 15,     
    "axis_lw": 0.8,
}
plt.rcParams.update({
    'font.family': STYLE["font.family"],
    'font.sans-serif': STYLE["font.sans-serif"],
})

# ==========================================
# 1. Data Loading & Simulation
# ==========================================
def extract_empirical_data(sid, raw_data_dir):
    clean_sid = sid.replace('NDAR_', '')
    file_pattern = os.path.join(raw_data_dir, f"*{clean_sid}*.csv*")
    matched_files = glob.glob(file_pattern)
    
    if not matched_files:
        return None
        
    df_raw = pd.read_csv(matched_files[0], low_memory=False)
    obs_rows = []
    trial_counter = 1
    
    for _, row in df_raw.iterrows():
        expcon = str(row.get('sst_expcon', ''))
        if expcon not in ['GoTrial', 'VariableStopTrial']: 
            continue
            
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
            'result': result, 
            'rt_ms': rt_ms_raw, 
            'ssd_ms': ssd_ms_raw, 
            'trial': trial_counter,
            'sim_id': 0 
        })
        trial_counter += 1
        
    return pd.DataFrame(obs_rows)

def get_pomdp_params(params_file, subject_id, year='baseline'):
    df = pd.read_csv(params_file)
    clean_id = str(subject_id).replace('NDAR_', '')
    df['clean_id'] = df['subject_id'].astype(str).str.replace('NDAR_', '', regex=False)
    
    year_col = 'subject_year' if 'subject_year' in df.columns else 'year'
    if year_col in df.columns:
        df[year_col] = df[year_col].astype(str)
        mask = (df["clean_id"] == clean_id) & (df[year_col] == str(year))
    else:
        mask = (df["clean_id"] == clean_id)
        
    if not mask.any():
        return None
    
    idx_col = 'index' if 'index' in df.columns else 'param'
    p_dict = df.loc[mask, [idx_col, "mean"]].set_index(idx_col)["mean"].to_dict()
    
    fixed = {"inv_temp": 59.93, "q_s_n": 0.22, "rate_stop_trial": 1.0 / 6.0, "q_d_n": 0.1370, "cost_go_missing": 2.2687}
    for k, v in fixed.items():
        if k not in p_dict: p_dict[k] = v
    return p_dict

def get_rdex_params(csv_path, sid):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    sub_df = df[df['subject'] == sid]
    if sub_df.empty:
        clean_sid = str(sid).replace('NDAR_', '')
        sub_df = df[df['subject'].str.replace('NDAR_', '', regex=False) == clean_sid]
        
    if sub_df.empty:
        return None
        
    row = sub_df.iloc[0]
    return {
        't0': row['t0'], 'B': row['B'], 'v_plus': row['vT'], 'v_minus': row['vF'],
        'v0': row['v0'], 'g': row['k'], 'mu': row['mu'], 'sigma': row['sigma'],
        'tau': row['tau'], 'p_gf': row['gf'], 'p_tf': row['tf']
    }

def run_multi_pomdp(params, n_sessions=N_SESSIONS):
    model = POMDP(**params)
    model.value_iteration_tensor()
    all_trials = []
    
    for sess_id in range(n_sessions):
        outcomes = simulation.simu_task(model, verbose=False)
        for t, row in enumerate(outcomes):
            # Handle None values for Go trials (no SSD) or omissions (no RT)
            rt_val = row[1] * 25.0 if row[1] is not None else np.nan
            ssd_val = row[2] * 25.0 if row[2] is not None else np.nan
            
            all_trials.append({
                'result': row[0], 
                'rt_ms': rt_val, 
                'ssd_ms': ssd_val, 
                'trial': t + 1, 
                'sim_id': sess_id
            })
    return pd.DataFrame(all_trials)


def run_multi_rdex(params, n_sessions=N_SESSIONS):
    model = RDEX(params=params)
    all_trials = []
    
    for sess_id in range(n_sessions):
        outcomes = model.simu_task(verbose=False)
        for t, row in enumerate(outcomes):
            all_trials.append({
                'result': row[0], 'rt_ms': row[1], 'ssd_ms': row[2], 
                'is_correct': row[3], 'trial': t + 1, 'sim_id': sess_id
            })
    return pd.DataFrame(all_trials)

# ==========================================
# 2. Plotting Utility
# ==========================================
def _clean_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(STYLE['axis_lw'])
    ax.spines['bottom'].set_linewidth(STYLE['axis_lw'])
    ax.tick_params(width=STYLE['axis_lw'], labelsize=STYLE['tick_fontsize'])

def get_outcome_stats(df, outcomes):
    if 'sim_id' not in df.columns or df['sim_id'].nunique() == 1:
        rates = df['result'].value_counts(normalize=True)
        return np.array([rates.get(o, 0) for o in outcomes]), np.zeros(len(outcomes))
    
    rates_per_sim = df.groupby('sim_id')['result'].value_counts(normalize=True).unstack(fill_value=0)
    rates_per_sim = rates_per_sim.reindex(columns=outcomes, fill_value=0)
    return rates_per_sim.mean().values, rates_per_sim.std().values

def plot_kde(ax, data, color, is_obs=False):
    if len(data) < 2: return
    kde = gaussian_kde(data)
    x_eval = np.linspace(0, 1000, 200)
    y_eval = kde(x_eval)
    
    if is_obs:
        ax.fill_between(x_eval, y_eval, color=color, alpha=0.3, edgecolor='none')
        ax.plot(x_eval, y_eval, color=color, lw=1.0, alpha=0.8)
    else:
        ax.fill_between(x_eval, y_eval, color=color, alpha=0.15, edgecolor='none')
        ax.plot(x_eval, y_eval, color=color, lw=1.5, alpha=0.9)

def plot_single_subject_comparison(subjects, raw_dir, pomdp_csv, rdex_csv, output_path="rdex_comp_reps.png"):
    fig, axs = plt.subplots(5, 3, figsize=(8.5, 9.5), constrained_layout=True)
    panel_labels = ['A', 'B', 'C', 'D', 'E']
    outcomes = ['GS', 'GE', 'GM', 'SS', 'SE']
    
    for col_idx, sid in enumerate(subjects):
        print(f"\nProcessing Subject: {sid}...")
        
        df_obs = extract_empirical_data(sid, raw_dir)
        p_pomdp = get_pomdp_params(pomdp_csv, sid)
        p_rdex = get_rdex_params(rdex_csv, sid)
        
        if df_obs is None or not p_pomdp or not p_rdex:
            print(f"  -> ERROR: Missing data or parameters for {sid}")
            continue
            
        df_pomdp = run_multi_pomdp(p_pomdp, N_SESSIONS)
        df_rdex = run_multi_rdex(p_rdex, N_SESSIONS)
        
        datasets = [
            ('Obs', df_obs, COLOR_OBS),
            ('POMDP', df_pomdp, COLOR_POMDP),
            ('RDEX', df_rdex, COLOR_RDEX)
        ]
        
        clean_sid_title = str(sid).replace('NDAR_', '')

        col_titles = [
            "Good Fit (95th Percentile)", 
            "Moderate Fit (50th Percentile)", 
            "Poor Fit (5th Percentile)"
        ]
        # -----------------------------------------------------------
        # Row 0 (A): Outcome Rates
        # -----------------------------------------------------------
        ax = axs[0, col_idx]
        x = np.arange(len(outcomes))
        bar_width = 0.25
        
        for i, (name, df, color) in enumerate(datasets):
            mean_rates, std_rates = get_outcome_stats(df, outcomes)
            offset = (i - 1) * bar_width
            
            ax.bar(x + offset, mean_rates, bar_width, label=name, color=color, alpha=0.8)
            if std_rates.sum() > 0:
                ax.errorbar(x + offset, mean_rates, yerr=std_rates, fmt='none', 
                            ecolor='black', elinewidth=0.8, capsize=2, alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(outcomes, fontsize=STYLE['tick_fontsize'])
        ax.set_ylim(0, 1.05)
        # ax.set_title(clean_sid_title, fontsize=STYLE['axes.titlesize'], pad=6)
        ax.set_title(col_titles[i], fontsize=STYLE['axes.titlesize'], pad=6)
        
        
        if col_idx == 0: 
            ax.set_ylabel("Proportion", fontsize=STYLE['axes.labelsize'])
            ax.text(-0.35, 1.05, panel_labels[0], transform=ax.transAxes, 
                    fontsize=STYLE['panel_label_size'], fontweight='bold', va='top', ha='right')
        _clean_spines(ax)

        # -----------------------------------------------------------
        # Row 1 & 2 (B & C): RT Distributions
        # -----------------------------------------------------------
        bins_ms = np.linspace(0, 1000, 31)
        
        for row_offset, res_type in enumerate(['GS', 'SE']):
            row_idx = 1 + row_offset
            ax = axs[row_idx, col_idx]
            
            for name, df, color in datasets:
                raw_data = df[df['result'] == res_type]['rt_ms'].values
                data = raw_data[np.isfinite(raw_data)]
                if len(data) < 2: continue
                
                if name == 'Obs':
                    # 观察数据使用 Histogram，显示最原始的分布
                    ax.hist(data, bins=bins_ms, density=True, 
                            color=color, alpha=0.35, edgecolor='none', zorder=1)
                else:
                    # 模型模拟数据使用 KDE，显示平滑的预测分布
                    plot_kde(ax, data, color, is_obs=False)
            
            ax.set_xlim(0, 1000)
            ax.set_xlabel("Time (ms)", fontsize=STYLE['axes.labelsize'])
            ax.set_ylim(bottom=0)
            
            if col_idx == 0:
                y_label = "Go Success RT" if res_type == 'GS' else "Stop Error RT"
                ax.set_ylabel(f"Density\n({y_label})", fontsize=STYLE['axes.labelsize'])
                ax.text(-0.35, 1.05, panel_labels[row_idx], transform=ax.transAxes, 
                        fontsize=STYLE['panel_label_size'], fontweight='bold', va='top', ha='right')
            _clean_spines(ax)

        # -----------------------------------------------------------
        # Row 3 (D): SSD Trajectory
        # -----------------------------------------------------------
        ax = axs[3, col_idx]
        for name, df, color in datasets:
            if name == 'Obs':
                df_st = df.copy()
                df_st['ssd_filled'] = df_st['ssd_ms'].ffill()
                ax.scatter(df_st['trial'], df_st['ssd_filled'], color=color, s=4, alpha=0.4, zorder=1)
            else:
                for sim_id in range(N_TRACES_PLOT):
                    df_sub = df[df['sim_id'] == sim_id].copy()
                    if df_sub.empty: continue
                    df_sub['ssd_filled'] = df_sub['ssd_ms'].ffill()
                    alpha_val = 0.8 if sim_id == 0 else 0.3
                    lw_val = 1.2 if sim_id == 0 else 0.8
                    ax.step(df_sub['trial'], df_sub['ssd_filled'], color=color, 
                            alpha=alpha_val, lw=lw_val, where='post', zorder=2)

        ax.set_xlim(0, 360)
        ax.set_xticks([1, 120, 240, 360])
        ax.set_yticks([0, 400, 800]) 
        ax.set_ylim(-25, 850)
        ax.set_xlabel("Trial", fontsize=STYLE['axes.labelsize'])
        
        if col_idx == 0: 
            ax.set_ylabel("SSD (ms)", fontsize=STYLE['axes.labelsize'])
            ax.text(-0.35, 1.05, panel_labels[3], transform=ax.transAxes, 
                    fontsize=STYLE['panel_label_size'], fontweight='bold', va='top', ha='right')
        _clean_spines(ax)

        # -----------------------------------------------------------
        # Row 4 (E): Inhibition Function P(SS)
        # -----------------------------------------------------------
        ax = axs[4, col_idx]
        
        for name, df, color in datasets:
            df_stop = df[df['result'].isin(['SS', 'SE'])].copy()
            if df_stop.empty: continue
            
            df_stop['is_ss'] = (df_stop['result'] == 'SS').astype(float)
            df_stop['ssd_bin'] = np.round(df_stop['ssd_ms'] / 50.0) * 50.0
            
            if name == 'Obs' or df['sim_id'].nunique() == 1:
                prob_ss = df_stop.groupby('ssd_bin')['is_ss'].mean()
                ax.plot(prob_ss.index, prob_ss.values, 'o--', color=color, 
                        linewidth=1.0, markersize=4, alpha=0.6)
            else:
                prob_per_sim = df_stop.pivot_table(index='sim_id', columns='ssd_bin', values='is_ss', aggfunc='mean')
                mean_p = prob_per_sim.mean(axis=0)
                std_p = prob_per_sim.std(axis=0).fillna(0)
                
                ax.fill_between(mean_p.index, 
                                np.clip(mean_p - std_p, 0, 1), 
                                np.clip(mean_p + std_p, 0, 1),
                                color=color, alpha=0.15, edgecolor='none')
                ax.plot(mean_p.index, mean_p.values, 'o-', color=color, 
                        linewidth=1.2, markersize=4, alpha=0.8)
                
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-25, 850)
        ax.set_xticks([0, 200, 400, 600, 800])
        ax.set_xlabel("SSD (ms)", fontsize=STYLE['axes.labelsize'])
        
        if col_idx == 0: 
            ax.set_ylabel("P(Stop Success)", fontsize=STYLE['axes.labelsize'])
            ax.text(-0.35, 1.05, panel_labels[4], transform=ax.transAxes, 
                    fontsize=STYLE['panel_label_size'], fontweight='bold', va='top', ha='right')
        _clean_spines(ax)

    # -----------------------------------------------------------
    # Global Legend
    # -----------------------------------------------------------
    legend_elements = [
        Patch(facecolor=COLOR_OBS, alpha=0.8, label='Observation'),
        Patch(facecolor=COLOR_POMDP, alpha=0.8, label='POMDP'),
        Patch(facecolor=COLOR_RDEX, alpha=0.8, label='RDEX')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 1.03), ncol=3, 
               frameon=False, fontsize=STYLE['legend_fontsize'], 
               columnspacing=3.0, handlelength=1.5)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSuccess: Grid plot saved to {output_path}")

# ==========================================
# Main Execution
# ==========================================
if __name__ == '__main__':
    plot_single_subject_comparison(
        subjects=SUBJECTS, 
        raw_dir=RAW_DATA_DIR, 
        pomdp_csv=POMDP_PARAMS_CSV, 
        rdex_csv=RDEX_PARAMS_CSV
    )