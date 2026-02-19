
from matplotlib import pyplot as plt, style
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from core import simulation

def plot_simu_trial_batch_belief_5conditions(out_go, out_stop, show_cnt=5, saveto=None):
    """
    PLOS Comp Biol compliant version (V2).
    Changes:
    - Adds Y-axis labels ("Belief State") to ALL subplots (A, B, C, D, E).
    - Slightly increases wspace to accommodate the new labels.
    """
    # ==========================================
    # PLOS STYLING & PALETTE
    # ==========================================
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    style = {
        "label_fontsize": 9,
        "title_fontsize": 10,
        "tick_fontsize": 8,
        "legend_fontsize": 9,
        "panel_label_size": 12,

        # Line Styles
        "mean_lw": 1.5,
        "mean_alpha": 1.0,
        "sem_alpha": 0.25,

        "sample_active_lw": 0.6,
        "sample_active_alpha": 0.4,
        
        "marker_size": 15, 
        "marker_edge_width": 0.5,

        "ssd_lw": 1.0,
        "ssd_alpha": 0.6,
    }

    colors = {
        "beta": "#4E7C59",    # Sage/Forest Green
        "sigma": "#800020",   # Burgundy
        "survival": "gray"
    }

    # ==========================================
    # DATA PREP
    # ==========================================
    df_go = pd.DataFrame(out_go)
    df_stop = pd.DataFrame(out_stop)
    ssd = df_stop.iloc[0]['ssd'] if len(df_stop) > 0 else None

    df_gs = df_go[df_go['result'] == 'GS']
    df_ge = df_go[df_go['result'] == 'GE']
    df_gm = df_go[df_go['result'] == 'GM']
    df_ss = df_stop[df_stop['result'] == 'SS']
    df_se = df_stop[df_stop['result'] == 'SE']

    df_list = [df_gs, df_ge, df_gm, df_ss, df_se]
    condition_names = ['Go Success', 'Go Error', 'Go Missing', 'Stop Success', 'Stop Error']

    # Keep figsize consistent
    fig, ax = plt.subplots(2, 3, figsize=(7.5, 4.8))
    
    MAX_TIME_STEPS = 40
    xticks = [1, 10, 20, 30, 40]
    panel_labels = ['A', 'B', 'C', 'D', 'E']

    for i, df in enumerate(df_list):
        row, col = divmod(i, 3)
        axis = ax[row, col]

        # Add Panel Label
        axis.text(-0.25, 1.1, panel_labels[i], transform=axis.transAxes,
                  fontsize=style['panel_label_size'], fontweight='bold', va='top', ha='right')

        if len(df) == 0:
            axis.axis('off')
            continue

        # -----------------------------------------------------------
        # 1. PRE-PROCESS
        # -----------------------------------------------------------
        beta_trunc_list = []
        sigma_trunc_list = []
        rt_indices = []

        for _, row_data in df.iterrows():
            beta_seq = row_data['beta_seq']
            sigma_seq = row_data['sigma_seq']
            policy_seq = row_data['policy_seq']

            decision_idx = len(policy_seq) - 1 
            for t_idx, act in enumerate(policy_seq):
                if act != 2: 
                    decision_idx = t_idx
                    break
            
            rt_indices.append(decision_idx)
            limit = decision_idx + 1
            beta_trunc_list.append(beta_seq[:limit])
            sigma_trunc_list.append(sigma_seq[:limit])

        # -----------------------------------------------------------
        # 2. SETUP DUAL AXIS (Shadow Style Survival)
        # -----------------------------------------------------------
        ax2 = axis.twinx()
        valid_rts = np.sort([r + 1 for r in rt_indices])
        
        if len(valid_rts) > 0:
            n_samples = len(valid_rts)
            y_survival = 1.0 - np.arange(1, n_samples + 1) / n_samples
            x_surv_plot = np.concatenate(([1], valid_rts))
            y_surv_plot = np.concatenate(([1.0], y_survival))

            ax2.fill_between(x_surv_plot, y_surv_plot, step='post', 
                             color=colors['survival'], alpha=0.15,
                             linewidth=0) 

        # Style Right Axis
        # Note: We keep the label here for all plots to balance the new left labels
        ax2.set_ylabel('Survival Prob.', color='gray', fontsize=style['label_fontsize'])
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=style['tick_fontsize'])
        ax2.set_ylim(0, 1.05) 
        ax2.set_yticks([0, 1.0])
        ax2.set_yticklabels(["0", "1"])
        
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_color('gray')
        ax2.spines['right'].set_linewidth(0.5)

        # -----------------------------------------------------------
        # 3. PLOT MEANS 
        # -----------------------------------------------------------
        df_beta_trunc = pd.DataFrame(beta_trunc_list)
        df_sigma_trunc = pd.DataFrame(sigma_trunc_list)

        beta_mean = df_beta_trunc.mean(axis=0)
        beta_sem = df_beta_trunc.sem(axis=0)
        sigma_mean = df_sigma_trunc.mean(axis=0)
        sigma_sem = df_sigma_trunc.sem(axis=0)

        x_vals_mean = np.arange(1, len(beta_mean) + 1)

        axis.plot(x_vals_mean, beta_mean, 
                  color=colors["beta"], linewidth=style['mean_lw'], alpha=style['mean_alpha'])
        axis.fill_between(x_vals_mean, beta_mean - beta_sem, beta_mean + beta_sem,
                          color=colors["beta"], alpha=style['sem_alpha'], linewidth=0)

        axis.plot(x_vals_mean, sigma_mean, 
                  color=colors["sigma"], linewidth=style['mean_lw'], alpha=style['mean_alpha'])
        axis.fill_between(x_vals_mean, sigma_mean - sigma_sem, sigma_mean + sigma_sem,
                          color=colors["sigma"], alpha=style['sem_alpha'], linewidth=0)

        # -----------------------------------------------------------
        # 4. PLOT SAMPLES
        # -----------------------------------------------------------
        subset_indices = list(range(min(show_cnt, len(df))))
        for j in subset_indices:
            d_idx = rt_indices[j]
            limit = d_idx + 1
            b_seq = df.iloc[j]['beta_seq']
            s_seq = df.iloc[j]['sigma_seq']
            x_sample = np.arange(1, limit + 1)
            is_last_step = (d_idx >= MAX_TIME_STEPS - 1)

            axis.plot(x_sample, b_seq[:limit],
                      color=colors["beta"], linewidth=style['sample_active_lw'], 
                      alpha=style['sample_active_alpha'])
            if not is_last_step:
                axis.scatter(x_sample[-1], b_seq[d_idx], 
                             s=style['marker_size'], color=colors["beta"], 
                             alpha=0.9, zorder=3, edgecolors='white', linewidth=style['marker_edge_width'])

            axis.plot(x_sample, s_seq[:limit],
                      color=colors["sigma"], linewidth=style['sample_active_lw'], 
                      alpha=style['sample_active_alpha'])
            if not is_last_step:
                axis.scatter(x_sample[-1], s_seq[d_idx], 
                             s=style['marker_size'], color=colors["sigma"], 
                             alpha=0.9, zorder=3, edgecolors='white', linewidth=style['marker_edge_width'])

        # -----------------------------------------------------------
        # 5. FINAL FORMATTING
        # -----------------------------------------------------------
        if condition_names[i].startswith('Stop') and ssd is not None:
            axis.axvline(x=ssd + 1, color='k', linestyle='--', linewidth=style['ssd_lw'], alpha=style['ssd_alpha'])
            axis.axvline(x=ssd + 12 + 1, color='k', linestyle='--', linewidth=style['ssd_lw'], alpha=style['ssd_alpha'])

        axis.set_xlim(1, MAX_TIME_STEPS)
        axis.set_ylim(0, 1.05)
        axis.set_xticks(xticks)
        axis.set_yticks([0, 0.5, 1.0])
        
        axis.set_title(f"{condition_names[i]}", pad=6, fontsize=style['title_fontsize'], fontweight='regular')
        axis.tick_params(axis='both', labelsize=style['tick_fontsize'])

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False) 
        
        axis.set_zorder(ax2.get_zorder() + 1) 
        axis.patch.set_visible(False)
        
        # [MODIFIED] Always set Y-label for every plot (B, C, E included)
        axis.set_ylabel(r"Belief State", fontsize=style['label_fontsize'])
        axis.set_xlabel("Time ", fontsize=style['label_fontsize'])

    ax[1, 2].axis("off")
    
    # [MODIFIED] Increased wspace 0.4 -> 0.55 to allow space for internal Y-labels
    plt.subplots_adjust(hspace=0.5, wspace=0.55, top=0.88, bottom=0.12, left=0.08, right=0.92)

    # Legend
    legend_handles = [
        Line2D([0], [0], color=colors["beta"], lw=style['mean_lw'], label=r'Go Belief $\beta^\nu_t$'),
        Line2D([0], [0], color=colors["sigma"], lw=style['mean_lw'], label=r'Stop Trial Belief $\sigma^\nu_t$'),
        Patch(facecolor=colors['survival'], alpha=0.15, label='Survival Probability'),
        Line2D([0], [0], color='k', linestyle='--', lw=style['ssd_lw'], alpha=style['ssd_alpha'], label='Stop Signal Onset / Offset')
    ]

    fig.legend(handles=legend_handles, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 1.02), 
               ncol=4, 
               frameon=False, 
               fontsize=style['legend_fontsize'],
               handlelength=1.5,
               columnspacing=1.5)

    if saveto:
        fig.savefig(saveto, dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
    else:
        plt.show()


def plot_simu_trial_batch_action_value_5conditions(out_go, out_stop, show_cnt=5, saveto=None):
    """
    PLOS Comp Biol compliant version (V2) - Action Values.
    Matches style of plot_simu_trial_batch_belief_5conditions.
    """
    # ==========================================
    # PLOS STYLING & PALETTE
    # ==========================================
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    style = {
        "label_fontsize": 9,
        "title_fontsize": 10,
        "tick_fontsize": 8,
        "legend_fontsize": 9,
        "panel_label_size": 12,

        # Line Styles
        "mean_lw": 1.5,
        "mean_alpha": 1.0,
        "sem_alpha": 0.25,

        "sample_active_lw": 0.6,
        "sample_active_alpha": 0.4,
        
        "marker_size": 15, 
        "marker_edge_width": 0.5,

        "ssd_lw": 1.0,
        "ssd_alpha": 0.6,
    }

    colors = {
        "Q_L": "#E3B23C", # Gold
        "Q_R": "#3A6EA5", # Blue
        "Q_W": "#7A7A7A", # Dark Gray for Wait Action
        "survival": "gray"
    }

    # ==========================================
    # DATA PREP
    # ==========================================
    df_go = pd.DataFrame(out_go)
    df_stop = pd.DataFrame(out_stop)
    ssd = df_stop.iloc[0]['ssd'] if len(df_stop) > 0 else None

    # Filter by result type
    df_gs = df_go[df_go['result'] == 'GS']
    df_ge = df_go[df_go['result'] == 'GE']
    df_gm = df_go[df_go['result'] == 'GM']
    df_ss = df_stop[df_stop['result'] == 'SS']
    df_se = df_stop[df_stop['result'] == 'SE']

    df_list = [df_gs, df_ge, df_gm, df_ss, df_se]
    condition_names = ['Go Success', 'Go Error', 'Go Missing', 'Stop Success', 'Stop Error']

    # Standard Journal Size
    fig, ax = plt.subplots(2, 3, figsize=(7.5, 4.8))
    
    MAX_TIME_STEPS = 40
    xticks = [1, 10, 20, 30, 40]
    panel_labels = ['A', 'B', 'C', 'D', 'E']

    for i, df in enumerate(df_list):
        row, col = divmod(i, 3)
        axis = ax[row, col]

        # Add Panel Label
        axis.text(-0.25, 1.1, panel_labels[i], transform=axis.transAxes,
                  fontsize=style['panel_label_size'], fontweight='bold', va='top', ha='right')

        if len(df) == 0:
            axis.axis('off')
            continue

        # -----------------------------------------------------------
        # 1. PRE-PROCESS: Extract Truncated Sequences & RTs
        # -----------------------------------------------------------
        q_l_trunc_list = []
        q_r_trunc_list = []
        q_w_trunc_list = []
        rt_indices = []

        for _, row_data in df.iterrows():
            ql_seq = row_data['value_left_seq']
            qr_seq = row_data['value_right_seq']
            qw_seq = row_data['value_wait_seq']
            policy_seq = row_data['policy_seq']

            decision_idx = len(policy_seq) - 1 
            for t_idx, act in enumerate(policy_seq):
                if act != 2: 
                    decision_idx = t_idx
                    break
            
            rt_indices.append(decision_idx)
            limit = decision_idx + 1
            
            q_l_trunc_list.append(ql_seq[:limit])
            q_r_trunc_list.append(qr_seq[:limit])
            q_w_trunc_list.append(qw_seq[:limit])

        # -----------------------------------------------------------
        # 2. SETUP DUAL AXIS (Shadow Style Survival)
        # -----------------------------------------------------------
        ax2 = axis.twinx()
        valid_rts = np.sort([r + 1 for r in rt_indices])
        
        if len(valid_rts) > 0:
            n_samples = len(valid_rts)
            y_survival = 1.0 - np.arange(1, n_samples + 1) / n_samples
            x_surv_plot = np.concatenate(([1], valid_rts))
            y_surv_plot = np.concatenate(([1.0], y_survival))
            
            # Shadow Area
            ax2.fill_between(x_surv_plot, y_surv_plot, step='post', 
                             color=colors['survival'], alpha=0.15,
                             linewidth=0) 

        # Style Right Axis
        ax2.set_ylabel('Survival Prob.', color='gray', fontsize=style['label_fontsize'])
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=style['tick_fontsize'])
        ax2.set_ylim(0, 1.05) 
        ax2.set_yticks([0, 1.0])
        ax2.set_yticklabels(["0", "1"])
        
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_color('gray')
        ax2.spines['right'].set_linewidth(0.5)

        # -----------------------------------------------------------
        # 3. PLOT MEANS 
        # -----------------------------------------------------------
        df_ql = pd.DataFrame(q_l_trunc_list)
        df_qr = pd.DataFrame(q_r_trunc_list)
        df_qw = pd.DataFrame(q_w_trunc_list)

        def plot_mean_with_sem(data_df, color):
            mean = data_df.mean(axis=0)
            sem = data_df.sem(axis=0)
            x_vals = np.arange(1, len(mean) + 1)
            axis.plot(x_vals, mean, 
                      color=color, linewidth=style['mean_lw'], alpha=style['mean_alpha'])
            axis.fill_between(x_vals, mean - sem, mean + sem,
                              color=color, alpha=style['sem_alpha'], linewidth=0)

        # Plot Order: Wait first (background), then Actions
        plot_mean_with_sem(df_qw, colors["Q_W"])
        plot_mean_with_sem(df_ql, colors["Q_L"])
        plot_mean_with_sem(df_qr, colors["Q_R"])

        # -----------------------------------------------------------
        # 4. PLOT SAMPLES
        # -----------------------------------------------------------
        subset_indices = list(range(min(show_cnt, len(df))))
        for j in subset_indices:
            d_idx = rt_indices[j]
            limit = d_idx + 1
            
            ql_seq = df.iloc[j]['value_left_seq']
            qr_seq = df.iloc[j]['value_right_seq']
            qw_seq = df.iloc[j]['value_wait_seq']
            
            x_sample = np.arange(1, limit + 1)
            is_last_step = (d_idx >= MAX_TIME_STEPS - 1)

            def plot_sample(seq, color):
                axis.plot(x_sample, seq[:limit],
                          color=color, linewidth=style['sample_active_lw'], 
                          alpha=style['sample_active_alpha'])
                if not is_last_step:
                    axis.scatter(x_sample[-1], seq[d_idx], 
                                 s=style['marker_size'], color=color, 
                                 alpha=0.9, zorder=3, edgecolors='white', linewidth=style['marker_edge_width'])

            plot_sample(qw_seq, colors["Q_W"])
            plot_sample(ql_seq, colors["Q_L"])
            plot_sample(qr_seq, colors["Q_R"])

        # -----------------------------------------------------------
        # 5. FINAL FORMATTING
        # -----------------------------------------------------------
        if condition_names[i].startswith('Stop') and ssd is not None:
            axis.axvline(x=ssd + 1, color='k', linestyle='--', linewidth=style['ssd_lw'], alpha=style['ssd_alpha'])
            axis.axvline(x=ssd + 12 + 1, color='k', linestyle='--', linewidth=style['ssd_lw'], alpha=style['ssd_alpha'])

        axis.set_xlim(1, MAX_TIME_STEPS)
        # axis.set_ylim(bottom=0) # Optional: Un-comment if strictly positive
        axis.set_xticks(xticks)
        
        axis.set_title(f"{condition_names[i]}", pad=6, fontsize=style['title_fontsize'], fontweight='regular')
        axis.tick_params(axis='both', labelsize=style['tick_fontsize'])

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False) 
        
        axis.set_zorder(ax2.get_zorder() + 1) 
        axis.patch.set_visible(False)

        axis.set_ylabel("Action Value", fontsize=style['label_fontsize'])
        axis.set_xlabel("Time Step", fontsize=style['label_fontsize'])

    ax[1, 2].axis("off")
    
    # [KEY CHANGE] wspace increased for labels
    plt.subplots_adjust(hspace=0.5, wspace=0.55, top=0.88, bottom=0.12, left=0.08, right=0.92)

    # ==========================================
    # GLOBAL LEGEND
    # ==========================================
    legend_handles = [
        Line2D([0], [0], color=colors["Q_W"], lw=style['mean_lw'], label=r'$Q_t(\mathbf{b}_t,W)$'),
        Line2D([0], [0], color=colors["Q_L"], lw=style['mean_lw'], label=r'$Q_t(\mathbf{b}_t,L)$'),
        Line2D([0], [0], color=colors["Q_R"], lw=style['mean_lw'], label=r'$Q_t(\mathbf{b}_t,R)$'),
        Patch(facecolor=colors['survival'], alpha=0.15, label='Survival Probability'),
        Line2D([0], [0], color='k', linestyle='--', lw=style['ssd_lw'], alpha=style['ssd_alpha'], label='Stop Signal Onset / Offset')
    ]

    fig.legend(handles=legend_handles, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 1.02), 
               ncol=5, 
               frameon=False, 
               fontsize=style['legend_fontsize'],
               handlelength=1.5,
               columnspacing=1.5)
    
    if saveto:
        fig.savefig(saveto, dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
    else:
        plt.show()


def plot_composite_2x3(
    model,
    params: dict,
    rates=(0.05, 1/6, 0.35, 0.55, 0.80),
    ssd=10,
    n_batch=500,     
    n_session=1000,  
    saveto=None,
    t_max=40
):
    """
    PLOS Comp Biol Final V6 "Aligned Legends" Version.
    - Colors: Forest/Sage (Beh), Burgundy (Belief), Coffee (Gap).
    - Internal Legend: Snapped to bottom-right corner (1.0, 0.02).
    - Global Legend: Horizontal layout text + handles, left-aligned rows.
    """
    
    # ==========================================
    # 1. EARTH TONE PALETTE
    # ==========================================
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    style = {
        "label_fontsize": 9,     
        "tick_fontsize": 8,      
        "legend_fontsize": 8,    
        "panel_label_size": 12,  
        
        "line_width": 1.0,      
        "marker_size": 3.5,        
        "cap_size": 1.5,
        "shadow_color": "gray"
    }

    # Behavior: Forest & Sage
    color_acc = '#363636' 
    color_rt  = "#7f7f7f" 
    
    # Belief: Burgundy
    cmap_bel = LinearSegmentedColormap.from_list("earth_red", ["#E6B8B8", "#800020"]) 
    rate_colors_bel = [cmap_bel(i) for i in np.linspace(0.1, 1.0, len(rates))]

    # Gap: Coffee/Bronze
    cmap_gap = LinearSegmentedColormap.from_list("earth_brown", ["#D7CCC8", "#5D4037"])
    rate_colors_gap = [cmap_gap(i) for i in np.linspace(0.1, 1.0, len(rates))]

    # ==========================================
    # 2. SIMULATION (Standard)
    # ==========================================
    def _truncate_and_collect(seq_list):
        if not seq_list: 
            return np.full(t_max, np.nan), np.full(t_max, np.nan)
        df = pd.DataFrame(seq_list).reindex(columns=range(t_max))
        with np.errstate(invalid='ignore'):
            mu = df.mean(axis=0).to_numpy()
            sd = df.std(axis=0, ddof=1).to_numpy()
            count = df.count(axis=0).to_numpy()
            se = np.where(count > 1, sd / np.sqrt(count), np.nan)
        return mu, se

    beh_rows = []
    timecourse_go = []
    timecourse_stop = []

    print("Simulating (V6)...")
    for r_idx, r in enumerate(rates):
        p = params.copy()
        p['rate_stop_trial'] = float(r)
        m = model.__class__(**p)
        m.value_iteration_tensor()

        # Behavior
        out_sess = []
        for _ in range(n_session):
            is_stop = np.random.rand() < float(r)
            gs = np.random.choice(["left", "right"])
            type_ = "stop" if is_stop else "nonstop"
            ssd_ = ssd if is_stop else None
            res_tuple = simulation.simu_trial(m, gs, type_, ssd_, verbose=False) 
            out_sess.append(res_tuple)
        
        df_beh = pd.DataFrame(out_sess, columns=["result", "rt", "ssd"])
        
        # Stats
        go_df = df_beh[df_beh["result"].isin(["GS", "GE", "GM"])]
        if len(go_df) > 0:
            go_acc = (go_df["result"]=="GS").mean()
            go_acc_sem = np.sqrt(go_acc*(1-go_acc)/len(go_df))
            gs_df = go_df[go_df["result"]=="GS"]
            go_rt = gs_df["rt"].mean() if len(gs_df)>0 else np.nan
            go_rt_sem = gs_df["rt"].std(ddof=1)/np.sqrt(len(gs_df)) if len(gs_df)>0 else np.nan
        else:
            go_acc, go_acc_sem, go_rt, go_rt_sem = [np.nan]*4

        stop_df = df_beh[df_beh["result"].isin(["SS", "SE"])]
        if len(stop_df) > 0:
            stop_acc = (stop_df["result"]=="SS").mean()
            stop_acc_sem = np.sqrt(stop_acc*(1-stop_acc)/len(stop_df))
            se_df = stop_df[stop_df["result"]=="SE"]
            stop_rt = se_df["rt"].mean() if len(se_df)>0 else np.nan 
            stop_rt_sem = se_df["rt"].std(ddof=1)/np.sqrt(len(se_df)) if len(se_df)>0 else np.nan
        else:
            stop_acc, stop_acc_sem, stop_rt, stop_rt_sem = [np.nan]*4

        beh_rows.append({
            "rate": r,
            "go_acc": go_acc, "go_acc_sem": go_acc_sem, 
            "go_rt": go_rt, "go_rt_sem": go_rt_sem,
            "stop_acc": stop_acc, "stop_acc_sem": stop_acc_sem, 
            "stop_rt": stop_rt, "stop_rt_sem": stop_rt_sem
        })

        # Timecourse
        g_data = {"GS": {"sig": [], "gap": [], "rt": []}, 
                  "GE": {"sig": [], "gap": [], "rt": []}, 
                  "GM": {"sig": [], "gap": [], "rt": []}}
        s_data = {"SS": {"sig": [], "gap": [], "rt": []}, 
                  "SE": {"sig": [], "gap": [], "rt": []}}

        def process_trial_result(res_dict, store_dict):
            out_type = res_dict["result"] 
            if out_type not in store_dict: return
            
            policy_seq = res_dict["policy_seq"]
            decision_idx = len(policy_seq) - 1
            for t_idx, act in enumerate(policy_seq):
                if act != 2: 
                    decision_idx = t_idx
                    break
            limit = decision_idx + 1
            sigma_seq = np.array(res_dict["sigma_seq"])[:limit]
            vw = np.array(res_dict["value_wait_seq"])[:limit]
            vl = np.array(res_dict["value_left_seq"])[:limit]
            vr = np.array(res_dict["value_right_seq"])[:limit]
            
            if len(vl) > 0:
                min_go = np.minimum(vl, vr)
                gap_seq = vw - min_go
            else:
                gap_seq = []

            store_dict[out_type]["sig"].append(sigma_seq)
            store_dict[out_type]["gap"].append(gap_seq)
            store_dict[out_type]["rt"].append(decision_idx)

        for _ in range(n_batch):
            true_go = np.random.choice(["left", "right"])
            res_g = simulation.simu_trial(m, true_go, "nonstop", None, verbose=True)
            process_trial_result(res_g, g_data)
            res_s = simulation.simu_trial(m, true_go, "stop", ssd, verbose=True)
            process_trial_result(res_s, s_data)

        # Aggregate
        pkt_g = {
            "rate": r, 
            "color_bel": rate_colors_bel[r_idx], 
            "color_gap": rate_colors_gap[r_idx],
            "rts": {k: v["rt"] for k,v in g_data.items()}
        }
        for cat in ["GS", "GE", "GM"]:
            pkt_g[f"sigma_{cat}_mu"], pkt_g[f"sigma_{cat}_sem"] = _truncate_and_collect(g_data[cat]["sig"])
            pkt_g[f"gap_{cat}_mu"],   pkt_g[f"gap_{cat}_sem"]   = _truncate_and_collect(g_data[cat]["gap"])
        timecourse_go.append(pkt_g)

        pkt_s = {
            "rate": r, 
            "color_bel": rate_colors_bel[r_idx], 
            "color_gap": rate_colors_gap[r_idx],
            "rts": {k: v["rt"] for k,v in s_data.items()}
        }
        for cat in ["SS", "SE"]:
            pkt_s[f"sigma_{cat}_mu"], pkt_s[f"sigma_{cat}_sem"] = _truncate_and_collect(s_data[cat]["sig"])
            pkt_s[f"gap_{cat}_mu"],   pkt_s[f"gap_{cat}_sem"]   = _truncate_and_collect(s_data[cat]["gap"])
        timecourse_stop.append(pkt_s)

    df_beh_summary = pd.DataFrame(beh_rows)

    # ==========================================
    # 3. PLOTTING
    # ==========================================    
    fig = plt.figure(figsize=(7.5, 4.8)) 
    gs = fig.add_gridspec(2, 3, wspace=0.60, hspace=0.35)

    ax_go_beh   = fig.add_subplot(gs[0, 0])
    ax_go_bel   = fig.add_subplot(gs[0, 1])
    ax_go_gap   = fig.add_subplot(gs[0, 2])
    ax_stop_beh = fig.add_subplot(gs[1, 0])
    ax_stop_bel = fig.add_subplot(gs[1, 1])
    ax_stop_gap = fig.add_subplot(gs[1, 2])

    axes_all = [ax_go_beh, ax_go_bel, ax_go_gap, ax_stop_beh, ax_stop_bel, ax_stop_gap]
    axes_time = [ax_go_bel, ax_go_gap, ax_stop_bel, ax_stop_gap]
    x_time = np.arange(1, t_max+1)

    # 3.1 Shadow Axes
    shadow_axes = {}
    for ax in axes_time:
        ax.set_zorder(1)
        ax.patch.set_visible(False)
        ax_shad = ax.twinx()
        ax_shad.set_ylim(0, 1.05)
        ax_shad.set_yticks([0, 1.0]) 
        ax_shad.set_yticklabels(["0", "1"], fontsize=style['tick_fontsize'], color='gray')
        ax_shad.set_ylabel('Survival Prob.', fontsize=style['label_fontsize'], color='gray')
        ax_shad.spines['top'].set_visible(False)
        ax_shad.spines['left'].set_visible(False)
        ax_shad.spines['right'].set_color('gray')
        ax_shad.spines['right'].set_linewidth(0.5)
        ax_shad.set_zorder(0)
        shadow_axes[ax] = ax_shad

    
    def draw_shadow(target_ax, rt_list, alpha):
        if not rt_list or len(rt_list) == 0: return
        try:
            valid_rts = np.sort([int(float(r)) + 1 for r in rt_list])
        except: return
        n = len(valid_rts)
        if n == 0: return
        y = 1.0 - np.arange(1, n + 1) / n
        x_plot = np.concatenate(([1], valid_rts))
        y_plot = np.concatenate(([1.0], y))
        
        target_ax.fill_between(x_plot, y_plot, step='post', 
                               color="gray", 
                               alpha=alpha, 
                               linewidth=0)

    mid_idx = len(rates) // 2
    rts_go = timecourse_go[mid_idx]["rts"]
    rts_stop = timecourse_stop[mid_idx]["rts"]

    # --- Draw Shadows: Go Context ---
    # GM (Miss): Very faint (background)
    # GS (Success): Medium faint
    # GE (Error): Darker (Highlight)
    draw_shadow(shadow_axes[ax_go_bel], rts_go.get("GM", []), alpha=0.05)
    draw_shadow(shadow_axes[ax_go_bel], rts_go.get("GS", []), alpha=0.10)
    draw_shadow(shadow_axes[ax_go_bel], rts_go.get("GE", []), alpha=0.15)
    
    draw_shadow(shadow_axes[ax_go_gap], rts_go.get("GM", []), alpha=0.05)
    draw_shadow(shadow_axes[ax_go_gap], rts_go.get("GS", []), alpha=0.10)
    draw_shadow(shadow_axes[ax_go_gap], rts_go.get("GE", []), alpha=0.15)
    # --- Draw Shadows: Stop Context ---
    # SS (Success): Very faint
    # SE (Error): Darker
    draw_shadow(shadow_axes[ax_stop_bel], rts_stop.get("SS", []), alpha=0.05)
    draw_shadow(shadow_axes[ax_stop_bel], rts_stop.get("SE", []), alpha=0.10)
    
    draw_shadow(shadow_axes[ax_stop_gap], rts_stop.get("SS", []), alpha=0.05)
    draw_shadow(shadow_axes[ax_stop_gap], rts_stop.get("SE", []), alpha=0.10)
    
    # 3.2 Time Series
    def plot_tc(ax, dataset, metric_prefix, outcome_keys, styles_map, color_key):
        for d in dataset:
            col = d[color_key]
            for out_key in outcome_keys:
                mu = d[f"{metric_prefix}_{out_key}_mu"]
                se = d[f"{metric_prefix}_{out_key}_sem"]
                ls = styles_map[out_key]["ls"]
                mask = ~np.isnan(mu)
                if np.any(mask):
                    ax.plot(x_time[mask], mu[mask], color=col, ls=ls, lw=style['line_width'])
                    ax.fill_between(x_time[mask], mu[mask]-se[mask], mu[mask]+se[mask], 
                                    color=col, alpha=0.15, lw=0)

    st_map = {
        "GS": {"ls": "-"}, "GE": {"ls": "--"}, "GM": {"ls": ":"},
        "SS": {"ls": "-"}, "SE": {"ls": "--"}
    }
    
    plot_tc(ax_go_bel, timecourse_go, "sigma", ["GS", "GE", "GM"], st_map, "color_bel")
    plot_tc(ax_go_gap, timecourse_go, "gap",   ["GS", "GE", "GM"], st_map, "color_gap")
    
    plot_tc(ax_stop_bel, timecourse_stop, "sigma", ["SS", "SE"], st_map, "color_bel")
    plot_tc(ax_stop_gap, timecourse_stop, "gap",   ["SS", "SE"], st_map, "color_gap")

    # 3.3 Behavior Panels (Bottom-Right Corner)
    def plot_behavior_panel(ax, df, acc_key, acc_sem_key, rt_key, rt_sem_key, context_label):
        # --- LEFT AXIS (Accuracy) ---
        # 1. Line (Neutral Gray)
        ax.plot(df["rate"], df[acc_key], color=color_acc, linestyle='-', lw=style['line_width'], zorder=1)
        # 2. Error Bars (Neutral Gray)
        ax.errorbar(df["rate"], df[acc_key], yerr=df[acc_sem_key],
                    fmt='none', ecolor=color_acc, capsize=style['cap_size'], zorder=1)
        # 3. Markers (COLORED by Stop Prior using 'rate_colors_bel')
        #    Using scatter to allow individual coloring
        ax.scatter(df["rate"], df[acc_key], s=20, 
                   c=rate_colors_bel, marker='o', 
                   edgecolors=color_acc, linewidth=0.5, zorder=2)
        
        ax.set_ylabel("Accuracy", color=color_acc, fontsize=style['label_fontsize'])
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(axis='y', colors=color_acc, labelsize=style['tick_fontsize'])
        ax.spines['left'].set_color(color_acc)
        ax.spines['top'].set_visible(False)

        # --- RIGHT AXIS (RT) ---
        axR = ax.twinx()
        # 1. Line (Neutral Gray)
        axR.plot(df["rate"], df[rt_key], color=color_rt, linestyle='-', lw=style['line_width'], zorder=1)
        # 2. Error Bars (Neutral Gray)
        axR.errorbar(df["rate"], df[rt_key], yerr=df[rt_sem_key],
                    fmt='none', ecolor=color_rt, capsize=style['cap_size'], zorder=1)
        # 3. Markers (COLORED by Stop Prior using 'rate_colors_gap')
        #    Using scatter to allow individual coloring
        axR.scatter(df["rate"], df[rt_key], s=20, 
                    c=rate_colors_bel, marker='s', 
                    edgecolors=color_rt, linewidth=0.5, zorder=2)
        
        axR.set_ylabel("Time Step", color=color_rt, fontsize=style['label_fontsize'])
        axR.set_ylim(1, 40)
        axR.set_yticks([1, 20, 40])
        axR.tick_params(axis='y', colors=color_rt, labelsize=style['tick_fontsize'])
        axR.spines['right'].set_color(color_rt)
        axR.spines['top'].set_visible(False)
        
        # --- LEGEND ---
        # Custom handles to show the shape and meaning, with neutral color
        h_acc = Line2D([0], [0], color=color_acc, marker='o', linestyle='-', 
                       lw=style['line_width'], markersize=style['marker_size'], label=f"{context_label} Acc.")
        h_rt  = Line2D([0], [0], color=color_rt, marker='s', linestyle='-', 
                       lw=style['line_width'], markersize=style['marker_size'], label=f"{context_label} RT")

        ax.legend(handles=[h_acc, h_rt], loc='lower right', bbox_to_anchor=(1.0, 0.02), 
                  frameon=False, fontsize=style['tick_fontsize'], handlelength=1.0)
        
        return axR

    plot_behavior_panel(ax_go_beh, df_beh_summary, "go_acc", "go_acc_sem", "go_rt", "go_rt_sem", "Go")
    plot_behavior_panel(ax_stop_beh, df_beh_summary, "stop_acc", "stop_acc_sem", "stop_rt", "stop_rt_sem", "Stop")

    # Formatting
    for ax in [ax_go_beh, ax_stop_beh]:
        ax.set_xlabel(r"Stop Prior $\tilde{r}^\nu$", fontsize=style['label_fontsize'])
        ax.set_xlim(0, 1)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='x', labelsize=style['tick_fontsize'])

    for ax in axes_time:
        ax.set_xlim(1, 40)
        ax.set_xticks([1, 20, 40])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=style['tick_fontsize'])
        ax.set_xlabel("Time Step", fontsize=style['label_fontsize'])

    ax_go_bel.set_ylabel(r"Belief $\sigma^\nu_t$", fontsize=style['label_fontsize'])
    ax_stop_bel.set_ylabel(r"Belief $\sigma^\nu_t$", fontsize=style['label_fontsize'])
    label_gap = r"$Q(W) - \min(Q(L), Q(R))$"
    ax_go_gap.set_ylabel(label_gap, fontsize=style['label_fontsize'])
    ax_stop_gap.set_ylabel(label_gap, fontsize=style['label_fontsize'])
    
    stop_signal_duration = 12 
    if ssd:
        for ax in [ax_stop_bel, ax_stop_gap]:
            ax.axvline(ssd+1, color='k', ls='--', alpha=0.5, linewidth=0.8)
            ax.axvline(ssd+1 + stop_signal_duration, color='k', ls='--', alpha=0.5, linewidth=0.8)

    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    for ax, label in zip(axes_all, labels):
        ax.text(-0.15, 1.15, label, transform=ax.transAxes,
                fontsize=style['panel_label_size'], fontweight='bold', va='top', ha='right')

    # ==========================================
    # 4. GLOBAL LEGEND (Precisely Aligned)
    # ==========================================
    # Y-coordinates for the two rows
    y_row1 = 0.12  # First row (Red Belief / Go Outcome)
    y_row2 = 0.08  # Second row (Brown Gap / Stop Outcome)
    
    # --- LEFT GROUP: PRIORS ---
    # Text label anchored at x=0.02
    fig.text(0.02, y_row1, r"Stop Prior $\tilde{r}^\nu$", ha='left', va='center', 
             fontsize=style['legend_fontsize']+1, fontweight='normal')
    
    # 1. Handles (Red) - Row 1
    # bbox_to_anchor x=0.14 puts the start of legend right after the text
    h_bel = [Line2D([0],[0], color=c, lw=1, label=f"{r:.2f}") for r,c in zip(rates, rate_colors_bel)]
    fig.legend(handles=h_bel, loc='center left', bbox_to_anchor=(0.12, y_row1),
               ncol=len(rates), frameon=False, fontsize=style['legend_fontsize'],
               columnspacing=0.8, handletextpad=0.4)
    
    # 2. Handles (Brown) - Row 2
    # Aligned exactly below Red handles (x=0.14)
    h_gap = [Line2D([0],[0], color=c, lw=1, label=f"{r:.2f}") for r,c in zip(rates, rate_colors_gap)]
    fig.legend(handles=h_gap, loc='center left', bbox_to_anchor=(0.12, y_row2),
               ncol=len(rates), frameon=False, fontsize=style['legend_fontsize'],
               columnspacing=0.8, handletextpad=0.4)


    # --- RIGHT GROUP: OUTCOMES ---
    # Text label anchored at x=0.55 (Middle-Right)
    fig.text(0.55, y_row1, "Outcome", ha='left', va='center', 
             fontsize=style['legend_fontsize']+1, fontweight='normal')

    # 3. Handles (Go) - Row 1
    # bbox_to_anchor x=0.64 puts it right after "Outcome" text
    h_go = [
        Line2D([0],[0], color='k', ls='-',  lw=1, label='Go Success'),
        Line2D([0],[0], color='k', ls='--', lw=1, label='Go Error'),
        Line2D([0],[0], color='k', ls=':',  lw=1, label='Go Missing'),
    ]
    fig.legend(handles=h_go, loc='center left', bbox_to_anchor=(0.64, y_row1),
               ncol=3, frameon=False, fontsize=style['legend_fontsize'],
               columnspacing=1.2, handletextpad=0.5)

    # 4. Handles (Stop) - Row 2
    # Aligned exactly below Go handles (x=0.64)
    h_stop = [
        Line2D([0],[0], color='k', ls='-',  lw=1, label='Stop Success'),
        Line2D([0],[0], color='k', ls='--', lw=1, label='Stop Error'),
    ]
    fig.legend(handles=h_stop, loc='center left', bbox_to_anchor=(0.64, y_row2),
               ncol=2, frameon=False, fontsize=style['legend_fontsize'],
               columnspacing=1.2, handletextpad=0.5)

    # Important: Adjust bottom margin to allow space for these legends
    plt.subplots_adjust(bottom=0.24, top=0.92, left=0.08, right=0.98)

    if saveto:
        fig.savefig(saveto, dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
    else:
        plt.show()


# def plot_policy_simple(model, saveto=None, dpi=300, selected_timesteps=None):
#     """
#     PLOS Comp Biol compliant version (V8) - Softmax / Probabilistic Policy.
    
#     Updates:
#     - Plots on Beta (Y) vs Zeta (X).
#     - X-axis label corrected to "Stop Signal Belief \zeta".
#     - Uses Softmax probabilities based on negative costs.
#     """
#     # ==========================================
#     # PLOS STYLING
#     # ==========================================
#     plt.rcParams['font.family'] = 'sans-serif'
#     plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

#     style = {
#         "label_fontsize": 10,
#         "title_fontsize": 11,
#         "tick_fontsize": 9,
#         "legend_fontsize": 9,
#     }

#     # Define Base Colors
#     colors = {
#         "Q_L": "#E3B23C",  # Gold (Left)
#         "Q_R": "#3A6EA5",  # Blue (Right)
#         "Q_W": "#F1EFEFF3",  # White (Wait) for better visibility
#     }
    
#     # Convert hex to RGB numpy arrays (0-1 range)
#     rgb_L = np.array(to_rgb(colors["Q_L"]))
#     rgb_R = np.array(to_rgb(colors["Q_R"]))
#     rgb_W = np.array(to_rgb(colors["Q_W"]))

#     # ==========================================
#     # DATA PREP: CALCULATE PROBABILITIES
#     # ==========================================
#     # 1. Retrieve Cost Values from separate arrays
#     # Arrays are shape [Horizon, Beta_Bins, Zeta_Bins]
#     # We exclude the last timestep (terminal) for policy plotting usually
#     limit = -1
#     V_L = model.value_left[:limit, :, :]
#     V_R = model.value_right[:limit, :, :]
#     V_W = model.value_wait[:limit, :, :]
    
#     # 2. Stack into a single tensor [T, H, W, 3]
#     # Order matches Action Space: 0=Left, 1=Right, 2=Wait
#     # Dimensions: [Time, Beta, Zeta, Action]
#     Costs = np.stack([V_L, V_R, V_W], axis=-1)
    
#     # 3. Compute Softmax Probabilities
#     # Formula: P(a) = exp(-Cost(a) * inv_temp) / Sum(...)
#     inv_temp = getattr(model, 'inv_temp', 1.0)
    
#     # Note the negative sign: We want High Prob for Low Cost
#     logits = -Costs * inv_temp
    
#     # Numerical stability: subtract max per state
#     max_logits = np.nanmax(logits, axis=-1, keepdims=True)
#     exp_logits = np.exp(logits - max_logits)
#     sum_exp = np.nansum(exp_logits, axis=-1, keepdims=True)
    
#     # Handle division by zero (if all are nan)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         probs = exp_logits / sum_exp
    
#     # Fill NaNs with 0 to avoid plotting errors
#     probs = np.nan_to_num(probs, nan=0.0)

#     # 4. Create RGB Image via Linear Combination
#     # Image = P_L * Col_L + P_R * Col_R + P_W * Col_W
#     # Broadcast colors: [1, 1, 1, 3]
#     img_rgb = (probs[..., 0:1] * rgb_L.reshape(1,1,1,3) + 
#                probs[..., 1:2] * rgb_R.reshape(1,1,1,3) + 
#                probs[..., 2:3] * rgb_W.reshape(1,1,1,3))
    
#     # Clip to ensure valid RGB range [0, 1] due to float precision
#     img_rgb = np.clip(img_rgb, 0.0, 1.0)

#     # ==========================================
#     # PLOTTING
#     # ==========================================
#     num_timesteps = Costs.shape[0]

#     if selected_timesteps is None:
#         selected_timesteps = [0, 10, 20, 30]
    
#     # Filter valid timesteps
#     selected_timesteps = [t for t in selected_timesteps if 0 <= t < num_timesteps]
#     n_plots = len(selected_timesteps)

#     # Figure Setup
#     fig, axes = plt.subplots(1, n_plots, figsize=(7.5, 2.6)) 
#     if n_plots == 1: axes = [axes]
    
#     for i, t in enumerate(selected_timesteps):
#         ax = axes[i]
        
#         # Extract the RGB image for this timestep
#         # shape is [Beta, Zeta, 3]
#         matrix_plot = img_rgb[t] 

#         # Plot RGB image directly
#         # extent=[0, 1, 0, 1] maps indices to unit square
#         # origin='lower' puts index (0,0) at bottom-left
#         # So Y-axis is Beta (dim 0), X-axis is Zeta (dim 1)
#         ax.imshow(matrix_plot, 
#                   extent=[0, 1, 0, 1], origin='lower', 
#                   interpolation='nearest', aspect='equal') 
        
#         ax.set_title(f"Time {t + 1}", fontsize=style['title_fontsize'], fontweight='regular', pad=8)
        
#         # [KEY CHANGE] Label explicitly as Zeta
#         ax.set_xlabel(r"Stop Signal Belief $\zeta^\nu_t$", fontsize=style['label_fontsize'])
        
#         if i == 0:
#             ax.set_ylabel(r"Go Belief $\beta^\nu_t$", fontsize=style['label_fontsize'])
#         else:
#             ax.tick_params(labelleft=False)

#         ax.set_xticks([0, 0.5, 1])
#         ax.set_yticks([0, 0.5, 1])
#         ax.tick_params(axis='both', which='major', labelsize=style['tick_fontsize'])
        
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)

#     # ==========================================
#     # LEGEND (Far Left)
#     # ==========================================
#     legend_elements = [
#         Patch(facecolor=colors['Q_L'], edgecolor=None, label='Left'),
#         Patch(facecolor=colors['Q_R'], edgecolor=None, label='Right'),
#         Patch(facecolor=colors['Q_W'], edgecolor=None, label='Wait'),
#     ]

#     fig.legend(handles=legend_elements, 
#                loc='lower left',           
#                bbox_to_anchor=(0.08, 0.88), 
#                ncol=3, 
#                frameon=False, 
#                fontsize=style['legend_fontsize'],
#                handlelength=2.5,
#                handletextpad=1.2, 
#                columnspacing=2.5)

#     # Adjust layout
#     plt.subplots_adjust(top=0.82, bottom=0.18, wspace=0.35, left=0.08, right=0.95)

#     if saveto:
#         fig.savefig(saveto, dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
#         plt.close(fig)
#     else:
#         plt.show()


#     """
#     Publication-quality FULL policy plot - Margin of Victory (RGB Gradient).
    
#     Simultaneously shows all three actions (Left, Right, Wait) and the gradient 
#     of decision confidence (Margin of Victory over the 2nd best action).
#     - Gold = Left preferred
#     - Blue = Right preferred
#     - Grey = Wait preferred
#     - White = Decision Boundary (Gap approaching 0)
#     """
#     from matplotlib.colors import to_rgb
    
#     # ==========================================
#     # PLOS STYLING
#     # ==========================================
#     plt.rcParams['font.family'] = 'sans-serif'
#     plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

#     style = {
#         "title_fontsize": 10,
#         "label_fontsize": 12, 
#         "tick_fontsize": 8,
#         "legend_fontsize": 10,
#     }

#     # Base Colors (using your palette)
#     color_L = np.array(to_rgb("#E3B23C"))  # Gold
#     color_R = np.array(to_rgb("#3A6EA5"))  # Blue
#     color_W = np.array(to_rgb("#7A7A7A"))  # Dark Grey (from action value plot)
#     color_White = np.array([1.0, 1.0, 1.0])

#     # ==========================================
#     # DATA PREP: CALCULATE MARGIN OF VICTORY
#     # ==========================================
#     limit = -1
#     V_L = model.value_left[:limit, :, :]
#     V_R = model.value_right[:limit, :, :]
#     V_W = model.value_wait[:limit, :, :]
    
#     # Stack costs: [Time, Beta, Zeta, Action(3)]
#     Costs = np.stack([V_L, V_R, V_W], axis=-1)
#     num_timesteps = Costs.shape[0]

#     # 1. Identify the winning action (lowest cost)
#     # 0 = Left, 1 = Right, 2 = Wait
#     Winners = np.argmin(Costs, axis=-1)

#     # 2. Calculate the Margin of Victory (Gap between 1st and 2nd best)
#     Costs_sorted = np.sort(Costs, axis=-1)
#     Best_Cost = Costs_sorted[..., 0]
#     Second_Best_Cost = Costs_sorted[..., 1]
    
#     Gap = Second_Best_Cost - Best_Cost  # Always positive

#     # 3. Normalize Gap to [0, 1] for Color Intensity
#     # Use 98th percentile to prevent deadline values from washing out the gradient
#     max_gap = np.nanpercentile(Gap, 98) 
#     Intensity = np.clip(Gap / max_gap, 0, 1)

#     # 4. Construct the RGB Image [Time, Beta, Zeta, RGB(3)]
#     img_rgb = np.ones((*Winners.shape, 3))

#     mask_L = (Winners == 0)
#     mask_R = (Winners == 1)
#     mask_W = (Winners == 2)

#     # Linear interpolation: Color = (1 - Intensity) * White + Intensity * Target_Color
#     for i in range(3): # For R, G, B channels
#         img_rgb[mask_L, i] = 1 - Intensity[mask_L] * (1 - color_L[i])
#         img_rgb[mask_R, i] = 1 - Intensity[mask_R] * (1 - color_R[i])
#         img_rgb[mask_W, i] = 1 - Intensity[mask_W] * (1 - color_W[i])

#     # ==========================================
#     # PLOTTING LAYOUT
#     # ==========================================
#     cols = 8
#     rows = (num_timesteps + cols - 1) // cols

#     fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.3), dpi=dpi)
#     axes = axes.flatten()

#     for i in range(len(axes)):
#         ax = axes[i]
        
#         if i < num_timesteps:
#             # Draw the RGB composite image
#             ax.imshow(img_rgb[i], 
#                       extent=[0, 1, 0, 1], origin='lower', 
#                       interpolation='nearest', aspect='equal')
            
#             # Titles & Formatting
#             ax.set_title(f"Time {i+1}", fontsize=style['tick_fontsize'], pad=4)
#             ax.set_xticks([0, 1])
#             ax.set_yticks([0, 1])
#             ax.tick_params(axis='both', which='major', labelsize=style['tick_fontsize'], length=2)
            
#             ax.spines["top"].set_visible(False)
#             ax.spines["right"].set_visible(False)

#             row_idx = i // cols
#             col_idx = i % cols
#             if col_idx != 0: ax.set_yticklabels([])
#             if row_idx != rows - 1: ax.set_xticklabels([])
#         else:
#             ax.axis("off")

#     # ==========================================
#     # GLOBAL LABELS & LEGEND
#     # ==========================================
#     fig.text(0.5, 0.08, r"Stop Signal Belief $\zeta^\nu_t$", ha='center', va='center', fontsize=style['label_fontsize'])
#     fig.text(0.08, 0.5, r"Go Belief $\beta^\nu_t$", ha='center', va='center', rotation='vertical', fontsize=style['label_fontsize'])

#     # Create Custom Legend
#     from matplotlib.patches import Patch
#     from matplotlib.lines import Line2D
    
#     legend_elements = [
#         Patch(facecolor="#E3B23C", edgecolor=None, label='Go Left Preferred'),
#         Patch(facecolor="#3A6EA5", edgecolor=None, label='Go Right Preferred'),
#         Patch(facecolor="#7A7A7A", edgecolor=None, label='Wait Preferred'),
#         # Add a mock "gradient" icon using a white-to-grey line to explain the fade
#         Line2D([0], [0], color='white', marker='s', markerfacecolor='white', markeredgecolor='black', markersize=8, label='Decision Boundary (White)')
#     ]

#     fig.legend(handles=legend_elements, 
#                loc='lower center',            
#                bbox_to_anchor=(0.5, 0.01), 
#                ncol=4, 
#                frameon=False, 
#                fontsize=style['legend_fontsize'],
#                handletextpad=0.5,
#                columnspacing=1.5)

#     plt.subplots_adjust(top=0.92, bottom=0.15, left=0.12, right=0.95, hspace=0.4, wspace=0.1)

#     if saveto:
#         fig.savefig(saveto, dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
#         plt.close(fig)
#     else:
#         plt.show()


# def plot_policy(model, saveto, dpi=300):
#     """
#     Publication-quality FULL policy plot (Probabilistic / Softmax Version).
    
#     Updates:
#     - Renders RGB probability gradients (Softmax) instead of discrete blocks.
#     - Uses Beta (Y) vs Zeta (X).
#     - Consistent styling with plot_policy_simple.
#     """
#     # ==========================================
#     # PLOS STYLING
#     # ==========================================
#     plt.rcParams['font.family'] = 'sans-serif'
#     plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

#     colors = {
#         "Q_L": "#E3B23C",  # Gold (Left)
#         "Q_R": "#3A6EA5",  # Blue (Right)
#         "Q_W": "#F1EFEFF3", # White/Very Light Gray (Wait)
#     }
    
#     # Convert hex to RGB
#     rgb_L = np.array(to_rgb(colors["Q_L"]))
#     rgb_R = np.array(to_rgb(colors["Q_R"]))
#     rgb_W = np.array(to_rgb(colors["Q_W"]))

#     style = {
#         "title_fontsize": 10,
#         "label_fontsize": 12, 
#         "tick_fontsize": 8,
#         "legend_fontsize": 10,
#     }

#     # ==========================================
#     # DATA PREP: CALCULATE PROBABILITIES
#     # ==========================================
#     # 1. Retrieve Costs (Time, Beta, Zeta)
#     # Exclude terminal step
#     limit = -1
#     V_L = model.value_left[:limit, :, :]
#     V_R = model.value_right[:limit, :, :]
#     V_W = model.value_wait[:limit, :, :]
    
#     # Stack: [Time, Beta, Zeta, Action=3]
#     Costs = np.stack([V_L, V_R, V_W], axis=-1)
#     num_timesteps = Costs.shape[0]

#     # 2. Compute Softmax
#     inv_temp = getattr(model, 'inv_temp', 1.0)
#     logits = -Costs * inv_temp
    
#     # Numerical stability
#     max_logits = np.nanmax(logits, axis=-1, keepdims=True)
#     exp_logits = np.exp(logits - max_logits)
#     sum_exp = np.nansum(exp_logits, axis=-1, keepdims=True)
    
#     with np.errstate(divide='ignore', invalid='ignore'):
#         probs = exp_logits / sum_exp
#     probs = np.nan_to_num(probs, nan=0.0)

#     # 3. Create RGB Image [Time, Beta, Zeta, 3]
#     # Mix colors based on probability
#     img_rgb_all = (probs[..., 0:1] * rgb_L.reshape(1,1,1,3) + 
#                    probs[..., 1:2] * rgb_R.reshape(1,1,1,3) + 
#                    probs[..., 2:3] * rgb_W.reshape(1,1,1,3))
    
#     img_rgb_all = np.clip(img_rgb_all, 0.0, 1.0)

#     # ==========================================
#     # PLOTTING LAYOUT
#     # ==========================================
#     # Grid dimensions (e.g., 8 cols)
#     cols = 8
#     rows = (num_timesteps + cols - 1) // cols

#     # Figsize approx 1.2 inch per subplot width
#     fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.3), dpi=dpi)
#     axes = axes.flatten()

#     for i in range(len(axes)):
#         ax = axes[i]
        
#         if i < num_timesteps:
#             # Get the RGB image for this timestep
#             matrix_plot = img_rgb_all[i]

#             # Draw RGB Image
#             # extent=[0,1,0,1] maps to Beta/Zeta 0->1
#             # origin='lower' puts (0,0) at bottom-left
#             ax.imshow(matrix_plot, 
#                       extent=[0, 1, 0, 1], origin='lower', 
#                       interpolation='nearest', aspect='equal')
            
#             # Titles
#             ax.set_title(f"Time {i+1}", fontsize=style['tick_fontsize'], pad=4)

#             # Ticks & Spines
#             ax.set_xticks([0, 1])
#             ax.set_yticks([0, 1])
#             ax.tick_params(axis='both', which='major', labelsize=style['tick_fontsize'], length=2)
            
#             ax.spines["top"].set_visible(False)
#             ax.spines["right"].set_visible(False)

#             # Smart Labeling (Only edges)
#             row_idx = i // cols
#             col_idx = i % cols

#             if col_idx != 0:
#                 ax.set_yticklabels([])
            
#             # Only show X-tick labels for the bottom-most existing plots
#             if row_idx != rows - 1:
#                 ax.set_xticklabels([])

#         else:
#             ax.axis("off")

#     # ==========================================
#     # GLOBAL LABELS & LEGEND
#     # ==========================================
    
#     # Global Axis Labels (Updated to Beta / Zeta)
#     fig.text(0.5, 0.02, r"Stop Signal Belief $\zeta^\nu_t$", ha='center', va='center', fontsize=style['label_fontsize'])
#     fig.text(0.08, 0.5, r"Go Belief $\beta^\nu_t$", ha='center', va='center', rotation='vertical', fontsize=style['label_fontsize'])

#     # Legend Elements
#     legend_elements = [
#         Patch(facecolor=colors['Q_L'], edgecolor=None, label='Go Left'),
#         Patch(facecolor=colors['Q_R'], edgecolor=None, label='Go Right'),
#         Patch(facecolor=colors['Q_W'], edgecolor=None, label='Wait'),
#     ]

#     # Legend Position: Far Left
#     fig.legend(handles=legend_elements, 
#                loc='lower left',            
#                bbox_to_anchor=(0.12, 0.90), 
#                ncol=4, 
#                frameon=False, 
#                fontsize=style['legend_fontsize'],
#                handlelength=1.5,
#                columnspacing=2.0)

#     # Adjust layout
#     plt.subplots_adjust(top=0.88, bottom=0.08, left=0.12, right=0.95, hspace=0.4, wspace=0.1)

#     if saveto:
#         fig.savefig(saveto, dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
#         plt.close(fig)
#     else:
#         plt.show()


def plot_policy_gap(model, saveto=None, dpi=300):
    """
    Publication-quality FULL policy plot - Sequential "Decision Preference".
    
    Plots the action preference: Q(Wait) - min(Q(Left), Q(Right)).
    - Deep Wait preference (Negative) -> Maps to White/Lightest color.
    - Strong Go preference (Positive) -> Maps to Strong/Dark color.
    """
    # ==========================================
    # PLOS STYLING
    # ==========================================
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    style = {
        "title_fontsize": 10,
        "label_fontsize": 12, 
        "tick_fontsize": 8,
        "legend_fontsize": 10,
    }
    
    # cmap_gap = LinearSegmentedColormap.from_list("earth_brown", ["#FFFFFF", "#3B1D14"])
    # cmap_gap = LinearSegmentedColormap.from_list("earth_brown", ["#FFFFFF", "#FF7300"])
    cmap_gap = 'Purples'


    # ==========================================
    # DATA PREP: CALCULATE "ACTION PREFERENCE"
    # ==========================================
    limit = -1
    V_L = model.value_left[:limit, :, :]
    V_R = model.value_right[:limit, :, :]
    V_W = model.value_wait[:limit, :, :]
    
    Preference = V_W - np.minimum(V_L, V_R)
    num_timesteps = Preference.shape[0]

    vmin = np.nanpercentile(Preference, 0)   
    vmax = np.nanpercentile(Preference, 100)  

    # ==========================================
    # PLOTTING LAYOUT
    # ==========================================
    cols = 8
    rows = (num_timesteps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.3), dpi=dpi)
    axes = axes.flatten()

    im = None

    for i in range(len(axes)):
        ax = axes[i]
        
        if i < num_timesteps:
            matrix_plot = Preference[i]

            im = ax.imshow(matrix_plot, 
                           extent=[0, 1, 0, 1], origin='lower', 
                           cmap=cmap_gap, vmin=vmin, vmax=vmax,
                           interpolation='nearest', aspect='equal')
            
            ax.set_title(f"Time {i+1}", fontsize=style['tick_fontsize'], pad=4)

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.tick_params(axis='both', which='major', labelsize=style['tick_fontsize'], length=2)
            
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            row_idx = i // cols
            col_idx = i % cols

            if col_idx != 0:
                ax.set_yticklabels([])
            
            if row_idx != rows - 1:
                ax.set_xticklabels([])

        else:
            ax.axis("off")

    # ==========================================
    # GLOBAL LABELS & VERTICAL COLORBAR (Right Side)
    # ==========================================
    fig.text(0.47, 0.03, r"Stop Signal Belief $\zeta^\nu_t$", ha='center', va='center', fontsize=style['label_fontsize'])
    fig.text(0.04, 0.49, r"Go Belief $\beta^\nu_t$", ha='center', va='center', rotation='vertical', fontsize=style['label_fontsize'])

    fig.text(0.08, 0.95, r"Action Preference  $Q(W) - \min(Q(L), Q(R))$",
             ha='left', va='center', fontsize=style['label_fontsize'])

    cbar_ax = fig.add_axes([0.88, 0.10, 0.015, 0.78]) 
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    
    cbar.set_ticks([vmin, 0, vmax])
    cbar.set_ticklabels(['Strong Wait\nPreference', 
                         'Decision\nBoundary\n(0)', 
                         'Strong Go\nPreference'])
    
    cbar.ax.tick_params(labelsize=style['tick_fontsize'], length=0, pad=5)
    
    tick_labels = cbar.ax.get_yticklabels()
    tick_labels[0].set_verticalalignment('bottom')
    tick_labels[2].set_verticalalignment('top')

    plt.subplots_adjust(top=0.88, bottom=0.10, left=0.08, right=0.86, hspace=0.45, wspace=0.1)

    if saveto:
        fig.savefig(saveto, dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
    else:
        plt.show()

def plot_policy_simple_combined(model, saveto=None, dpi=300, selected_timesteps=None):
    """
    Publication-quality COMBINED simple policy plot (2 rows x 4 cols).
    Right-side annotation panel layout (Pixel-perfect alignment, matched heights).
    """
    # ==========================================
    # PLOS STYLING
    # ==========================================
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    style = {
        "title_fontsize": 11,
        "label_fontsize": 11, 
        "tick_fontsize": 9,
        "legend_fontsize": 10,
    }

    # Colors
    colors = {
        # "Q_L": "#E3B23C",  # Gold (Left)
        "Q_L": "#E3B23C",  # Orange (Left) - Adjusted for better contrast
        
        "Q_R": "#3A6EA5",  # Blue (Right)
        "Q_W": "#E2E2E2"   # Visible Light Gray (Wait)
    }
    rgb_L = np.array(to_rgb(colors["Q_L"]))
    rgb_R = np.array(to_rgb(colors["Q_R"]))
    rgb_W = np.array(to_rgb(colors["Q_W"]))
    
    cmap_gap = LinearSegmentedColormap.from_list("earth_brown", ["#FFFFFF", "#3B1D14"])

    # ==========================================
    # DATA PREP
    # ==========================================
    limit = -1
    V_L = model.value_left[:limit, :, :]
    V_R = model.value_right[:limit, :, :]
    V_W = model.value_wait[:limit, :, :]
    
    # 1. RGB Probabilities (Row 1)
    Costs = np.stack([V_L, V_R, V_W], axis=-1)
    inv_temp = getattr(model, 'inv_temp', 1.0)
    logits = -Costs * inv_temp
    
    max_logits = np.nanmax(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    sum_exp = np.nansum(exp_logits, axis=-1, keepdims=True)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        probs = exp_logits / sum_exp
    probs = np.nan_to_num(probs, nan=0.0)

    img_rgb = (probs[..., 0:1] * rgb_L.reshape(1,1,1,3) + 
               probs[..., 1:2] * rgb_R.reshape(1,1,1,3) + 
               probs[..., 2:3] * rgb_W.reshape(1,1,1,3))
    img_rgb = np.clip(img_rgb, 0.0, 1.0)

    # 2. Action Preference Gap (Row 2)
    Preference = V_W - np.minimum(V_L, V_R)
    vmin = np.nanpercentile(Preference, 0)
    vmax = np.nanpercentile(Preference, 100)

    if selected_timesteps is None:
        selected_timesteps = [0, 29, 37, 39]
    n_plots = len(selected_timesteps)

    # ==========================================
    # PLOTTING LAYOUT
    # ==========================================
    fig, axes = plt.subplots(2, n_plots, figsize=(8.5, 4.2), dpi=dpi)
    
    for i, t in enumerate(selected_timesteps):
        # --------------------
        # ROW 1: RGB Softmax
        # --------------------
        ax1 = axes[0, i]
        ax1.imshow(img_rgb[t], extent=[0, 1, 0, 1], origin='lower', interpolation='nearest', aspect='equal')
        
        ax1.set_title(f"Time {t+1}", fontsize=style['title_fontsize'], pad=10)
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.tick_params(axis='both', which='major', labelsize=style['tick_fontsize'])
        
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.set_xticklabels([])  
        if i > 0: ax1.set_yticklabels([])

        # --------------------
        # ROW 2: Preference Gap
        # --------------------
        ax2 = axes[1, i]
        im_gap = ax2.imshow(Preference[t], extent=[0, 1, 0, 1], origin='lower', 
                            cmap=cmap_gap, vmin=vmin, vmax=vmax, interpolation='nearest', aspect='equal')
        
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.tick_params(axis='both', which='major', labelsize=style['tick_fontsize'])
        
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        if i > 0: ax2.set_yticklabels([])

    # ==========================================
    # GLOBAL AXIS LABELS
    # ==========================================
    plot_center_x = 0.42 # 绝对居中于子图网格
    fig.text(plot_center_x, 0.02, r"Stop Signal Belief $\zeta^\nu_t$", ha='center', va='center', fontsize=style['label_fontsize'])
    fig.text(0.04, 0.5, r"Go Belief $\beta^\nu_t$", ha='center', va='center', rotation='vertical', fontsize=style['label_fontsize'])

    # ==========================================
    # RIGHT SIDEBAR ANNOTATIONS (Pixel-perfect Aligned)
    # ==========================================
    sidebar_x = 0.77  # 侧边栏所有元素的统一起始X坐标

    # --- ROW 1 PANEL (Top Half) ---
    fig.text(sidebar_x, 0.881, "Probabilistic Policy\n(Softmax)", va='center', ha='left', 
             fontsize=style['label_fontsize'], fontweight='regular')
    
    legend_elements = [
        Patch(facecolor=colors['Q_L'], edgecolor='none', label='Go Left'),
        Patch(facecolor=colors['Q_R'], edgecolor='none', label='Go Right'),
        Patch(facecolor=colors['Q_W'], edgecolor='none', label='Wait'),
    ]
    # 【关键修改】：去除默认 padding，确保 Patch 的左边缘精准贴合 sidebar_x
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(sidebar_x, 0.72), 
               frameon=False, fontsize=style['legend_fontsize'], handlelength=1.5, handletextpad=0.8,
               borderaxespad=0., borderpad=0.)

    # --- ROW 2 PANEL (Bottom Half) ---
    # 【关键修改】：Y轴微调到0.47，避免文字碰到 Colorbar 的“Strong Go”
    fig.text(sidebar_x, 0.47, "Action Preference\n" + r"$Q(W) - \min(Q(L), Q(R))$", 
             va='bottom', ha='left', fontsize=style['label_fontsize'], fontweight='regular')
    
    # 【关键修改】：高度精准缩减至 0.315，以补偿 equal aspect 带来的高度折损，底部(0.12)完全齐平
    cbar_ax = fig.add_axes([sidebar_x, 0.15, 0.015, 0.3])
    cbar = fig.colorbar(im_gap, cax=cbar_ax, orientation='vertical')
    
    cbar.set_ticks([vmin, 0, vmax])
    cbar.set_ticklabels(['Strong Wait\nPreference', 'Decision\nBoundary\n(0)', 'Strong Go\nPreference'])
    cbar.ax.tick_params(labelsize=style['tick_fontsize'], length=0, pad=8)
    
    tick_labels = cbar.ax.get_yticklabels()
    tick_labels[0].set_verticalalignment('bottom')
    tick_labels[2].set_verticalalignment('top')

    # ==========================================
    # LAYOUT MARGINS
    # ==========================================
    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.75, hspace=0.10, wspace=0.1)

    if saveto:
        fig.savefig(saveto, dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
    else:
        plt.show()







    # Hide any unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    # ==========================================
    # LEGEND & FINISHING
    # ==========================================
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor=colors['Q_L'], label='Left'),
        Patch(facecolor=colors['Q_R'], label='Right'),
        Patch(facecolor=colors['Q_W'], label='Wait'),
    ]
    if trajectories:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                                     markersize=6, alpha=0.5, label='Agent State'))

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=4, frameon=False, fontsize=style['legend_fontsize'])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust to make room for legend

    if saveto:
        fig.savefig(saveto, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_policy_simple(model, saveto=None, dpi=300, selected_timesteps=None):
    """
    PLOS Comp Biol compliant version (V9) - Logit-based Policy Visualization.
    
    Updates:
    - Uses Logits (-Cost * inv_temp) for color intensity instead of Softmax Probs.
    - Normalizes Logits per timestep to map the "best" action to full color intensity.
    """
    # ==========================================
    # PLOS STYLING
    # ==========================================
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    style = {
        "label_fontsize": 10,
        "title_fontsize": 11,
        "tick_fontsize": 9,
        "legend_fontsize": 9,
    }

    # Define Base Colors
    colors = {
        "Q_L": "#E3B23C",  # Gold (Left)
        "Q_R": "#3A6EA5",  # Blue (Right)
        "Q_W": "#F1EFEFF3", # Off-White (Wait)
    }
    
    rgb_L = np.array(to_rgb(colors["Q_L"]))
    rgb_R = np.array(to_rgb(colors["Q_R"]))
    rgb_W = np.array(to_rgb(colors["Q_W"]))

    # ==========================================
    # DATA PREP: CALCULATE LOGITS
    # ==========================================
    limit = -1
    V_L = model.value_left[:limit, :, :]
    V_R = model.value_right[:limit, :, :]
    V_W = model.value_wait[:limit, :, :]
    
    # Stack into [Time, Beta, Zeta, Action]
    Costs = np.stack([V_L, V_R, V_W], axis=-1)
    
    inv_temp = getattr(model, 'inv_temp', 1.0)
    # Logits: Higher is better (lower cost)
    logits = -Costs * inv_temp
    
    # --- Normalization for Visualization ---
    # To map Logits to RGB, we scale them per state so the highest logit 
    # (best action) has the most influence.
    # Subtract the minimum logit to ensure all values are positive for the blend
    logits_min = np.nanmin(logits, axis=-1, keepdims=True)
    logits_shifted = logits - logits_min
    
    # Normalize by the sum of shifted logits to create mixing weights
    sum_logits = np.nansum(logits_shifted, axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = logits_shifted / sum_logits
    weights = np.nan_to_num(weights, nan=0.0)

    # 4. Create RGB Image via Linear Combination of Weights
    img_rgb = (weights[..., 0:1] * rgb_L.reshape(1,1,1,3) + 
               weights[..., 1:2] * rgb_R.reshape(1,1,1,3) + 
               weights[..., 2:3] * rgb_W.reshape(1,1,1,3))
    
    img_rgb = np.clip(img_rgb, 0.0, 1.0)

    # ==========================================
    # PLOTTING
    # ==========================================
    num_timesteps = Costs.shape[0]
    if selected_timesteps is None:
        selected_timesteps = [0, 10, 20, 30]
    
    selected_timesteps = [t for t in selected_timesteps if 0 <= t < num_timesteps]
    n_plots = len(selected_timesteps)

    fig, axes = plt.subplots(1, n_plots, figsize=(7.5, 2.6)) 
    if n_plots == 1: axes = [axes]
    
    for i, t in enumerate(selected_timesteps):
        ax = axes[i]
        ax.imshow(img_rgb[t], extent=[0, 1, 0, 1], origin='lower', 
                  interpolation='nearest', aspect='equal') 
        
        ax.set_title(f"Time {t + 1}", fontsize=style['title_fontsize'], pad=8)
        ax.set_xlabel(r"Stop Signal Belief $\zeta^\nu_t$", fontsize=style['label_fontsize'])
        
        if i == 0:
            ax.set_ylabel(r"Go Belief $\beta^\nu_t$", fontsize=style['label_fontsize'])
        else:
            ax.tick_params(labelleft=False)

        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(axis='both', which='major', labelsize=style['tick_fontsize'])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ==========================================
    # LEGEND
    # ==========================================
    legend_elements = [
        Patch(facecolor=colors['Q_L'], label='Left'),
        Patch(facecolor=colors['Q_R'], label='Right'),
        Patch(facecolor=colors['Q_W'], label='Wait'),
    ]

    fig.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.08, 0.88), 
               ncol=3, frameon=False, fontsize=style['legend_fontsize'],
               handlelength=2.5, handletextpad=1.2, columnspacing=2.5)

    plt.subplots_adjust(top=0.82, bottom=0.18, wspace=0.35, left=0.08, right=0.95)

    if saveto:
        fig.savefig(saveto, dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
    else:
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def plot_policy_with_trajectories(model, samples_go=None, samples_stop=None, saveto=None, dpi=300):
    """
    全量策略图 + 模拟信念轨迹叠加。
    
    参数:
    - model: POMDP 模型实例。
    - samples_go: simulation.simu_trial_batch 的 Go 试次输出。
    - samples_stop: simulation.simu_trial_batch 的 Stop 试次输出。
    """
    # ==========================================
    # 1. 样式与颜色设置
    # ==========================================
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    colors = {
        "Q_L": "#E3B23C",    # Go Left (金)
        "Q_R": "#3A6EA5",    # Go Right (蓝)
        "Q_W": "#F1EFEFF3",   # Wait (灰)
        "dot_go": "black",    # Go 试次的信念点颜色
        "dot_stop": "#800020" # Stop 试次的信念点颜色 (酒红)
    }
    rgb_L, rgb_R, rgb_W = np.array(to_rgb(colors["Q_L"])), np.array(to_rgb(colors["Q_R"])), np.array(to_rgb(colors["Q_W"]))

    # ==========================================
    # 2. 策略数据计算 (背景)
    # ==========================================
    limit = -1
    V_L, V_R, V_W = model.value_left[:limit, :, :], model.value_right[:limit, :, :], model.value_wait[:limit, :, :]
    Costs = np.stack([V_L, V_R, V_W], axis=-1)
    num_timesteps = Costs.shape[0]
    inv_temp = getattr(model, 'inv_temp', 1.0)
    logits = -Costs * inv_temp
    
    # 计算背景权重
    logits_min = np.nanmin(logits, axis=-1, keepdims=True)
    logits_shifted = logits - logits_min
    weights = np.nan_to_num(logits_shifted / np.nansum(logits_shifted, axis=-1, keepdims=True), nan=0.0)
    img_rgb_all = np.clip((weights[..., 0:1] * rgb_L.reshape(1,1,1,3) + 
                           weights[..., 1:2] * rgb_R.reshape(1,1,1,3) + 
                           weights[..., 2:3] * rgb_W.reshape(1,1,1,3)), 0, 1)

    # 等高线数据
    max_logits = np.nanmax(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = np.nan_to_num(exp_logits / np.nansum(exp_logits, axis=-1, keepdims=True), nan=0.0)
    wait_prob_all = probs[..., 2]

    # ==========================================
    # 3. 绘图网格 (8列 x 5行)
    # ==========================================
    cols = 8
    rows = (num_timesteps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.4), dpi=dpi)
    axes_flat = axes.flatten()

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i < num_timesteps:
            # A. 绘制策略地图背景
            # ax.imshow(img_rgb_all[i], extent=[0, 1, 0, 1], origin='lower', 
            #           interpolation='nearest', aspect='equal', alpha=0.7)
            
            # B. 叠加 P(Wait) 等高线 (辅助线)
            ny, nx = wait_prob_all[i].shape
            X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
            ax.contour(X, Y, wait_prob_all[i], levels=[0.5, 0.99], 
                       colors='k', linewidths=0.2, alpha=0.2)
            
            # C. 叠加当前时间步 t=i 的所有模拟信念点
            # 叠加 Go 试次点
            if samples_go is not None:
                for trial in samples_go:
                    if i < len(trial['beta_seq']):
                        ax.scatter(trial['sigma_seq'][i], trial['beta_seq'][i], 
                                   s=2, color=colors["dot_go"], alpha=0.5, zorder=5)
            
            # 叠加 Stop 试次点
            if samples_stop is not None:
                for trial in samples_stop:
                    if i < len(trial['beta_seq']):
                        ax.scatter(trial['sigma_seq'][i], trial['beta_seq'][i], 
                                   s=2, color=colors["dot_stop"], alpha=0.6, zorder=6)

            # 细节美化
            ax.set_title(f"T {i+1}", fontsize=8)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.tick_params(axis='both', labelsize=7)
            if i % cols != 0: ax.set_yticklabels([])
            if i // cols != rows - 1: ax.set_xticklabels([])
        else:
            ax.axis("off")

    # 全局标签与图例
    fig.text(0.5, 0.02, r"Stop Signal Belief $\zeta^\nu_t$", ha='center', fontsize=12)
    fig.text(0.06, 0.5, r"Go Belief $\beta^\nu_t$", ha='center', rotation='vertical', fontsize=12)

    legend_elements = [
        Patch(facecolor=colors['Q_L'], label='Go Left Pref.'),
        Patch(facecolor=colors['Q_R'], label='Go Right Pref.'),
        Patch(facecolor=colors['Q_W'], label='Wait Pref.'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['dot_go'], label='Go Trial States'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['dot_stop'], label='Stop Trial States'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
               ncol=5, frameon=False, fontsize=10)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.4, wspace=0.2)

    if saveto:
        fig.savefig(saveto, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


