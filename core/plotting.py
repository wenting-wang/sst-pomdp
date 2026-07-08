# Standard library imports
import sys
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

# Local application/library imports
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from core import simulation


def plot_belief(out_go, out_stop, show_cnt=5, saveto=None):
    """
    Generates a 2x3 composite figure illustrating the evolution of belief states 
    (Go belief and Stop trial belief) over time across five distinct behavioral outcomes.
    
    Args:
        out_go: Simulation output dictionary for Go trials.
        out_stop: Simulation output dictionary for Stop trials.
        show_cnt: Number of individual trial sample paths to overlay.
        saveto: Optional filepath to save the output figure.
    """
    
    # Configure PLOS-compliant typography and color palette
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    style = {
        "label_fontsize": 9,
        "title_fontsize": 10,
        "tick_fontsize": 8,
        "legend_fontsize": 9,
        "panel_label_size": 12,

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

    df_go = pd.DataFrame(out_go)
    df_stop = pd.DataFrame(out_stop)
    ssd = df_stop.iloc[0]['ssd'] if len(df_stop) > 0 else None

    # Categorize data by behavioral outcome
    df_gs = df_go[df_go['result'] == 'GS']
    df_ge = df_go[df_go['result'] == 'GE']
    df_gm = df_go[df_go['result'] == 'GM']
    df_ss = df_stop[df_stop['result'] == 'SS']
    df_se = df_stop[df_stop['result'] == 'SE']

    df_list = [df_gs, df_ge, df_gm, df_ss, df_se]
    condition_names = ['Go Success', 'Go Error', 'Go Missing', 'Stop Success', 'Stop Error']

    # Initialize standard journal-sized figure
    fig, ax = plt.subplots(2, 3, figsize=(7.5, 4.8))
    
    MAX_TIME_STEPS = 40
    xticks = [1, 10, 20, 30, 40]
    panel_labels = ['A', 'B', 'C', 'D', 'E']

    for i, df in enumerate(df_list):
        row, col = divmod(i, 3)
        axis = ax[row, col]

        axis.text(-0.25, 1.1, panel_labels[i], transform=axis.transAxes,
                  fontsize=style['panel_label_size'], fontweight='bold', va='top', ha='right')

        if len(df) == 0:
            axis.axis('off')
            continue

        # Extract truncated belief sequences and reaction times up to the decision point
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

        # Configure secondary y-axis for survival probability shadow
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

        ax2.set_ylabel('Survival Prob.', color='gray', fontsize=style['label_fontsize'])
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=style['tick_fontsize'])
        ax2.set_ylim(0, 1.05) 
        ax2.set_yticks([0, 1.0])
        ax2.set_yticklabels(["0", "1"])
        
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_color('gray')
        ax2.spines['right'].set_linewidth(0.5)

        # Plot aggregate mean and standard error for belief states
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

        # Overlay individual sample paths
        # subset_indices = list(range(min(show_cnt, len(df))))
        
        np.random.seed(42)
        subset_indices = np.random.choice(len(df), size=min(show_cnt, len(df)), replace=False)
        
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

        # Format subplot axes and boundaries
        if condition_names[i].startswith('Stop') and ssd is not None:
            # Assuming a standard 12-timestep Stop Signal Duration
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
        
        axis.set_ylabel(r"Belief State", fontsize=style['label_fontsize'])
        axis.set_xlabel("Time", fontsize=style['label_fontsize'])

    ax[1, 2].axis("off")
    
    # Increase horizontal space to accommodate internal Y-labels
    plt.subplots_adjust(hspace=0.5, wspace=0.55, top=0.88, bottom=0.12, left=0.08, right=0.92)

    # Construct global legend
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


def plot_action_value(out_go, out_stop, show_cnt=5, saveto=None):
    """
    Generates a 2x3 composite figure illustrating the action values Q(b, a) 
    over time across five distinct behavioral outcomes.
    
    Args:
        out_go: Simulation output dictionary for Go trials.
        out_stop: Simulation output dictionary for Stop trials.
        show_cnt: Number of individual trial sample paths to overlay.
        saveto: Optional filepath to save the output figure.
    """
    
    # Configure PLOS-compliant typography and color palette
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    style = {
        "label_fontsize": 9,
        "title_fontsize": 10,
        "tick_fontsize": 8,
        "legend_fontsize": 9,
        "panel_label_size": 12,

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

    df_go = pd.DataFrame(out_go)
    df_stop = pd.DataFrame(out_stop)
    ssd = df_stop.iloc[0]['ssd'] if len(df_stop) > 0 else None

    # Categorize data by behavioral outcome
    df_gs = df_go[df_go['result'] == 'GS']
    df_ge = df_go[df_go['result'] == 'GE']
    df_gm = df_go[df_go['result'] == 'GM']
    df_ss = df_stop[df_stop['result'] == 'SS']
    df_se = df_stop[df_stop['result'] == 'SE']

    df_list = [df_gs, df_ge, df_gm, df_ss, df_se]
    condition_names = ['Go Success', 'Go Error', 'Go Missing', 'Stop Success', 'Stop Error']

    # Initialize standard journal-sized figure
    fig, ax = plt.subplots(2, 3, figsize=(7.5, 4.8))
    
    MAX_TIME_STEPS = 40
    xticks = [1, 10, 20, 30, 40]
    panel_labels = ['A', 'B', 'C', 'D', 'E']

    for i, df in enumerate(df_list):
        row, col = divmod(i, 3)
        axis = ax[row, col]

        axis.text(-0.25, 1.1, panel_labels[i], transform=axis.transAxes,
                  fontsize=style['panel_label_size'], fontweight='bold', va='top', ha='right')

        if len(df) == 0:
            axis.axis('off')
            continue

        # Extract truncated action value sequences and reaction times up to the decision point
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

        # Configure secondary y-axis for survival probability shadow
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

        ax2.set_ylabel('Survival Prob.', color='gray', fontsize=style['label_fontsize'])
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=style['tick_fontsize'])
        ax2.set_ylim(0, 1.05) 
        ax2.set_yticks([0, 1.0])
        ax2.set_yticklabels(["0", "1"])
        
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_color('gray')
        ax2.spines['right'].set_linewidth(0.5)

        # Plot aggregate mean and standard error for action values
        df_ql = pd.DataFrame(q_l_trunc_list)
        df_qr = pd.DataFrame(q_r_trunc_list)
        df_qw = pd.DataFrame(q_w_trunc_list)

        def plot_mean_with_sem(data_df, color):
            """Helper to compute and plot the mean sequence with SEM shading."""
            mean = data_df.mean(axis=0)
            sem = data_df.sem(axis=0)
            x_vals = np.arange(1, len(mean) + 1)
            axis.plot(x_vals, mean, 
                      color=color, linewidth=style['mean_lw'], alpha=style['mean_alpha'])
            axis.fill_between(x_vals, mean - sem, mean + sem,
                              color=color, alpha=style['sem_alpha'], linewidth=0)

        # Layer order: Wait action (background), then active choices
        plot_mean_with_sem(df_qw, colors["Q_W"])
        plot_mean_with_sem(df_ql, colors["Q_L"])
        plot_mean_with_sem(df_qr, colors["Q_R"])

        # Overlay individual sample paths
        # subset_indices = list(range(min(show_cnt, len(df))))
        
        np.random.seed(42)
        subset_indices = np.random.choice(len(df), size=min(show_cnt, len(df)), replace=False)
        
        for j in subset_indices:
            d_idx = rt_indices[j]
            limit = d_idx + 1
            
            ql_seq = df.iloc[j]['value_left_seq']
            qr_seq = df.iloc[j]['value_right_seq']
            qw_seq = df.iloc[j]['value_wait_seq']
            
            x_sample = np.arange(1, limit + 1)
            is_last_step = (d_idx >= MAX_TIME_STEPS - 1)

            def plot_sample(seq, color):
                """Helper to plot individual trial traces and decision markers."""
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

        # Format subplot axes and boundaries
        if condition_names[i].startswith('Stop') and ssd is not None:
            # Assuming a standard 12-timestep Stop Signal Duration
            axis.axvline(x=ssd + 1, color='k', linestyle='--', linewidth=style['ssd_lw'], alpha=style['ssd_alpha'])
            axis.axvline(x=ssd + 12 + 1, color='k', linestyle='--', linewidth=style['ssd_lw'], alpha=style['ssd_alpha'])

        axis.set_xlim(1, MAX_TIME_STEPS)
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
    
    plt.subplots_adjust(hspace=0.5, wspace=0.55, top=0.88, bottom=0.12, left=0.08, right=0.92)

    # Construct global legend
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


def plot_stop_prior_sensitivity(
    model,
    params: dict,
    rates=(0.05, 0.10, 1/6, 0.20, 0.25, 0.30),
    ssds=np.arange(2, 35, 4), 
    n_session=40,  
    saveto=None,
    min_rt_trials=3 
):
    """
    Plots the sensitivity of behavioral metrics to varying stop prior probabilities
    and Stop Signal Delays (SSDs).
    Generates a 1x3 composite figure illustrating Go outcomes and parameterized 
    Stop outcomes across the SSD spectrum with a discretized color legend.
    
    Args:
        model: The computational model object.
        params: Dictionary of model parameters.
        rates: Tuple of stop trial prior probabilities to simulate.
        ssds: Array of Stop Signal Delays to sweep (should be evenly spaced integers).
        n_session: Number of simulated trials per rate/SSD condition.
        saveto: Optional filepath to save the output figure.
        min_rt_trials: Minimum number of trials required to plot an RT data point.
    """
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    style = {
        "label_fontsize": 10,     
        "tick_fontsize": 9,      
        "legend_fontsize": 10,    
        "panel_label_size": 14,  
        "line_width": 1.2,      
        "marker_size": 4.5,        
        "cap_size": 1.5,
    }

    color_primary = '#000000' 
    
    res_stop_acc = {s: [] for s in ssds}
    res_stop_acc_err = {s: [] for s in ssds}
    res_stop_rt = {s: [] for s in ssds}
    res_stop_rt_err = {s: [] for s in ssds}
    
    go_acc_list = []
    go_acc_err = []
    go_rt_list = []
    go_rt_err = []

    print("Simulating multidimensional sensitivity (Prior x SSD)...")
    
    for r in rates:
        p = params.copy()
        p['rate_stop_trial'] = float(r)
        
        m = model.__class__(**p)
        m.value_iteration_tensor()

        all_go_df_list = []
        
        for current_ssd in ssds:
            out_sess = []
            for _ in range(n_session):
                is_stop = np.random.rand() < float(r)
                gs = np.random.choice(["left", "right"])
                type_ = "stop" if is_stop else "nonstop"
                ssd_ = current_ssd if is_stop else None
                
                res_tuple = simulation.simu_trial(m, gs, type_, ssd_, verbose=False) 
                out_sess.append(res_tuple[:3])
            
            df_beh = pd.DataFrame(out_sess, columns=["result", "rt", "ssd"])
            
            go_df = df_beh[df_beh["result"].isin(["GS", "GE", "GM"])]
            all_go_df_list.append(go_df)
            
            stop_df = df_beh[df_beh["result"].isin(["SS", "SE"])]
            if len(stop_df) > 0:
                s_acc = (stop_df["result"] == "SS").mean()
                s_acc_err = np.sqrt(s_acc * (1 - s_acc) / len(stop_df))
                
                se_df = stop_df[stop_df["result"] == "SE"]
                if len(se_df) >= min_rt_trials:
                    s_rt = se_df["rt"].mean()
                    s_rt_err = se_df["rt"].std(ddof=1) / np.sqrt(len(se_df))
                else:
                    s_rt, s_rt_err = np.nan, np.nan
            else:
                s_acc, s_acc_err, s_rt, s_rt_err = [np.nan] * 4
                
            res_stop_acc[current_ssd].append(s_acc)
            res_stop_acc_err[current_ssd].append(s_acc_err)
            res_stop_rt[current_ssd].append(s_rt)
            res_stop_rt_err[current_ssd].append(s_rt_err)

        all_go_df = pd.concat(all_go_df_list)
        if len(all_go_df) > 0:
            g_acc = (all_go_df["result"] == "GS").mean()
            g_acc_err = np.sqrt(g_acc * (1 - g_acc) / len(all_go_df))
            
            gs_df = all_go_df[all_go_df["result"] == "GS"]
            if len(gs_df) >= min_rt_trials:
                g_rt = gs_df["rt"].mean()
                g_rt_err = gs_df["rt"].std(ddof=1) / np.sqrt(len(gs_df))
            else:
                g_rt, g_rt_err = np.nan, np.nan
        else:
            g_acc, g_acc_err, g_rt, g_rt_err = [np.nan] * 4
            
        go_acc_list.append(g_acc)
        go_acc_err.append(g_acc_err)
        go_rt_list.append(g_rt)
        go_rt_err.append(g_rt_err)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    ax_go, ax_s_acc, ax_s_rt = axes
    
    plt.subplots_adjust(wspace=0.45, bottom=0.20, top=0.80, right=0.88)

    # --- Panel A: Go Trials (Twin Axes) ---
    ax_go.plot(rates, go_acc_list, color=color_primary, linestyle='-', lw=style['line_width'], zorder=1)
    ax_go.errorbar(rates, go_acc_list, yerr=go_acc_err, fmt='none', ecolor=color_primary, capsize=style['cap_size'], zorder=1)
    ax_go.scatter(rates, go_acc_list, s=30, color=color_primary, marker='o', zorder=2)
    
    ax_go.set_ylabel("Accuracy", color=color_primary, fontsize=style['label_fontsize'])
    ax_go.set_ylim(-0.05, 1.05)
    ax_go.set_yticks([0, 0.5, 1.0])
    ax_go.tick_params(axis='y', colors=color_primary, labelsize=style['tick_fontsize'])
    ax_go.spines['left'].set_color(color_primary)
    ax_go.spines['top'].set_visible(False)
    ax_go.set_title("Go Trials", fontsize=style['label_fontsize'] + 1, pad=10)

    ax_go_rt = ax_go.twinx()
    ax_go_rt.plot(rates, go_rt_list, color=color_primary, linestyle='--', lw=style['line_width'], zorder=1)
    ax_go_rt.errorbar(rates, go_rt_list, yerr=go_rt_err, fmt='none', ecolor=color_primary, capsize=style['cap_size'], zorder=1)
    ax_go_rt.scatter(rates, go_rt_list, s=30, color=color_primary, marker='s', zorder=2)
    
    ax_go_rt.set_ylabel("Reaction Time", color=color_primary, fontsize=style['label_fontsize'])
    
    ax_go_rt.set_ylim(-2, 40)
    ax_go_rt.set_yticks([0, 10, 20, 30, 40])
    
    # ax_go_rt.set_ylim(-2, 30)
    # ax_go_rt.set_yticks([0, 5, 10, 15, 20, 25, 30])
    
    ax_go_rt.tick_params(axis='y', colors=color_primary, labelsize=style['tick_fontsize'])
    ax_go_rt.spines['right'].set_color(color_primary)
    ax_go_rt.spines['top'].set_visible(False)

    h_acc = Line2D([0], [0], color=color_primary, marker='o', linestyle='-', lw=style['line_width'], label="Accuracy")
    h_rt  = Line2D([0], [0], color=color_primary, marker='s', linestyle='--', lw=style['line_width'], label="Reaction Time")
    fig.legend(handles=[h_acc, h_rt], loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False, fontsize=style['legend_fontsize'])

    # --- Panel B & C: Stop Trials (SSD Sweep) ---
    # Create a discrete colormap based on the number of SSDs
    num_ssds = len(ssds)
    cmap = LinearSegmentedColormap.from_list('trunc_grey', ['#B0B0B0', '#000000'], N=num_ssds)
    
    for i, current_ssd in enumerate(ssds):
        color = cmap(i)
        
        ax_s_acc.plot(rates, res_stop_acc[current_ssd], color=color, lw=style['line_width'], ls='-', zorder=1)
        ax_s_acc.errorbar(rates, res_stop_acc[current_ssd], yerr=res_stop_acc_err[current_ssd], fmt='o', color=color, markersize=style['marker_size'], capsize=style['cap_size'], zorder=2)
        
        ax_s_rt.plot(rates, res_stop_rt[current_ssd], color=color, lw=style['line_width'], ls='--', zorder=1)
        ax_s_rt.errorbar(rates, res_stop_rt[current_ssd], yerr=res_stop_rt_err[current_ssd], fmt='s', color=color, markersize=style['marker_size'], capsize=style['cap_size'], zorder=2)

    ax_s_acc.set_ylabel("Accuracy", fontsize=style['label_fontsize'])
    ax_s_acc.set_ylim(-0.05, 1.05)
    ax_s_acc.set_yticks([0, 0.5, 1.0])
    ax_s_acc.set_title("Stop Trials", fontsize=style['label_fontsize'] + 1, pad=10)
    ax_s_acc.spines['top'].set_visible(False)
    ax_s_acc.spines['right'].set_visible(False)

    ax_s_rt.set_ylabel("Reaction Time", fontsize=style['label_fontsize'])
    
    ax_s_rt.set_ylim(-2, 40)
    ax_s_rt.set_yticks([0, 10, 20, 30, 40])
    
    # ax_s_rt.set_ylim(-2, 30)
    # ax_s_rt.set_yticks([0, 5, 10, 15, 20, 25, 30])
    
    ax_s_rt.set_title("Stop Trials", fontsize=style['label_fontsize'] + 1, pad=10)
    ax_s_rt.spines['top'].set_visible(False)
    ax_s_rt.spines['right'].set_visible(False)

    x_min, x_max = min(rates), max(rates)
    padding = (x_max - x_min) * 0.1 if x_max > x_min else 0.05
    tick_locs = [float(r) for r in rates]
    
    for ax in axes:
        ax.set_xlabel(r"Stop Prior $\tilde{r}^\nu$", fontsize=style['label_fontsize'])
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_xticks(tick_locs)
        ax.set_xticklabels([f"{r:.2f}" for r in tick_locs], rotation=45)
        ax.tick_params(axis='x', labelsize=style['tick_fontsize'])

    # --- Discrete Colorbar Setup ---
    # Calculate bounds centered around the SSD values to create discrete bins
    step = ssds[1] - ssds[0] if len(ssds) > 1 else 1
    bounds = np.arange(ssds[0] - step/2, ssds[-1] + step, step)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    cax = fig.add_axes([0.91, 0.20, 0.015, 0.60])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cax, ticks=ssds, spacing='uniform')
    
    cbar.set_label('Stop Signal Delay (SSD)', fontsize=style['label_fontsize'])
    cbar.ax.set_yticklabels([str(s) for s in ssds])
    
    # Explicitly format major ticks and turn off all minor ticks
    cbar.ax.tick_params(axis='y', which='major', labelsize=style['tick_fontsize'])
    cbar.ax.minorticks_off() 
    
    cbar.outline.set_visible(False)

    labels = ['A', 'B', 'C']
    for ax, label in zip(axes, labels):
        ax.text(-0.25, 1.12, label, transform=ax.transAxes, 
                fontsize=style['panel_label_size'], fontweight='bold', va='top', ha='right')

    if saveto:
        fig.savefig(saveto, dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
    else:
        plt.show()


def plot_policy_summary(model, out_dict, selected_timesteps, saveto=None, dpi=300, gamma=0.3):
    """
    Generates a publication-quality compact summary plot of the policy across selected timesteps.
    
    Args:
        model: The computational model containing value arrays.
        out_dict: Dictionary containing simulation outcomes.
        selected_timesteps: List of specific timesteps to visualize.
        saveto: Optional filepath to save the figure.
        dpi: Resolution for the saved figure.
        gamma: Gamma correction factor for visual clarity of probabilities.
    """
    
    # Configure PLOS-compliant typography and color palette
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    colors = {
        "Q_L": "#E3B23C",   # Gold (Left)
        "Q_R": "#3A6EA5",   # Blue (Right)
        "Q_W": "#F7F7F7",   # Light Gray (Wait)
    }
    
    rgb_L = np.array(to_rgb(colors["Q_L"]))
    rgb_R = np.array(to_rgb(colors["Q_R"]))
    rgb_W = np.array(to_rgb(colors["Q_W"]))

    style = {
        "title_fontsize": 11,
        "label_fontsize": 12, 
        "tick_fontsize": 9,
        "legend_fontsize": 10,
    }

    # Compute action probabilities using softmax and apply gamma correction
    limit = -1
    V_L = model.value_left[:limit, :, :]
    V_R = model.value_right[:limit, :, :]
    V_W = model.value_wait[:limit, :, :]
    
    Costs = np.stack([V_L, V_R, V_W], axis=-1)
    
    inv_temp = getattr(model, 'inv_temp', 1.0)
    logits = -Costs * inv_temp
    
    max_logits = np.nanmax(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    sum_exp = np.nansum(exp_logits, axis=-1, keepdims=True)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        probs = exp_logits / sum_exp
    probs = np.nan_to_num(probs, nan=0.0)

    probs_vis = probs ** gamma
    probs_vis = probs_vis / np.sum(probs_vis, axis=-1, keepdims=True)

    img_rgb_all = (probs_vis[..., 0:1] * rgb_L.reshape(1,1,1,3) + 
                   probs_vis[..., 1:2] * rgb_R.reshape(1,1,1,3) + 
                   probs_vis[..., 2:3] * rgb_W.reshape(1,1,1,3))
    img_rgb_all = np.clip(img_rgb_all, 0.0, 1.0)

    def get_belief_scatter(data, t):
        """Helper to extract belief states (zeta, beta) for a given timestep."""
        if data is None or len(data) == 0:
            return [], []
        betas, zetas = [], []
        for trial in data:
            b_seq = trial.get('beta_seq', [])
            z_seq = trial.get('sigma_seq', [])
            
            rt = trial.get('rt')
            if rt is None:
                rt = float('inf')
            
            if t < len(b_seq) and t < len(z_seq) and t <= rt:
                betas.append(b_seq[t])
                zetas.append(z_seq[t])
                
        return zetas, betas

    def draw_right_triangle_legend(fig, rect, gamma, colors_dict, angle_deg=135):
        """Helper to draw a rotated colormap legend with shifted axis ticks."""
        x_orig, y_orig, w_orig, h_orig = rect
        w_new, h_new = w_orig * 3.0, h_orig * 3.0
        x_new, y_new = x_orig - w_orig * 1.0, y_orig - h_orig * 1.0
        
        ax_leg = fig.add_axes([x_new, y_new, w_new, h_new])
        ax_leg.axis('off')
        
        n = 300
        x_vals = np.linspace(-0.5, 1.5, n)
        y_vals = np.linspace(-0.5, 1.5, n)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        cx, cy = 0.5, 0.5
        theta_inv = np.deg2rad(-angle_deg)
        c_inv, s_inv = np.cos(theta_inv), np.sin(theta_inv)
        
        X_orig = (X - cx) * c_inv - (Y - cy) * s_inv + cx
        Y_orig = (X - cx) * s_inv + (Y - cy) * c_inv + cy
        
        V_R = X_orig
        V_L = Y_orig
        V_W = 1.0 - X_orig - Y_orig
        
        valid = (V_R >= -0.001) & (V_L >= -0.001) & (V_W >= -0.001)
        V_R, V_L, V_W = np.clip(V_R, 0, 1), np.clip(V_L, 0, 1), np.clip(V_W, 0, 1)
        
        r_W = np.array(to_rgb(colors_dict["Q_W"]))
        r_L = np.array(to_rgb(colors_dict["Q_L"]))
        r_R = np.array(to_rgb(colors_dict["Q_R"]))
        
        img = np.zeros((n, n, 4))
        img[valid, :3] = V_W[valid, None]*r_W + V_L[valid, None]*r_L + V_R[valid, None]*r_R
        img[valid, 3] = 1.0 
        
        ax_leg.imshow(img, origin='lower', extent=[-0.5, 1.5, -0.5, 1.5], clip_on=False)
        
        def fwd_rot(px, py):
            t = np.deg2rad(angle_deg)
            return (px - cx) * np.cos(t) - (py - cy) * np.sin(t) + cx, \
                   (px - cx) * np.sin(t) + (py - cy) * np.cos(t) + cy

        def true_to_visual(p, g):
            if p == 0: return 0.0
            if p == 1: return 1.0
            return (p**g) / (p**g + (1-p)**g)


        tick_labels = ['0', '.33', '.67', '1']
        tick_pos = [0.0, 0.333, 0.667, 1.0] 
        
        ax_leg.plot(*zip(fwd_rot(0,0), fwd_rot(1,0)), color='black', lw=1, clip_on=False)
        ax_leg.plot(*zip(fwd_rot(0,0), fwd_rot(0,1)), color='black', lw=1, clip_on=False)
        
        base_pad = -0.22
        
        for p_vis, lbl in zip(tick_pos, tick_labels):
            # Go Right Axis (Top)
            tx, ty = fwd_rot(p_vis, 0)
            dx, dy = fwd_rot(p_vis, -0.04)
            ax_leg.plot([tx, dx], [ty, dy], color='black', lw=0.8, clip_on=False)
            
            lx, ly = fwd_rot(p_vis, base_pad)
            ax_leg.text(lx, ly, lbl, rotation=0, ha='center', va='center', fontsize=8, fontstretch='ultra-condensed')
            
            # Go Left Axis (Bottom)
            tx2, ty2 = fwd_rot(0, p_vis)
            dx2, dy2 = fwd_rot(-0.04, p_vis)
            ax_leg.plot([tx2, dx2], [ty2, dy2], color='black', lw=0.8, clip_on=False)
            
            lx2, ly2 = fwd_rot(base_pad, p_vis)
            ax_leg.text(lx2, ly2, lbl, rotation=0, ha='center', va='center', fontsize=8, fontstretch='ultra-condensed')

        # Placement for outer peripheral text labels
        label_color = "#000000"
        ax_leg.text(0.5, 2, 'Softmax Policy', ha='left', va='bottom', fontsize=10, color=label_color) 
        ax_leg.text(0.5, 1.55, 'Go Right', ha='left', va='bottom', fontsize=9, color=label_color)   
        ax_leg.text(0.5, -0.55, 'Go Left', ha='left', va='top', fontsize=9, color=label_color)      
        ax_leg.text(0.27, 0.5, 'Prob.', rotation=90, ha='center', va='center', fontsize=9, color="#565555") 
        wx, wy = fwd_rot(0, 0)
        ax_leg.text(wx + 0.35, wy, 'Wait', ha='left', va='center', fontsize=9, color=label_color) 

    # Initialize dynamic grid layout based on outcomes and selected timesteps
    n_rows = len(out_dict)
    n_cols = len(selected_timesteps)
    outcome_names = list(out_dict.keys())
    
    go_keys = ["Go Success", "Go Error", "Go Missing"]
    stop_keys = ["Stop Success", "Stop Error"]
    
    total_go = sum([len(out_dict[k]) for k in go_keys if k in out_dict and out_dict[k] is not None])
    total_stop = sum([len(out_dict[k]) for k in stop_keys if k in out_dict and out_dict[k] is not None])
    total_trials = total_go + total_stop
    
    fig_width = 7.5 
    fig_height = max(4.0, n_rows * 1.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=dpi)
    
    if n_rows == 1: axes = np.expand_dims(axes, axis=0)
    if n_cols == 1: axes = np.expand_dims(axes, axis=1)

    for r in range(n_rows):
        out_name = outcome_names[r]
        out_data = out_dict[out_name]
        
        count = len(out_data) if out_data is not None else 0
        if out_name in go_keys:
            pct = (count / total_trials * 100) if total_trials > 0 else 0
        elif out_name in stop_keys:
            pct = (count / total_trials * 100) if total_trials > 0 else 0
        else:
            pct = 0.0
        
        for c in range(n_cols):
            t = selected_timesteps[c]
            ax = axes[r, c]
            
            if t < Costs.shape[0]:
                ax.imshow(img_rgb_all[t], extent=[0, 1, 0, 1], origin='lower', 
                          interpolation='nearest', aspect='equal')
                
                zetas_data, betas_data = get_belief_scatter(out_data, t)
                if zetas_data:
                    points = np.round(np.column_stack((zetas_data, betas_data)), 4)
                    unique_pts, counts = np.unique(points, axis=0, return_counts=True)
                    sizes = 2 + np.log1p(counts) * 8 
                    ax.scatter(unique_pts[:,0], unique_pts[:,1], 
                               c='#222222', s=sizes, alpha=0.75, 
                               edgecolors='none', zorder=10, clip_on=False)

            if r == 0:
                ax.set_title(f"Time {t+1}", fontsize=style['title_fontsize'], pad=8, fontweight='normal')
            
            if c == 0:
                row_label = f"{out_name}\n({pct:.1f}%)"
                ax.text(-0.95, 0.5, row_label, transform=ax.transAxes, 
                        ha='center', va='center', rotation=90, 
                        fontsize=style['label_fontsize'], fontweight='normal')

            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(axis='both', which='major', labelsize=style['tick_fontsize'], length=2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if c != 0: ax.set_yticklabels([])
            else: ax.set_yticklabels(['0', '0.5', '1'])
                
            if r != n_rows - 1: ax.set_xticklabels([])
            else: ax.set_xticklabels(['0', '0.5', '1'])

    # Add global axis labels and legend placement
    fig.text(0.44, 0.05, r"Stop Signal Belief $\zeta^\nu_t$", ha='center', va='center', fontsize=style['label_fontsize'])
    fig.text(0.079, 0.5, r"Go Belief $\beta^\nu_t$", ha='center', va='center', rotation='vertical', fontsize=style['label_fontsize'])

    draw_right_triangle_legend(fig, [0.7, 0.65, 0.068, 0.068], gamma, colors, angle_deg=135)

    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor='#222222', markersize=6, alpha=0.75, 
    #            label='Belief States') 
    # ]
    # fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.68, 0.54), 
    #            frameon=False, fontsize=style['legend_fontsize'], handletextpad=1.5)
    

    example_counts = [1, 10, 100, 1000]
    legend_elements = []
    
    for count in example_counts:
        s_val = 2 + np.log1p(count) * 8
        ms = np.sqrt(s_val)
        legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#222222', 
               markeredgecolor='none',
               markersize=ms, alpha=0.75, label=f'N = {count}')
    )

    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.68, 0.54), 
           frameon=False, fontsize=style['legend_fontsize'], handletextpad=1.5,
           title="Belief State Frequency", title_fontsize=style['legend_fontsize'])
    
    
    plt.subplots_adjust(top=0.90, bottom=0.10, left=0.14, right=0.68, hspace=0.15, wspace=0.15)

    if saveto:
        fig.savefig(saveto, dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
    else:
        plt.show()
        

def plot_policy(model, out_dict, outcome_name, saveto=None, dpi=300, gamma=0.3):
    """
    Generates a publication-quality scatter plot of the policy over time.
    
    Args:
        model: The computational model containing value arrays.
        out_dict: Dictionary containing simulation outcomes.
        outcome_name: The specific outcome key to plot (e.g., 'Go Success').
        saveto: Optional filepath to save the figure.
        dpi: Resolution for the saved figure.
        gamma: Gamma correction factor for visual clarity of probabilities.
    """
    
    # Configure PLOS-compliant typography and color palette
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    colors = {
        "Q_L": "#E3B23C",   # Gold (Left)
        "Q_R": "#3A6EA5",   # Blue (Right)
        "Q_W": "#F7F7F7",   # Light Gray (Wait)
    }
    
    rgb_L = np.array(to_rgb(colors["Q_L"]))
    rgb_R = np.array(to_rgb(colors["Q_R"]))
    rgb_W = np.array(to_rgb(colors["Q_W"]))

    style = {
        "title_fontsize": 10,
        "label_fontsize": 12, 
        "tick_fontsize": 8,
        "legend_fontsize": 10,
    }

    # Calculate the percentage of the selected outcome relative to its trial type
    go_keys = ["Go Success", "Go Error", "Go Missing"]
    stop_keys = ["Stop Success", "Stop Error"]
    
    total_go = sum([len(out_dict[k]) for k in go_keys if k in out_dict and out_dict[k] is not None])
    total_stop = sum([len(out_dict[k]) for k in stop_keys if k in out_dict and out_dict[k] is not None])
    total_trials = total_go + total_stop
    
    out_data = out_dict.get(outcome_name, [])
    count = len(out_data) if out_data is not None else 0
    
    if outcome_name in go_keys:
        pct = (count / total_trials * 100) if total_trials > 0 else 0
    elif outcome_name in stop_keys:
        pct = (count / total_trials * 100) if total_trials > 0 else 0
    else:
        pct = 0.0

    # Compute action probabilities using softmax and apply gamma correction
    limit = -1
    V_L = model.value_left[:limit, :, :]
    V_R = model.value_right[:limit, :, :]
    V_W = model.value_wait[:limit, :, :]
    
    Costs = np.stack([V_L, V_R, V_W], axis=-1)
    num_timesteps = Costs.shape[0]

    inv_temp = getattr(model, 'inv_temp', 1.0)
    logits = -Costs * inv_temp
    
    max_logits = np.nanmax(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    sum_exp = np.nansum(exp_logits, axis=-1, keepdims=True)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        probs = exp_logits / sum_exp
    probs = np.nan_to_num(probs, nan=0.0)

    probs_vis = probs ** gamma
    probs_vis = probs_vis / np.sum(probs_vis, axis=-1, keepdims=True)

    img_rgb_all = (probs_vis[..., 0:1] * rgb_L.reshape(1,1,1,3) + 
                   probs_vis[..., 1:2] * rgb_R.reshape(1,1,1,3) + 
                   probs_vis[..., 2:3] * rgb_W.reshape(1,1,1,3))
    img_rgb_all = np.clip(img_rgb_all, 0.0, 1.0)

    def get_belief_scatter(data, t):
        """Helper to extract belief states (zeta, beta) for a given timestep."""
        if data is None or len(data) == 0:
            return [], []
        betas, zetas = [], []
        for trial in data:
            b_seq = trial.get('beta_seq', [])
            z_seq = trial.get('sigma_seq', [])
            
            rt = trial.get('rt')
            if rt is None:
                rt = float('inf')
            
            if t < len(b_seq) and t < len(z_seq) and t <= rt:
                betas.append(b_seq[t])
                zetas.append(z_seq[t])
                
        return zetas, betas

    def draw_right_triangle_legend(fig, rect, gamma, colors_dict, angle_deg=135):
        """Helper to draw a rotated colormap legend for the three-way policy."""
        x_orig, y_orig, w_orig, h_orig = rect
        w_new, h_new = w_orig * 3.0, h_orig * 3.0
        x_new, y_new = x_orig - w_orig * 1.0, y_orig - h_orig * 1.0
        
        ax_leg = fig.add_axes([x_new, y_new, w_new, h_new])
        ax_leg.axis('off')
        
        n = 300
        x_vals = np.linspace(-0.5, 1.5, n)
        y_vals = np.linspace(-0.5, 1.5, n)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        cx, cy = 0.5, 0.5
        theta_inv = np.deg2rad(-angle_deg)
        c_inv, s_inv = np.cos(theta_inv), np.sin(theta_inv)
        
        X_orig = (X - cx) * c_inv - (Y - cy) * s_inv + cx
        Y_orig = (X - cx) * s_inv + (Y - cy) * c_inv + cy
        
        V_R = X_orig
        V_L = Y_orig
        V_W = 1.0 - X_orig - Y_orig
        
        valid = (V_R >= -0.001) & (V_L >= -0.001) & (V_W >= -0.001)
        V_R, V_L, V_W = np.clip(V_R, 0, 1), np.clip(V_L, 0, 1), np.clip(V_W, 0, 1)
        
        r_W = np.array(to_rgb(colors_dict["Q_W"]))
        r_L = np.array(to_rgb(colors_dict["Q_L"]))
        r_R = np.array(to_rgb(colors_dict["Q_R"]))
        
        img = np.zeros((n, n, 4))
        img[valid, :3] = V_W[valid, None]*r_W + V_L[valid, None]*r_L + V_R[valid, None]*r_R
        img[valid, 3] = 1.0 
        
        ax_leg.imshow(img, origin='lower', extent=[-0.5, 1.5, -0.5, 1.5], clip_on=False)
        
        def fwd_rot(px, py):
            t = np.deg2rad(angle_deg)
            return (px - cx) * np.cos(t) - (py - cy) * np.sin(t) + cx, \
                   (px - cx) * np.sin(t) + (py - cy) * np.cos(t) + cy

        def true_to_visual(p, g):
            if p == 0: return 0.0
            if p == 1: return 1.0
            return (p**g) / (p**g + (1-p)**g)

        tick_labels = ['0', '.33', '.67', '1']
        tick_pos = [0.0, 0.333, 0.667, 1.0] 
        
        ax_leg.plot(*zip(fwd_rot(0,0), fwd_rot(1,0)), color='black', lw=1, clip_on=False)
        ax_leg.plot(*zip(fwd_rot(0,0), fwd_rot(0,1)), color='black', lw=1, clip_on=False)
        
        base_pad = -0.22
        
        for p_vis, lbl in zip(tick_pos, tick_labels):
            # Go Right Axis (Top)
            tx, ty = fwd_rot(p_vis, 0)
            dx, dy = fwd_rot(p_vis, -0.04)
            ax_leg.plot([tx, dx], [ty, dy], color='black', lw=0.8, clip_on=False)
            
            lx, ly = fwd_rot(p_vis, base_pad)
            ax_leg.text(lx, ly, lbl, rotation=0, ha='center', va='center', fontsize=8, fontstretch='ultra-condensed')
            
            # Go Left Axis (Bottom)
            tx2, ty2 = fwd_rot(0, p_vis)
            dx2, dy2 = fwd_rot(-0.04, p_vis)
            ax_leg.plot([tx2, dx2], [ty2, dy2], color='black', lw=0.8, clip_on=False)
            
            lx2, ly2 = fwd_rot(base_pad, p_vis)
            ax_leg.text(lx2, ly2, lbl, rotation=0, ha='center', va='center', fontsize=8, fontstretch='ultra-condensed')
       
        label_color = "#000000"
        offset_right = 0.4
        
        gx_r, gy_r = fwd_rot(1, 0)
        ax_leg.text(gx_r + offset_right, gy_r + 0.15, 'Go Right', ha='left', va='center', fontsize=9, color=label_color)
        
        gx_l, gy_l = fwd_rot(0, 1)
        ax_leg.text(gx_l + offset_right, gy_l - 0.15, 'Go Left', ha='left', va='center', fontsize=9, color=label_color)
        
        wx, wy = fwd_rot(0, 0)
        ax_leg.text(wx + offset_right, wy, 'Wait', ha='left', va='center', fontsize=9, color=label_color) 
        
        ax_leg.text(0.27, 0.5, 'Prob.', rotation=90, ha='center', va='center', fontsize=9, color="#565555") 

    # Initialize grid layout for timesteps
    cols = 8
    rows = (num_timesteps + cols - 1) // cols
    
    fig_width = cols * 1.2
    fig_height = rows * 1.3 + 1.8  
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=dpi)
    axes = axes.flatten()

    for i in range(len(axes)):
        ax = axes[i]
        
        if i < num_timesteps:
            ax.imshow(img_rgb_all[i], extent=[0, 1, 0, 1], origin='lower', 
                      interpolation='nearest', aspect='equal')
            
            zetas_data, betas_data = get_belief_scatter(out_data, i)
            if zetas_data:
                points = np.round(np.column_stack((zetas_data, betas_data)), 4)
                unique_pts, counts = np.unique(points, axis=0, return_counts=True)
                
                sizes = 2 + np.log1p(counts) * 6
                ax.scatter(unique_pts[:,0], unique_pts[:,1], 
                           c='#222222', s=sizes, alpha=0.75, 
                           edgecolors='none', zorder=10, clip_on=False)

            ax.set_title(f"Time {i+1}", fontsize=style['tick_fontsize'], pad=4)
            
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(['0', '0.5', '1'])
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(['0', '0.5', '1'])
            ax.tick_params(axis='both', which='major', labelsize=style['tick_fontsize'], length=2)
            
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            row_idx = i // cols
            col_idx = i % cols
            if col_idx != 0: ax.set_yticklabels([])
            if row_idx != rows - 1: ax.set_xticklabels([])
        else:
            ax.axis("off")

    # Add global axis labels and legend placement
    fig.text(0.5, 0.02, r"Stop Signal Belief $\zeta^\nu_t$", ha='center', va='center', fontsize=style['label_fontsize'])
    fig.text(0.04, 0.5, r"Go Belief $\beta^\nu_t$", ha='center', va='center', rotation='vertical', fontsize=style['label_fontsize'])

    base_y = 0.81
    leg_w = 0.05
    leg_h = leg_w * (fig_width / fig_height)
    
    draw_right_triangle_legend(fig, [0.1, base_y - leg_h/2, leg_w, leg_h], gamma, colors, angle_deg=135)
    fig.text(0.27, base_y, 'Softmax Policy', ha='left', va='center', fontsize=10, color='#000000')

    # label_str = f'Belief States ({outcome_name}, {pct:.1f}%)'
    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor='#222222', markersize=6, alpha=0.75, 
    #            label=label_str)
    # ]
    # fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.58, base_y), 
    #            frameon=False, fontsize=style['legend_fontsize'])

    label_str = f'Belief States ({outcome_name}, {pct:.1f}%)'
    example_counts = [1, 10, 100, 1000] 
    legend_elements = []

    for count in example_counts:
        s_val = 2 + np.log1p(count) * 8
        ms = np.sqrt(s_val)
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#222222', 
                   markeredgecolor='none', markersize=ms, alpha=0.75, label=f'N = {count}')
        )

    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.58, base_y), 
               frameon=False, fontsize=style['legend_fontsize'], handletextpad=1.5,
               title=label_str, title_fontsize=style['legend_fontsize'])


    plt.subplots_adjust(top=0.68, bottom=0.08, left=0.08, right=0.95, hspace=0.45, wspace=0.15)

    if saveto:
        fig.savefig(saveto, dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
    else:
        plt.show()


