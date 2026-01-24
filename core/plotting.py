
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from core import simulation

def plot_simu_trial_batch_belief_5conditions(out_go, out_stop, show_cnt=5, saveto=None):
    # ==========================================
    # 🎨 TUNING SECTION
    # ==========================================
    style = {
        # Fonts
        "label_fontsize": 14,
        "title_fontsize": 16,
        "tick_fontsize": 12,
        
        # Mean Lines
        "mean_lw": 1.5,          
        "mean_alpha": 1.0,       
        "sem_alpha": 0.3,        

        # Individual Samples
        "sample_active_lw": 0.8,    
        "sample_active_alpha": 0.6, 

        # Decision Dot
        "marker_size": 40,       
        "marker_alpha": 1.0,     
        "marker_edge_width": 1.0,

        # SSD Vertical Lines
        "ssd_lw": 1.2,
        "ssd_alpha": 0.8,
        
        # Survival Curve Styles
        "survival_color": "gray",
        "survival_lw": 0.0,       # The line width of the curve itself
        "survival_alpha": 0.4,    # Opacity of the curve line
        "survival_fill_alpha": 0.2, # Opacity of the area under the curve
    }

    colors = {
        "beta": "#4E7C59",
        "sigma": "#B75D69",
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

    fig, ax = plt.subplots(2, 3, figsize=(15, 8)) 
    
    # Standardize X-axis
    MAX_TIME_STEPS = 40
    xticks = [1, 10, 20, 30, 40]

    for i, df in enumerate(df_list):
        row, col = divmod(i, 3)
        axis = ax[row, col]

        if len(df) == 0:
            axis.axis('off')
            continue

        # -----------------------------------------------------------
        # 1. PRE-PROCESS: Extract Truncated Sequences & RTs
        # -----------------------------------------------------------
        beta_trunc_list = []
        sigma_trunc_list = []
        rt_indices = []

        for _, row_data in df.iterrows():
            beta_seq = row_data['beta_seq']
            sigma_seq = row_data['sigma_seq']
            policy_seq = row_data['policy_seq']

            # Find decision time
            decision_idx = len(policy_seq) - 1 
            for t_idx, act in enumerate(policy_seq):
                if act != 2: # 2 is WAIT
                    decision_idx = t_idx
                    break
            
            rt_indices.append(decision_idx)
            limit = decision_idx + 1
            beta_trunc_list.append(beta_seq[:limit])
            sigma_trunc_list.append(sigma_seq[:limit])

        # -----------------------------------------------------------
        # 2. SETUP DUAL AXIS (Survival Curve)
        # -----------------------------------------------------------
        ax2 = axis.twinx()
        
        # Convert indices (0-based) to Time Step (1-based)
        valid_rts = np.sort([r + 1 for r in rt_indices])
        
        if len(valid_rts) > 0:
            n_samples = len(valid_rts)
            y_survival = 1.0 - np.arange(1, n_samples + 1) / n_samples
            
            # Prepend (0, 1.0)
            x_surv_plot = np.concatenate(([1], valid_rts))
            y_surv_plot = np.concatenate(([1.0], y_survival))

            # Plot Step Function (The Line)
            ax2.step(x_surv_plot, y_surv_plot, where='post',
                     color=style['survival_color'], lw=style['survival_lw'], 
                     alpha=style['survival_alpha'], label='Survival')
            
            # Plot Filled Area (The Shade) - linewidth=0 removes the "edge"
            ax2.fill_between(x_surv_plot, y_surv_plot, step='post', 
                             color=style['survival_color'], alpha=style['survival_fill_alpha'],
                             linewidth=0) 

        # Style the Right Axis
        ax2.set_ylabel('Survival Prob.', color='gray', fontsize=style['label_fontsize'])
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=style['tick_fontsize'])
        ax2.set_ylim(0, 1.05) 
        ax2.set_yticks([0, 0.5, 1.0])
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_color('gray')

        # -----------------------------------------------------------
        # 3. PLOT MEANS (Left Axis - Foreground)
        # -----------------------------------------------------------
        df_beta_trunc = pd.DataFrame(beta_trunc_list)
        df_sigma_trunc = pd.DataFrame(sigma_trunc_list)

        beta_mean = df_beta_trunc.mean(axis=0)
        beta_sem = df_beta_trunc.sem(axis=0)
        sigma_mean = df_sigma_trunc.mean(axis=0)
        sigma_sem = df_sigma_trunc.sem(axis=0)

        x_vals_mean = np.arange(1, len(beta_mean) + 1)

        axis.plot(x_vals_mean, beta_mean, label=r'Go Belief $\beta^\nu_t$',
                  color=colors["beta"], linewidth=style['mean_lw'], alpha=style['mean_alpha'])
        axis.fill_between(x_vals_mean, beta_mean - beta_sem, beta_mean + beta_sem,
                          color=colors["beta"], alpha=style['sem_alpha'])

        axis.plot(x_vals_mean, sigma_mean, label=r'Stop Trial Belief $\sigma^\nu_t$',
                  color=colors["sigma"], linewidth=style['mean_lw'], alpha=style['mean_alpha'])
        axis.fill_between(x_vals_mean, sigma_mean - sigma_sem, sigma_mean + sigma_sem,
                          color=colors["sigma"], alpha=style['sem_alpha'])

        # -----------------------------------------------------------
        # 4. PLOT SAMPLES (Left Axis - Foreground)
        # -----------------------------------------------------------
        subset_indices = list(range(min(show_cnt, len(df))))
        for j in subset_indices:
            d_idx = rt_indices[j]
            limit = d_idx + 1
            b_seq = df.iloc[j]['beta_seq']
            s_seq = df.iloc[j]['sigma_seq']
            x_sample = np.arange(1, limit + 1)
            
            is_last_step = (d_idx >= MAX_TIME_STEPS - 1)

            # Beta
            axis.plot(x_sample, b_seq[:limit],
                      color=colors["beta"], linewidth=style['sample_active_lw'], 
                      alpha=style['sample_active_alpha'])
            if not is_last_step:
                axis.scatter(x_sample[-1], b_seq[d_idx], 
                             s=style['marker_size'], color=colors["beta"], alpha=style['marker_alpha'], 
                             zorder=3, edgecolors='white', linewidth=style['marker_edge_width'])

            # Sigma
            axis.plot(x_sample, s_seq[:limit],
                      color=colors["sigma"], linewidth=style['sample_active_lw'], 
                      alpha=style['sample_active_alpha'])
            if not is_last_step:
                axis.scatter(x_sample[-1], s_seq[d_idx], 
                             s=style['marker_size'], color=colors["sigma"], alpha=style['marker_alpha'], 
                             zorder=3, edgecolors='white', linewidth=style['marker_edge_width'])

        # -----------------------------------------------------------
        # 5. FINAL FORMATTING
        # -----------------------------------------------------------
        if condition_names[i].startswith('Stop') and ssd is not None:
            axis.axvline(x=ssd + 1, color='k', linestyle='--', linewidth=style['ssd_lw'], alpha=style['ssd_alpha'])
            axis.axvline(x=ssd + 12 + 1, color='k', linestyle='--', linewidth=style['ssd_lw'], alpha=style['ssd_alpha'])

        axis.set_xlim(1, MAX_TIME_STEPS)
        axis.set_ylim(0, 1.1)
        axis.set_xticks(xticks)
        axis.set_title(f"{condition_names[i]}", pad=12, fontsize=style['title_fontsize'])
        axis.tick_params(axis='both', labelsize=style['tick_fontsize'])

        # Labels on ALL subplots as requested
        axis.set_xlabel("Time Step", fontsize=style['label_fontsize'])
        axis.set_ylabel(r"Belief State", fontsize=style['label_fontsize'], color='black')

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False) 
        
        # Z-Order Trick
        axis.set_zorder(ax2.get_zorder() + 1) 
        axis.patch.set_visible(False)

    ax[1, 2].axis("off")
    # Increased wspace slightly to allow labels on every plot to breathe
    plt.subplots_adjust(hspace=0.45, wspace=0.45, top=0.92, bottom=0.08, left=0.08, right=0.92)

    # Legend
    legend_handles = [
        Line2D([0], [0], color=colors["beta"], lw=style['mean_lw'], label=r'Go Belief $\beta^\nu_t$'),
        Line2D([0], [0], color=colors["sigma"], lw=style['mean_lw'], label=r'Stop Trial Belief $\sigma^\nu_t$'),
        # Use a Patch (rectangle) to represent the filled survival area
        Patch(facecolor=style['survival_color'], alpha=style['survival_fill_alpha'], label='Survival Probability')
    ]

    fig.legend(handles=legend_handles, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 1.05), 
               ncol=3, 
               frameon=False, 
               fontsize=style['label_fontsize']+1)
    if saveto:
        fig.savefig(saveto, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_simu_trial_batch_action_value_5conditions(out_go, out_stop, show_cnt=5, saveto=None):
    # ==========================================
    # 🎨 TUNING SECTION (Matched to Belief Plot)
    # ==========================================
    style = {
        # Fonts
        "label_fontsize": 14,
        "title_fontsize": 16,
        "tick_fontsize": 12,
        
        # Mean Lines
        "mean_lw": 1.5,          
        "mean_alpha": 1.0,       
        "sem_alpha": 0.3,        

        # Individual Samples
        "sample_active_lw": 0.8,    
        "sample_active_alpha": 0.6, 

        # Decision Dot
        "marker_size": 40,       
        "marker_alpha": 1.0,     
        "marker_edge_width": 1.0,

        # SSD Vertical Lines
        "ssd_lw": 1.2,
        "ssd_alpha": 0.8,
        
        # Survival Curve Styles
        "survival_color": "gray",
        "survival_lw": 0.0,       
        "survival_alpha": 0.4,    
        "survival_fill_alpha": 0.2, 
    }

    colors = {
        "Q_L": "#E3B23C", # Gold
        "Q_R": "#3A6EA5", # Blue
        "Q_W": "#7A7A7A", # Gray
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

    fig, ax = plt.subplots(2, 3, figsize=(15, 8)) 
    
    # Standardize X-axis
    MAX_TIME_STEPS = 40
    xticks = [1, 10, 20, 30, 40]

    for i, df in enumerate(df_list):
        row, col = divmod(i, 3)
        axis = ax[row, col]

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
            # Get full sequences
            ql_seq = row_data['value_left_seq']
            qr_seq = row_data['value_right_seq']
            qw_seq = row_data['value_wait_seq']
            policy_seq = row_data['policy_seq']

            # Find decision time
            decision_idx = len(policy_seq) - 1 
            for t_idx, act in enumerate(policy_seq):
                if act != 2: # 2 is WAIT
                    decision_idx = t_idx
                    break
            
            rt_indices.append(decision_idx)
            limit = decision_idx + 1
            
            # Truncate to decision point for consistent mean calculation
            q_l_trunc_list.append(ql_seq[:limit])
            q_r_trunc_list.append(qr_seq[:limit])
            q_w_trunc_list.append(qw_seq[:limit])

        # -----------------------------------------------------------
        # 2. SETUP DUAL AXIS (Survival Curve)
        # -----------------------------------------------------------
        ax2 = axis.twinx()
        
        valid_rts = np.sort([r + 1 for r in rt_indices])
        
        if len(valid_rts) > 0:
            n_samples = len(valid_rts)
            y_survival = 1.0 - np.arange(1, n_samples + 1) / n_samples
            
            x_surv_plot = np.concatenate(([1], valid_rts))
            y_surv_plot = np.concatenate(([1.0], y_survival))

            ax2.step(x_surv_plot, y_surv_plot, where='post',
                     color=style['survival_color'], lw=style['survival_lw'], 
                     alpha=style['survival_alpha'])
            
            ax2.fill_between(x_surv_plot, y_surv_plot, step='post', 
                             color=style['survival_color'], alpha=style['survival_fill_alpha'],
                             linewidth=0) 

        # Style the Right Axis
        ax2.set_ylabel('Survival Prob.', color='gray', fontsize=style['label_fontsize'])
        ax2.tick_params(axis='y', labelcolor='gray', labelsize=style['tick_fontsize'])
        ax2.set_ylim(0, 1.05) 
        ax2.set_yticks([0, 0.5, 1.0])
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_color('gray')

        # -----------------------------------------------------------
        # 3. PLOT MEANS (Left Axis - Foreground)
        # -----------------------------------------------------------
        # Convert lists to DataFrames to handle variable lengths (pads with NaN)
        df_ql = pd.DataFrame(q_l_trunc_list)
        df_qr = pd.DataFrame(q_r_trunc_list)
        df_qw = pd.DataFrame(q_w_trunc_list)

        # Plot Helper
        def plot_mean_with_sem(data_df, color, label=None):
            mean = data_df.mean(axis=0)
            sem = data_df.sem(axis=0)
            x_vals = np.arange(1, len(mean) + 1)
            axis.plot(x_vals, mean, label=label,
                      color=color, linewidth=style['mean_lw'], alpha=style['mean_alpha'])
            axis.fill_between(x_vals, mean - sem, mean + sem,
                              color=color, alpha=style['sem_alpha'])

        plot_mean_with_sem(df_qw, colors["Q_W"], r'$Q^\nu_t(\mathbf{b}_t^\nu, W)$')
        plot_mean_with_sem(df_ql, colors["Q_L"], r'$Q^\nu_t(\mathbf{b}_t^\nu, L)$')
        plot_mean_with_sem(df_qr, colors["Q_R"], r'$Q^\nu_t(\mathbf{b}_t^\nu, R)$')

        # -----------------------------------------------------------
        # 4. PLOT SAMPLES (Left Axis - Foreground)
        # -----------------------------------------------------------
        subset_indices = list(range(min(show_cnt, len(df))))
        for j in subset_indices:
            d_idx = rt_indices[j]
            limit = d_idx + 1
            
            # Retrieve raw sequences
            ql_seq = df.iloc[j]['value_left_seq']
            qr_seq = df.iloc[j]['value_right_seq']
            qw_seq = df.iloc[j]['value_wait_seq']
            
            x_sample = np.arange(1, limit + 1)
            is_last_step = (d_idx >= MAX_TIME_STEPS - 1)

            # Helper for lines and dots
            def plot_sample(seq, color):
                # Line
                axis.plot(x_sample, seq[:limit],
                          color=color, linewidth=style['sample_active_lw'], 
                          alpha=style['sample_active_alpha'])
                # Dot (if not max time)
                if not is_last_step:
                    axis.scatter(x_sample[-1], seq[d_idx], 
                                 s=style['marker_size'], color=color, alpha=style['marker_alpha'], 
                                 zorder=3, edgecolors='white', linewidth=style['marker_edge_width'])

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
        # Auto-scale Y or fix it? Belief is 0-1. Q-values can be larger.
        # Let's let matplotlib auto-scale the Y axis but ensure 0 is included if possible.
        # axis.set_ylim(bottom=0) 
        
        axis.set_xticks(xticks)
        axis.set_title(f"{condition_names[i]}", pad=12, fontsize=style['title_fontsize'])
        axis.tick_params(axis='both', labelsize=style['tick_fontsize'])

        axis.set_xlabel("Time Step", fontsize=style['label_fontsize'])
        axis.set_ylabel("Action Value", fontsize=style['label_fontsize'], color='black')

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False) 
        
        # Z-Order Trick
        axis.set_zorder(ax2.get_zorder() + 1) 
        axis.patch.set_visible(False)

    ax[1, 2].axis("off")
    plt.subplots_adjust(hspace=0.45, wspace=0.45, top=0.88, bottom=0.08, left=0.08, right=0.92)

    # Legend
    legend_handles = [
        Line2D([0], [0], color=colors["Q_W"], lw=style['mean_lw'], label=r'Wait Action Value $Q^\nu_t(\mathbf{b}_t^\nu, W)$'),
        Line2D([0], [0], color=colors["Q_L"], lw=style['mean_lw'], label=r'Go-Left Action Value $Q^\nu_t(\mathbf{b}_t^\nu, L)$'),
        Line2D([0], [0], color=colors["Q_R"], lw=style['mean_lw'], label=r'Go-Right Action Value $Q^\nu_t(\mathbf{b}_t^\nu, R)$'),
        Patch(facecolor=style['survival_color'], alpha=style['survival_fill_alpha'], label='Survival Probability')
    ]

    fig.legend(handles=legend_handles, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 1.05), 
               ncol=2, 
               frameon=False, 
               fontsize=style['label_fontsize']+1)
    
    if saveto:
        fig.savefig(saveto, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_policy(model, saveto, dpi=150):
    """
    Publication-quality plot of the optimal policy over all time steps.
    """
    colors = {
        "beta": "#4E7C59",
        "sigma": "#B75D69",
        "Q_L": "#E3B23C",
        "Q_R": "#3A6EA5",
        "Q_W": "#7A7A7A",
    }

    policy = model.policy[:-1, :, :].copy()
    num_timesteps = policy.shape[0]

    # # Mask undefined beliefs
    # beta = 0.5
    # zeta = 0
    # for i in range(num_timesteps):
    #     beta = model.update_beta(beta, zeta=0, x=1)
    #     beta_idx = int(beta * model.bins)
    #     beta_idx = np.clip(beta_idx, 0, model.bins - 1)
    #     policy[i, beta_idx + 2:, :] = np.nan
    #     policy[i, :(model.bins - beta_idx) - 1, :] = np.nan

    #     zeta = model.update_zeta(zeta, 1, i)
    #     sigma = model.update_sigma(zeta, i)
    #     sigma_idx = int(sigma * model.bins)
    #     sigma_idx = np.clip(sigma_idx, 0, model.bins - 1)
    #     policy[i, :, sigma_idx + 2:] = np.nan


    # Define color mapping
    cmap = ListedColormap(
        [colors['Q_L'], colors['Q_R'], "lightgray", "white"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    # Layout grid
    cols = min(6, num_timesteps)
    rows = (num_timesteps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5),
                                dpi=dpi, constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for i in range(num_timesteps):
        # matrix_plot = np.where(np.isnan(policy[i]), 3, policy[i])
        matrix_plot = model.policy[i].copy()
        matrix_plot[np.isnan(matrix_plot)] = 3

        ax = axes[i]
        img = ax.imshow(matrix_plot, cmap=cmap,
                        norm=norm, extent=[0, 1, 0, 1])
        ax.set_title(f"t = {i + 1}", fontsize=14)
        ax.set_xlabel(r'$\sigma^\nu_t$', fontsize=14)
        ax.set_ylabel(r'$\beta^\nu_t$', fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis='both', which='both',
                        direction='out', length=4, width=0.8)

    for i in range(num_timesteps, len(axes)):
        axes[i].axis("off")

    # Shared colorbar
    cbar = fig.colorbar(
        img, ax=axes, orientation='horizontal', shrink=1, aspect=50, pad=0.02)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(["Left", "Right", "Wait", "None"])
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.set_title("Action", fontsize=14, pad=10,
                        rotation=0, loc='left')

    # Optional: title
    # fig.suptitle("Optimal Policy Over Time", fontsize=16, y=1.03)

    plt.savefig(saveto, dpi=dpi, bbox_inches="tight")
    plt.show()

def plot_policy_simple(model, saveto, dpi=150, selected_timesteps=None):
    """
    Simplified publication-quality policy plot.

    Parameters:
        selected_timesteps: list of int, time step indices to plot.
    """
    colors = {
        "beta": "#4E7C59",
        "sigma": "#B75D69",
        "Q_L": "#E3B23C",
        "Q_R": "#3A6EA5",
        "Q_W": "#7A7A7A",
    }

    policy = model.policy[:-1, :, :].copy()
    num_timesteps = policy.shape[0]

    if selected_timesteps is None:
        selected_timesteps = list(range(min(4, num_timesteps)))
    else:
        selected_timesteps = [
            t for t in selected_timesteps if 0 <= t < num_timesteps]

    # mask for policy according to belief constraints
    beta = 0.5
    zeta = 0
    for i in range(num_timesteps):
        beta = model.update_beta(beta, zeta=0, x=1)
        beta_idx = int(beta * model.bins)
        beta_idx = np.clip(beta_idx, 0, model.bins - 1)
        policy[i, beta_idx + 2:, :] = np.nan
        policy[i, :(model.bins - beta_idx) - 1, :] = np.nan

        zeta = model.update_zeta(zeta, 1, i)
        sigma = model.update_sigma(zeta, i)
        sigma_idx = int(sigma * model.bins)
        sigma_idx = np.clip(sigma_idx, 0, model.bins - 1)
        policy[i, :, sigma_idx + 2:] = np.nan

    # Set colormap
    cmap = ListedColormap(
        [colors['Q_R'], colors['Q_L'], "lightgray", "white"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    cols = min(4, len(selected_timesteps))
    rows = (len(selected_timesteps) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5),
                                dpi=dpi, constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for i, t in enumerate(selected_timesteps):
        matrix_plot = np.where(np.isnan(policy[t]), 3, policy[t])
        ax = axes[i]
        img = ax.imshow(matrix_plot, cmap=cmap,
                        norm=norm, extent=[0, 1, 0, 1])

        ax.set_title(f"t = {t + 1}", fontsize=14)
        ax.set_xlabel(r'$\sigma^\nu_t$', fontsize=14)
        ax.set_ylabel(r'$\beta^\nu_t$', fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis='both', which='both',
                        direction='out', length=4, width=0.8)

    for i in range(len(selected_timesteps), len(axes)):
        axes[i].axis("off")

    cbar = fig.colorbar(
        img, ax=axes, orientation='vertical', shrink=0.8, pad=0.02)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(["Right", "Left", "Wait", "None"])
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.set_title("Action", fontsize=14, pad=10,
                        rotation=0, loc='left')
    plt.savefig(saveto, dpi=dpi, bbox_inches="tight")
    plt.show()

###################### Viz Across-trial ######################

def _simulate_mixed_session(model, n_trials=2000, ssd=12):
    import numpy as np
    import pandas as pd
    out = []
    for _ in range(n_trials):
        is_stop = np.random.rand() < model.rate_stop_trial
        gs = np.random.choice(["left", "right"])
        if is_stop:
            r = simulation.simu_trial(model, gs, "stop", ssd, verbose=False)
        else:
            r = simulation.simu_trial(model, gs, "nonstop", None, verbose=False)
        out.append(r)
    return pd.DataFrame(out, columns=["result", "rt", "ssd"])

def _summarize_behavior_with_sem(df, step_in_ms=25.0):
    import numpy as np
    is_go = df["result"].isin(["GS", "GE", "GM"])
    go_df = df.loc[is_go]
    n_go = len(go_df)
    n_gs = (go_df["result"] == "GS").sum()
    go_acc = (n_gs / n_go) if n_go > 0 else np.nan
    go_acc_sem = np.sqrt(go_acc * (1.0 - go_acc) /
                            n_go) if n_go > 0 else np.nan

    is_stop = df["result"].isin(["SS", "SE"])
    stop_df = df.loc[is_stop]
    n_stop = len(stop_df)
    n_ss = (stop_df["result"] == "SS").sum()
    stop_succ = (n_ss / n_stop) if n_stop > 0 else np.nan
    stop_succ_sem = np.sqrt(
        stop_succ * (1.0 - stop_succ) / n_stop) if n_stop > 0 else np.nan

    rt_gs = df.loc[df["result"] == "GS", "rt"].to_numpy() * step_in_ms
    rt_se = df.loc[df["result"] == "SE", "rt"].to_numpy() * step_in_ms
    rt_go_succ = np.nanmean(rt_gs) if rt_gs.size > 0 else np.nan
    rt_go_succ_sem = (np.nanstd(rt_gs, ddof=1) /
                        np.sqrt(rt_gs.size)) if rt_gs.size > 1 else np.nan
    rt_stop_err = np.nanmean(rt_se) if rt_se.size > 0 else np.nan
    rt_stop_err_sem = (np.nanstd(rt_se, ddof=1) /
                        np.sqrt(rt_se.size)) if rt_se.size > 1 else np.nan

    return dict(
        go_acc=go_acc, go_acc_sem=go_acc_sem,
        stop_succ=stop_succ, stop_succ_sem=stop_succ_sem,
        rt_go_succ=rt_go_succ, rt_go_succ_sem=rt_go_succ_sem,
        rt_stop_err=rt_stop_err, rt_stop_err_sem=rt_stop_err_sem
    )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from core import simulation

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
    Revised 2x3 Composite Figure.
    - Uses Truncated Means (Active Trials Only).
    - Adds visible Twin Axis (ax2) for Survival Probability on B, C, E, F.
    - Uses alpha accumulation for shadow 'ink' effect.
    """
    
    # ==========================================
    # 🎨 STYLE CONFIG
    # ==========================================
    style = {
        "label_fontsize": 18,
        "tick_fontsize": 16,
        "panel_label_size": 28, 
        "line_width": 2.5,
        "marker_size": 10,
        "sem_alpha": 0.2,
        "cap_size": 4,
        "legend_fontsize": 16,
        
        # Shadow Settings
        "shadow_alpha": 0.1,    # Accumulates to create darker overlaps
        "shadow_color": "gray"
    }

    color_acc = 'black'
    color_rt  = '#08519c'
    
    cmap_warm = LinearSegmentedColormap.from_list("journal_warm", ["#C46254", "#49000a"])
    rate_colors = [cmap_warm(i) for i in np.linspace(0.0, 1.0, len(rates))]

    # ==========================================
    # 1. HELPER: TRUNCATION LOGIC
    # ==========================================
    def _truncate_and_collect(seq_list):
        if not seq_list: 
            return np.full(t_max, np.nan), np.full(t_max, np.nan)
        
        df = pd.DataFrame(seq_list)
        df = df.reindex(columns=range(t_max))
        
        with np.errstate(invalid='ignore'):
            mu = df.mean(axis=0).to_numpy()
            sd = df.std(axis=0, ddof=1).to_numpy()
            count = df.count(axis=0).to_numpy()
            se = np.where(count > 1, sd / np.sqrt(count), np.nan)
        
        return mu, se

    # ==========================================
    # 2. RUN SIMULATIONS
    # ==========================================
    beh_rows = []
    timecourse_go = []
    timecourse_stop = []

    print("Simulating...")
    for r_idx, r in enumerate(rates):
        p = params.copy()
        p['rate_stop_trial'] = float(r)
        m = model.__class__(**p)
        m.value_iteration_tensor()

        # --- A. Behavior Session ---
        out_sess = []
        for _ in range(n_session):
            is_stop = np.random.rand() < float(r)
            gs = np.random.choice(["left", "right"])
            type_ = "stop" if is_stop else "nonstop"
            ssd_ = ssd if is_stop else None
            res = simulation.simu_trial(m, gs, type_, ssd_, verbose=False) 
            out_sess.append(res)
        
        df_beh = pd.DataFrame(out_sess, columns=["result", "rt", "ssd"])
        
        # Stats for A & D
        go_df = df_beh[df_beh["result"].isin(["GS", "GE", "GM"])]
        if len(go_df) > 0:
            go_acc = (go_df["result"]=="GS").mean()
            go_acc_sem = np.sqrt(go_acc*(1-go_acc)/len(go_df))
            gs_df = go_df[go_df["result"]=="GS"]
            go_rt = gs_df["rt"].mean()*25 if len(gs_df)>0 else np.nan
            go_rt_sem = gs_df["rt"].std(ddof=1)*25/np.sqrt(len(gs_df)) if len(gs_df)>0 else np.nan
        else:
            go_acc, go_acc_sem, go_rt, go_rt_sem = [np.nan]*4

        stop_df = df_beh[df_beh["result"].isin(["SS", "SE"])]
        if len(stop_df) > 0:
            stop_acc = (stop_df["result"]=="SS").mean()
            stop_acc_sem = np.sqrt(stop_acc*(1-stop_acc)/len(stop_df))
            se_df = stop_df[stop_df["result"]=="SE"]
            stop_rt = se_df["rt"].mean()*25 if len(se_df)>0 else np.nan 
            stop_rt_sem = se_df["rt"].std(ddof=1)*25/np.sqrt(len(se_df)) if len(se_df)>0 else np.nan
        else:
            stop_acc, stop_acc_sem, stop_rt, stop_rt_sem = [np.nan]*4

        beh_rows.append({
            "rate": r,
            "go_acc": go_acc, "go_acc_sem": go_acc_sem, 
            "go_rt": go_rt, "go_rt_sem": go_rt_sem,
            "stop_acc": stop_acc, "stop_acc_sem": stop_acc_sem, 
            "stop_rt": stop_rt, "stop_rt_sem": stop_rt_sem
        })

        # --- B. Timecourse Batches ---
        g_data = {"GS": {"sig": [], "gap": [], "rt": []}, 
                  "GE": {"sig": [], "gap": [], "rt": []}, 
                  "GM": {"sig": [], "gap": [], "rt": []}}
        
        s_data = {"SS": {"sig": [], "gap": [], "rt": []}, 
                  "SE": {"sig": [], "gap": [], "rt": []}}

        def process_trial_result(res, store_dict):
            out_type = res["result"]
            if out_type not in store_dict: return

            policy_seq = res["policy_seq"]
            decision_idx = len(policy_seq) - 1
            for t_idx, act in enumerate(policy_seq):
                if act != 2: # 2 is WAIT
                    decision_idx = t_idx
                    break
            
            limit = decision_idx + 1
            
            # Truncate sequences
            sigma_seq = np.array(res["sigma_seq"])[:limit]
            
            vw = np.array(res["value_wait_seq"])[:limit]
            vl = np.array(res["value_left_seq"])[:limit]
            vr = np.array(res["value_right_seq"])[:limit]
            
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
            
            # Go Trial
            res_g = simulation.simu_trial(m, true_go, "nonstop", None, verbose=True)
            process_trial_result(res_g, g_data)

            # Stop Trial
            res_s = simulation.simu_trial(m, true_go, "stop", ssd, verbose=True)
            process_trial_result(res_s, s_data)

        # Aggregate
        pkt_g = {"rate": r, "color": rate_colors[r_idx], "rts": {k: v["rt"] for k,v in g_data.items()}}
        for cat in ["GS", "GE", "GM"]:
            pkt_g[f"sigma_{cat}_mu"], pkt_g[f"sigma_{cat}_sem"] = _truncate_and_collect(g_data[cat]["sig"])
            pkt_g[f"gap_{cat}_mu"],   pkt_g[f"gap_{cat}_sem"]   = _truncate_and_collect(g_data[cat]["gap"])
        timecourse_go.append(pkt_g)

        pkt_s = {"rate": r, "color": rate_colors[r_idx], "rts": {k: v["rt"] for k,v in s_data.items()}}
        for cat in ["SS", "SE"]:
            pkt_s[f"sigma_{cat}_mu"], pkt_s[f"sigma_{cat}_sem"] = _truncate_and_collect(s_data[cat]["sig"])
            pkt_s[f"gap_{cat}_mu"],   pkt_s[f"gap_{cat}_sem"]   = _truncate_and_collect(s_data[cat]["gap"])
        timecourse_stop.append(pkt_s)

    df_beh_summary = pd.DataFrame(beh_rows)

    # ==========================================
    # 3. PLOTTING
    # =========================================    
    # Use a wide figure to accommodate the dual axes and labels
    fig = plt.figure(figsize=(25, 12.5)) 
    
    # Unified 2x3 Grid
    # wspace=0.5 ensures enough room for the dual y-axes between all columns
    gs = fig.add_gridspec(2, 3, wspace=0.4, hspace=0.4)

    ax_go_beh   = fig.add_subplot(gs[0, 0])
    ax_go_bel   = fig.add_subplot(gs[0, 1])
    ax_go_gap   = fig.add_subplot(gs[0, 2])
    
    ax_stop_beh = fig.add_subplot(gs[1, 0])
    ax_stop_bel = fig.add_subplot(gs[1, 1])
    ax_stop_gap = fig.add_subplot(gs[1, 2])

    axes_time = [ax_go_bel, ax_go_gap, ax_stop_bel, ax_stop_gap]
    x_time = np.arange(1, t_max+1)
    # -----------------------------
    # 3.1 SETUP SHADOW AXES (TwinX)
    # -----------------------------
    shadow_axes = {}
    
    for ax in axes_time:
        ax_shad = ax.twinx()
        
        # Style matches plot_simu_trial_batch_belief_5conditions
        ax_shad.set_ylabel('Survival Prob.', color='gray', fontsize=style['label_fontsize'])
        ax_shad.tick_params(axis='y', labelcolor='gray', labelsize=style['tick_fontsize'])
        
        ax_shad.set_ylim(0, 1.05)
        ax_shad.set_yticks([0, 0.5, 1.0])
        
        ax_shad.spines['top'].set_visible(False)
        ax_shad.spines['left'].set_visible(False)
        ax_shad.spines['right'].set_color('gray')
        ax_shad.spines['right'].set_visible(True)
        
        # Move shadow to background
        ax_shad.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_visible(False) 
        
        shadow_axes[ax] = ax_shad

    # -----------------------------
    # 3.2 PLOT SHADOWS
    # -----------------------------
    def draw_shadow(target_ax, rt_list):
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
                               color=style['shadow_color'], 
                               alpha=style['shadow_alpha'], 
                               linewidth=0)

    # Use Middle Rate for Reference Shadows
    mid_idx = len(rates) // 2
    rts_go = timecourse_go[mid_idx]["rts"]
    rts_stop = timecourse_stop[mid_idx]["rts"]
    
    for cat in ["GM", "GS", "GE"]:
        draw_shadow(shadow_axes[ax_go_bel], rts_go.get(cat, []))
        draw_shadow(shadow_axes[ax_go_gap], rts_go.get(cat, []))
    
    for cat in ["SS", "SE"]:
        draw_shadow(shadow_axes[ax_stop_bel], rts_stop.get(cat, []))
        draw_shadow(shadow_axes[ax_stop_gap], rts_stop.get(cat, []))

    # -----------------------------
    # 3.3 PLOT TRUNCATED CURVES (Main Axis)
    # -----------------------------
    def plot_tc(ax, dataset, metric_prefix, outcome_keys, styles_map):
        for d in dataset:
            col = d["color"]
            for out_key in outcome_keys:
                mu = d[f"{metric_prefix}_{out_key}_mu"]
                se = d[f"{metric_prefix}_{out_key}_sem"]
                ls = styles_map[out_key]["ls"]
                
                mask = ~np.isnan(mu)
                if np.any(mask):
                    ax.plot(x_time[mask], mu[mask], color=col, ls=ls, lw=2)
                    ax.fill_between(x_time[mask], mu[mask]-se[mask], mu[mask]+se[mask], 
                                    color=col, alpha=0.2, lw=0)

    st_map = {
        "GS": {"ls": "-"}, "GE": {"ls": "--"}, "GM": {"ls": ":"},
        "SS": {"ls": "-"}, "SE": {"ls": "--"}
    }

    plot_tc(ax_go_bel, timecourse_go, "sigma", ["GS", "GE", "GM"], st_map)
    plot_tc(ax_go_gap, timecourse_go, "gap",   ["GS", "GE", "GM"], st_map)
    plot_tc(ax_stop_bel, timecourse_stop, "sigma", ["SS", "SE"], st_map)
    plot_tc(ax_stop_gap, timecourse_stop, "gap",   ["SS", "SE"], st_map)

    # -----------------------------
    # 3.4 BEHAVIOR PANELS (A & D)
    # -----------------------------
    def plot_behavior_panel(ax, df, acc_key, acc_sem_key, rt_key, rt_sem_key, rt_label, acc_label):
            l1 = ax.errorbar(df["rate"], df[acc_key], yerr=df[acc_sem_key],
                        color=color_acc, linestyle='-', marker='o', 
                        markersize=style['marker_size'], lw=style['line_width'], 
                        capsize=style['cap_size'], label=acc_label)
            
            ax.set_ylabel("Accuracy", color=color_acc, fontsize=style['label_fontsize'])
            ax.tick_params(axis='y', labelcolor=color_acc)
            ax.set_ylim(0, 1.05)

            axR = ax.twinx()
            l2 = axR.errorbar(df["rate"], df[rt_key], yerr=df[rt_sem_key],
                        color=color_rt, linestyle='-', marker='s', 
                        markersize=style['marker_size'], lw=style['line_width'], 
                        capsize=style['cap_size'], label=rt_label)
            
            axR.set_ylabel("RT (ms)", color=color_rt, fontsize=style['label_fontsize'])
            axR.tick_params(axis='y', labelcolor=color_rt)
            
            ax.legend([l1, l2], [acc_label, rt_label], loc='lower right', 
                    frameon=False, fontsize=style['tick_fontsize'])
            return axR

    axR_go = plot_behavior_panel(ax_go_beh, df_beh_summary, 
                                 "go_acc", "go_acc_sem", 
                                 "go_rt", "go_rt_sem", "Go Success RT", "Go Accuracy")

    axR_stop = plot_behavior_panel(ax_stop_beh, df_beh_summary, 
                                   "stop_acc", "stop_acc_sem", 
                                   "stop_rt", "stop_rt_sem", "Stop Error RT", "Stop Accuracy")

    # -----------------------------
    # FORMATTING & LEGENDS
    # -----------------------------
    for ax in [ax_go_beh, ax_stop_beh]:
        ax.set_xlabel(r"Stop Prior $\tilde{r}^\nu$", fontsize=style['label_fontsize'])
        ax.spines['top'].set_visible(False)
        axR = ax.get_shared_x_axes().get_siblings(ax)[0] 
        axR.spines['top'].set_visible(False)
        ax.tick_params(axis='both', labelsize=style['tick_fontsize'])
        axR.tick_params(axis='both', labelsize=style['tick_fontsize'])

    for ax in axes_time:
        ax.set_xlabel("Time Step", fontsize=style['label_fontsize'])
        ax.set_xticks([1, 10, 20, 30, 40])
        ax.set_xlim(1, 40)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=style['tick_fontsize'])
    
    ax_go_bel.set_ylabel(r"Belief State $\sigma^\nu_t$", fontsize=style['label_fontsize'])
    ax_stop_bel.set_ylabel(r"Belief State $\sigma^\nu_t$", fontsize=style['label_fontsize'])
    
    label_gap = r"$Q_t^{\nu}(W) - \min(Q_t^{\nu}(L), Q_t^{\nu}(R))$"
    ax_go_gap.set_ylabel(label_gap, fontsize=style['label_fontsize'])
    ax_stop_gap.set_ylabel(label_gap, fontsize=style['label_fontsize'])
    
    if ssd:
        for ax in [ax_stop_bel, ax_stop_gap]:
            ax.axvline(ssd+1, color='k', ls='--', alpha=0.8)
            ax.axvline(ssd+12+1, color='k', ls='--', alpha=0.8)

    # LEGENDS
    rate_handles = [Line2D([0],[0], color=c, lw=3, label=f"{r:.2f}") for r,c in zip(rates, rate_colors)]
    out_handles_go  = [
        Line2D([0],[0], color='k', ls='-', label='Go Success'),
        Line2D([0],[0], color='k', ls='--', label='Go Error'),
        Line2D([0],[0], color='k', ls=':', label='Go Missing')
    ]
    out_handles_stop  = [
        Line2D([0],[0], color='k', ls='-', label='Stop Success'),
        Line2D([0],[0], color='k', ls='--', label='Stop Error')
    ]
    
    # Shadow Handle for Legend
    shadow_handle = Patch(facecolor=style['shadow_color'], alpha=style['shadow_alpha']*3, label='Survival Prob.')
    
    l1 = ax_go_gap.legend(handles=rate_handles, title=r"Stop Prior $\tilde{r}^\nu$", 
                     bbox_to_anchor=(1.18, 1), loc='upper left', frameon=False, 
                     fontsize=style['legend_fontsize'], title_fontsize=style['legend_fontsize']+2)
    l1._legend_box.align = "left"
    ax_go_gap.add_artist(l1)
    
    l2 = ax_go_gap.legend(handles=out_handles_go + [shadow_handle], title="Outcome", 
                     bbox_to_anchor=(1.18, 0.45), loc='upper left', frameon=False,
                       fontsize=style['legend_fontsize'], title_fontsize=style['legend_fontsize']+2)
    l2._legend_box.align = "left"

    l3 = ax_stop_gap.legend(handles=rate_handles, title=r"Stop Prior $\tilde{r}^\nu$", 
                       bbox_to_anchor=(1.18, 1), loc='upper left', frameon=False, 
                       fontsize=style['legend_fontsize'], title_fontsize=style['legend_fontsize']+2)
    l3._legend_box.align = "left"
    ax_stop_gap.add_artist(l3)
    
    l4 = ax_stop_gap.legend(handles=out_handles_stop + [shadow_handle], title="Outcome", 
                       bbox_to_anchor=(1.18, 0.45), loc='upper left', frameon=False,
                       fontsize=style['legend_fontsize'], title_fontsize=style['legend_fontsize']+2)
    l4._legend_box.align = "left"

    for ax, label in zip([ax_go_beh, ax_go_bel, ax_go_gap, ax_stop_beh, ax_stop_bel, ax_stop_gap], 
                         ["A", "B", "C", "D", "E", "F"]):
        ax.text(-0.15, 1.1, label, transform=ax.transAxes, 
                fontsize=style['panel_label_size'], fontweight='bold', va='top', ha='right')

    if saveto:
        fig.savefig(saveto, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()