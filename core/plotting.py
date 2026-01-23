
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
        "mean_lw": 2.0,          
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
        "mean_lw": 2.0,          
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

# def plot_rate_timecourses(
#     model,
#     params: dict,
#     rates=(0.05, 1/6, 0.35, 0.55, 0.80),
#     ssd=None,
#     n_batch=300,
#     t_max=40,
#     saveto=None,
#     trial_type: str = "stop",   # "stop" or "nonstop"
# ):
#     """
#     Across-rate within-trial dynamics.

#     If trial_type == "stop":
#         - Simulate stop trials with the given SSD (if None, default to 10).
#         - Left: mean sigma_t (Stop belief), colored by stop-rate (darker = higher rate).
#         - Right: gap Q(W) - min(Q(L), Q(R)), gray gradient, outcome styles SS(solid)/SE(dashed).
#         - SSD markers (vertical) at ssd+1 and ssd+12+1 if ssd is not None.

#     If trial_type == "nonstop":
#         - Simulate go-only (nonstop) trials; SSD is ignored.
#         - Left/Right panels as above; outcome styles GS(solid)/GE(dashed)/GM(dot-dash).
#         - No SSD markers.

#     Note:
#         Color encodes the across-trial stop probability r~ even for nonstop,
#         because it still shapes priors and value/belief dynamics.
#     """
#     import numpy as np
#     from matplotlib.colors import to_rgb, rgb_to_hsv, hsv_to_rgb

#     def desat_ramp(base="#7A1F2C", n=5, s_scale=0.55, v_low=0.42, v_high=0.88):
#         b = np.array(to_rgb(base))
#         h, s, v = rgb_to_hsv(b[np.newaxis, np.newaxis, :])[0, 0]
#         s = np.clip(s * s_scale, 0, 1)
#         vals = np.linspace(v_low, v_high, n)
#         return [tuple(hsv_to_rgb([h, s, vv])) for vv in vals]

#     def gray_ramp(n=5, g_low=0.18, g_high=0.70):
#         return [(g, g, g) for g in np.linspace(g_high, g_low, n)]

#     trial_type = str(trial_type).lower().strip()
#     if trial_type not in ("stop", "nonstop"):
#         raise ValueError("trial_type must be 'stop' or 'nonstop'.")

#     # ---------- helpers ----------
#     def _pad_to_matrix(seq_list, T):
#         if len(seq_list) == 0:
#             return np.full((0, T), np.nan)
#         out = np.full((len(seq_list), T), np.nan)
#         for i, s in enumerate(seq_list):
#             if s is None:
#                 continue
#             arr = np.asarray(s, dtype=float)
#             out[i, :min(len(arr), T)] = arr[:min(len(arr), T)]
#         return out

#     def _mean_sem(mat):
#         mean = np.nanmean(mat, axis=0)
#         n = np.sum(~np.isnan(mat), axis=0)
#         std = np.nanstd(mat, axis=0, ddof=1)
#         sem = np.where(n > 1, std / np.sqrt(n), np.nan)  # require n>1
#         return mean, sem

#     def _sigma_mean_sem(df):
#         if len(df) == 0 or "sigma_seq" not in df.columns:
#             return np.full(t_max, np.nan), np.full(t_max, np.nan)
#         mat = _pad_to_matrix(df["sigma_seq"].tolist(), t_max)
#         return _mean_sem(mat)

#     def _gap_mean_sem(df):
#         if len(df) == 0:
#             return np.full(t_max, np.nan), np.full(t_max, np.nan)
#         vl = _pad_to_matrix(df.get("value_left_seq", []).tolist(), t_max)
#         vr = _pad_to_matrix(df.get("value_right_seq", []).tolist(), t_max)
#         vw = _pad_to_matrix(df.get("value_wait_seq", []).tolist(), t_max)
#         if vl.shape[0] == 0 or vr.shape[0] == 0 or vw.shape[0] == 0:
#             return np.full(t_max, np.nan), np.full(t_max, np.nan)
#         min_go = np.nanmin(np.stack([vl, vr], axis=0), axis=0)
#         gap = vw - min_go
#         return _mean_sem(gap)

#     # ---------- simulate per rate ----------
#     series = []
#     for r in rates:
#         current_params = params.copy()
#         current_params['rate_stop_trial'] = float(r)

#         m = model.__class__(**current_params)
#         m.value_iteration_tensor()

#         # A=success, B=error, M=miss (GM only)
#         rec_a, rec_b, rec_m = [], [], []
#         n_iter = int(n_batch)

#         for _ in range(n_iter):
#             true_go = np.random.choice(["left", "right"])

#             if trial_type == "stop":
#                 ssd_use = 10 if ssd is None else int(ssd)
#                 res = simulation.simu_trial(m, true_go, "stop", ssd_use, verbose=True)
#                 outcome = res["result"] if isinstance(res, dict) else (
#                     res[0] if isinstance(res, (list, tuple)) else None)
#                 if outcome == "SS":
#                     rec_a.append(res)
#                 elif outcome == "SE":
#                     rec_b.append(res)
#                 # no "SM" in stop trials; keep two buckets
#             else:
#                 res = simulation.simu_trial(m, true_go, "nonstop", None, verbose=True)
#                 outcome = res["result"] if isinstance(res, dict) else (
#                     res[0] if isinstance(res, (list, tuple)) else None)
#                 if outcome == "GS":
#                     rec_a.append(res)
#                 elif outcome == "GE":
#                     rec_b.append(res)
#                 elif outcome == "GM":
#                     rec_m.append(res)

#         df_a, df_b, df_m = pd.DataFrame(
#             rec_a), pd.DataFrame(rec_b), pd.DataFrame(rec_m)

#         sigma_a_mean, sigma_a_sem = _sigma_mean_sem(df_a)
#         sigma_b_mean, sigma_b_sem = _sigma_mean_sem(df_b)
#         sigma_m_mean, sigma_m_sem = _sigma_mean_sem(df_m)

#         gap_a_mean, gap_a_sem = _gap_mean_sem(df_a)
#         gap_b_mean, gap_b_sem = _gap_mean_sem(df_b)
#         gap_m_mean, gap_m_sem = _gap_mean_sem(df_m)

#         series.append(dict(
#             rate=float(r),
#             sigma_a_mean=sigma_a_mean, sigma_a_sem=sigma_a_sem,
#             sigma_b_mean=sigma_b_mean, sigma_b_sem=sigma_b_sem,
#             sigma_m_mean=sigma_m_mean, sigma_m_sem=sigma_m_sem,
#             gap_a_mean=gap_a_mean,   gap_a_sem=gap_a_sem,
#             gap_b_mean=gap_b_mean,   gap_b_sem=gap_b_sem,
#             gap_m_mean=gap_m_mean,   gap_m_sem=gap_m_sem
#         ))

#     # ---------- plotting ----------
#     T = np.arange(1, t_max + 1)
#     fig, axes = plt.subplots(1, 2, figsize=(
#         12, 3.5), constrained_layout=True)
#     lw = 1.5
#     xticks = [1, max(1, t_max//4), max(1, t_max//2),
#                 max(1, 3*t_max//4), t_max]

#     for ax in axes:
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)

#     n_rates = len(series)
#     sorted_rates = sorted(rates)
#     rate_ranks = {r: i for i, r in enumerate(sorted_rates)}
#     reds = desat_ramp("#7A1F2C", n_rates, s_scale=0.55,
#                         v_low=0.42, v_high=0.88)[::-1]
#     grays = gray_ramp(n_rates, g_low=0.18, g_high=0.70)

#     if trial_type == "stop":
#         style_a_lbl, style_b_lbl, style_m_lbl = "SS", "SE", None
#     else:
#         style_a_lbl, style_b_lbl, style_m_lbl = "GS", "GE", "GM"

#     # ----- Left panel: sigma_t -----
#     ax = axes[0]
#     rate_lines = []
#     for s in series:
#         r = s["rate"]
#         rank = rate_ranks[r]
#         col = reds[rank]
#         # lw = 2.2 - 0.6*rank/(n_rates-1)
#         # al = 0.95 - 0.35*rank/(n_rates-1)
#         lw = 1.5
#         al = 0.18

#         ax.plot(T, s["sigma_a_mean"], color=col, lw=lw, linestyle='-')
#         ax.fill_between(T, s["sigma_a_mean"] - s["sigma_a_sem"],
#                         s["sigma_a_mean"] + s["sigma_a_sem"], color=col, alpha=al)

#         ax.plot(T, s["sigma_b_mean"], color=col, lw=lw, linestyle='--')
#         ax.fill_between(T, s["sigma_b_mean"] - s["sigma_b_sem"],
#                         s["sigma_b_mean"] + s["sigma_b_sem"], color=col, alpha=al)

#         if trial_type == "nonstop":
#             ax.plot(T, s["sigma_m_mean"], color=col, lw=lw, linestyle='-.')
#             ax.fill_between(T, s["sigma_m_mean"] - s["sigma_m_sem"],
#                             s["sigma_m_mean"] + s["sigma_m_sem"], color=col, alpha=al)

#         rate_lines.append(
#             Line2D([0], [0], color=col, lw=lw, label=f"{r:.2f}"))

#     style_lines = [
#         Line2D([0], [0], color='k', lw=lw,
#                 linestyle='-',  label=style_a_lbl),
#         Line2D([0], [0], color='k', lw=lw,
#                 linestyle='--', label=style_b_lbl),
#     ]
#     if trial_type == "nonstop":
#         style_lines.append(
#             Line2D([0], [0], color='k', lw=lw, linestyle='-.', label=style_m_lbl))

#     if trial_type == "stop" and ssd is not None:
#         ax.axvline(x=int(ssd) + 1,      color='k',
#                     linestyle='--', linewidth=1)
#         ax.axvline(x=int(ssd) + 12 + 1, color='k',
#                     linestyle='--', linewidth=1)

#     legend_title = (
#         "Stop Prob. " + r"$\tilde{r}^{\nu}$" + " (color)\n" +
#         ("Outcome (style)" if trial_type == "nonstop" else "Outcome (style)")
#     )

#     leg = ax.legend(
#         handles=rate_lines + style_lines,
#         frameon=False,
#         loc='upper left',
#         bbox_to_anchor=(1.02, 1),
#         title=legend_title,
#         fontsize=12
#     )
#     leg.get_title().set_fontsize(12)

#     ax.set_ylim(0, 1.1)
#     ax.set_xlim(1, t_max)
#     ax.set_xticks(xticks)
#     ax.set_xlabel("Time Step", fontsize=12)
#     ax.set_ylabel(r"Belief State $\sigma^\nu_t$", fontsize=12)

#     # ----- Right panel: gap -----
#     ax = axes[1]
#     rate_lines = []
#     for s in series:
#         r = s["rate"]
#         rank = rate_ranks[r]
#         col = grays[rank]
#         # lw = 2.2 - 0.6*rank/(n_rates-1)
#         # al = 0.95 - 0.35*rank/(n_rates-1)
#         lw = 1.5
#         al = 0.18

#         ax.plot(T, s["gap_a_mean"], color=col, lw=lw, linestyle='-')
#         ax.fill_between(T, s["gap_a_mean"] - s["gap_a_sem"],
#                         s["gap_a_mean"] + s["gap_a_sem"], color=col, alpha=al)

#         ax.plot(T, s["gap_b_mean"], color=col, lw=lw, linestyle='--')
#         ax.fill_between(T, s["gap_b_mean"] - s["gap_b_sem"],
#                         s["gap_b_mean"] + s["gap_b_sem"], color=col, alpha=al)

#         if trial_type == "nonstop":
#             ax.plot(T, s["gap_m_mean"], color=col, lw=lw, linestyle='-.')
#             ax.fill_between(T, s["gap_m_mean"] - s["gap_m_sem"],
#                             s["gap_m_mean"] + s["gap_m_sem"], color=col, alpha=al)

#         rate_lines.append(
#             Line2D([0], [0], color=col, lw=lw, label=f"{r:.2f}"))

#     style_lines = [
#         Line2D([0], [0], color='k', lw=lw,
#                 linestyle='-',  label=style_a_lbl),
#         Line2D([0], [0], color='k', lw=lw,
#                 linestyle='--', label=style_b_lbl),
#     ]
#     if trial_type == "nonstop":
#         style_lines.append(
#             Line2D([0], [0], color='k', lw=lw, linestyle='-.', label=style_m_lbl))

#     if trial_type == "stop" and ssd is not None:
#         ax.axvline(x=int(ssd) + 1,      color='k',
#                     linestyle='--', linewidth=1)
#         ax.axvline(x=int(ssd) + 12 + 1, color='k',
#                     linestyle='--', linewidth=1)

#     leg = ax.legend(
#         handles=rate_lines + style_lines,
#         frameon=False,
#         loc='upper left',
#         bbox_to_anchor=(1.02, 1),
#         title=legend_title,
#         fontsize=12
#     )
#     leg.get_title().set_fontsize(12)

#     ax.set_xlim(1, t_max)
#     ax.set_xticks(xticks)
#     ax.set_xlabel("Time Step", fontsize=12)
#     ax.set_ylabel(
#         r"$Q^\nu_t(W) - \min(Q^\nu_t(L), Q^\nu_t(R))$", fontsize=12)

#     plt.tight_layout()
#     if saveto:
#         fig.savefig(saveto, dpi=300, bbox_inches='tight', pad_inches=0.03)
#         plt.show()
#         plt.close(fig)
#     else:
#         plt.show()

def plot_rate_timecourses(
    model,
    params: dict,
    rates=(0.05, 1/6, 0.35, 0.55, 0.80),
    ssd=None,
    n_batch=500,
    saveto=None,
    trial_type: str = "stop",
    t_max=40
):
    """
    Publication-quality visualization of Belief and Action Value dynamics.
    Matches the style of plot_simu_trial_batch_belief_5conditions.
    """
    
    # ==========================================
    # 🎨 STYLE CONFIG
    # ==========================================
    style = {
        "label_fontsize": 16,
        "title_fontsize": 16,
        "tick_fontsize": 14,
        "legend_fontsize": 16,
        "panel_label_size": 20,
        "line_width_mean": 2,
        "sem_alpha": 0.2,
    }

    # Color Palette (Inferno Reversed) - Professional Gradient
    # Dark Red -> Orange -> Yellow
    cmap = plt.get_cmap('inferno') 
    # Use range 0.2 to 0.85 to avoid indistinguishable blacks or invisible yellows
    color_indices = np.linspace(0.3, 0.7, len(rates)) 
    rate_colors = [cmap(i) for i in color_indices]

    styles = {
        "A": {"ls": "-", "label": "Success"},      # Solid
        "B": {"ls": "--", "label": "Error"},       # Dashed
        "M": {"ls": ":", "label": "Miss"},         # Dotted
    }

    # ==========================================
    # 1. SIMULATION LOOP
    # ==========================================
    trial_type = str(trial_type).lower().strip()
    series = []
    
    def _pad_to_matrix(seq_list, T):
        if not seq_list or len(seq_list) == 0:
            return np.full((0, T), np.nan)
        out = np.full((len(seq_list), T), np.nan)
        for i, s in enumerate(seq_list):
            if s is None: continue
            arr = np.asarray(s, dtype=float)
            limit = min(len(arr), T)
            out[i, :limit] = arr[:limit]
        return out

    def _mean_sem(mat):
        if mat.shape[0] == 0:
            return np.full(mat.shape[1], np.nan), np.full(mat.shape[1], np.nan)
        with np.errstate(invalid='ignore'): 
            mean = np.nanmean(mat, axis=0)
            n = np.sum(~np.isnan(mat), axis=0)
            std = np.nanstd(mat, axis=0, ddof=1)
            sem = np.where(n > 1, std / np.sqrt(n), np.nan)
        return mean, sem

    for r_idx, r in enumerate(rates):
        current_params = params.copy()
        current_params['rate_stop_trial'] = float(r)
        
        m = model.__class__(**current_params)
        m.value_iteration_tensor()

        rec_a, rec_b, rec_m = [], [], []
        
        for _ in range(int(n_batch)):
            true_go = np.random.choice(["left", "right"])
            
            # --- CRITICAL: verbose=True to get sequences ---
            if trial_type == "stop":
                ssd_use = 10 if ssd is None else int(ssd)
                res = simulation.simu_trial(m, true_go, "stop", ssd_use, verbose=True)
                outcome = res["result"] if isinstance(res, dict) else res[0]
                if outcome == "SS": rec_a.append(res)
                elif outcome == "SE": rec_b.append(res)
            else:
                res = simulation.simu_trial(m, true_go, "nonstop", None, verbose=True)
                outcome = res["result"] if isinstance(res, dict) else res[0]
                if outcome == "GS": rec_a.append(res)
                elif outcome == "GE": rec_b.append(res)
                elif outcome == "GM": rec_m.append(res)

        data_packet = {"rate": r, "color": rate_colors[r_idx]}
        
        for cat, rec_list in zip(["A", "B", "M"], [rec_a, rec_b, rec_m]):
            df = pd.DataFrame(rec_list)
            
            # 1. Belief (Sigma)
            if "sigma_seq" in df.columns:
                mat_sig = _pad_to_matrix(df["sigma_seq"].tolist(), t_max)
                mu, se = _mean_sem(mat_sig)
            else:
                mu, se = np.full(t_max, np.nan), np.full(t_max, np.nan)
            
            data_packet[f"sigma_{cat}_mean"] = mu
            data_packet[f"sigma_{cat}_sem"] = se

            # 2. Value Gap
            if "value_wait_seq" in df.columns:
                vl = _pad_to_matrix(df["value_left_seq"].tolist(), t_max)
                vr = _pad_to_matrix(df["value_right_seq"].tolist(), t_max)
                vw = _pad_to_matrix(df["value_wait_seq"].tolist(), t_max)

                if vl.shape[0] > 0:
                    min_go = np.nanmin(np.stack([vl, vr], axis=0), axis=0)
                    gap = vw - min_go
                    mu_g, se_g = _mean_sem(gap)
                else:
                    mu_g, se_g = np.full(t_max, np.nan), np.full(t_max, np.nan)
            else:
                mu_g, se_g = np.full(t_max, np.nan), np.full(t_max, np.nan)
                
            data_packet[f"gap_{cat}_mean"] = mu_g
            data_packet[f"gap_{cat}_sem"] = se_g

        series.append(data_packet)

    # ==========================================
    # 2. PLOTTING
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(16*0.8, 6*0.8))
    plt.subplots_adjust(wspace=0.4)

    x_vals = np.arange(1, t_max + 1)
    
    # Standardized Ticks per request: 1, 10, 20, 30, 40
    xticks_val = [1, 10, 20, 30, 40]
    
    ssd_lines = [ssd + 1, ssd + 12 + 1] if ssd else []

    # Draw Lines
    for i, s in enumerate(series):
        col = s["color"]
        for cat in ["A", "B", "M"]:
            if trial_type == "stop" and cat == "M": continue
            
            sig_mu = s[f"sigma_{cat}_mean"]
            sig_sem = s[f"sigma_{cat}_sem"]
            gap_mu = s[f"gap_{cat}_mean"]
            gap_sem = s[f"gap_{cat}_sem"]
            
            ls = styles[cat]["ls"]
            
            # Panel A: Belief
            if not np.all(np.isnan(sig_mu)):
                axes[0].plot(x_vals, sig_mu, color=col, linestyle=ls, lw=style['line_width_mean'])
                axes[0].fill_between(x_vals, sig_mu - sig_sem, sig_mu + sig_sem,
                                     color=col, alpha=style['sem_alpha'], lw=0)

            # Panel B: Value Gap
            if not np.all(np.isnan(gap_mu)):
                axes[1].plot(x_vals, gap_mu, color=col, linestyle=ls, lw=style['line_width_mean'])
                axes[1].fill_between(x_vals, gap_mu - gap_sem, gap_mu + gap_sem,
                                     color=col, alpha=style['sem_alpha'], lw=0)

    # Formatting
    for ax in axes:
        ax.set_xlabel("Time Step", fontsize=style['label_fontsize'])
        ax.set_xticks(xticks_val) # <--- Force 1, 10, 20, 30, 40
        ax.tick_params(axis='both', labelsize=style['tick_fontsize'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(x_vals[0], x_vals[-1])
        
        # SSD Markers
        if trial_type == "stop" and ssd is not None:
            for line_x in ssd_lines:
                ax.axvline(x=line_x, color='k', linestyle='--', linewidth=1.2, alpha=0.8)

    axes[0].set_ylabel(r"Belief State $\sigma^\nu_t$", fontsize=style['label_fontsize'])
    axes[1].set_ylabel(r"Value Gap $Q^\nu_t(W) - Q^\nu_t(Go)$", fontsize=style['label_fontsize'])
    axes[0].set_ylim(0, 1.05)

    # Panel Tags (A / B)
    axes[0].text(-0.15, 1.05, "A", transform=axes[0].transAxes, 
                 fontsize=style['panel_label_size'], fontweight='bold', va='top', ha='right')
    axes[1].text(-0.15, 1.05, "B", transform=axes[1].transAxes, 
                 fontsize=style['panel_label_size'], fontweight='bold', va='top', ha='right')

    # Legend Construction
    # 1. Create Handle List for RATES
    rate_handles = []
    for r, col in zip(rates, rate_colors):
        rate_handles.append(Line2D([0], [0], color=col, lw=3, label=f"{r:.2f}"))
    
    # 2. Create Handle List for OUTCOMES
    outcome_handles = []
    if trial_type == "stop":
        outcome_handles.append(Line2D([0], [0], color='k', ls=styles["A"]["ls"], lw=2, label="Stop Success"))
        outcome_handles.append(Line2D([0], [0], color='k', ls=styles["B"]["ls"], lw=2, label="Stop Error"))
    else:
        outcome_handles.append(Line2D([0], [0], color='k', ls=styles["A"]["ls"], lw=2, label="Go Success"))
        outcome_handles.append(Line2D([0], [0], color='k', ls=styles["B"]["ls"], lw=2, label="Go Error"))
        outcome_handles.append(Line2D([0], [0], color='k', ls=styles["M"]["ls"], lw=2, label="Go Miss"))

    # 3. Add First Legend (Rates) - Top Right
    leg1 = axes[1].legend(
        handles=rate_handles, 
        title=r'$\bf{Stop\ Prior\ \tilde{r}^\nu}$',
        loc='upper left', 
        bbox_to_anchor=(1.05, 1.02),  # Position: Top aligned
        frameon=False, 
        fontsize=style['legend_fontsize'],
        title_fontsize=style['legend_fontsize'] + 1
    )
    # Force Left Alignment of Title and Handles
    leg1._legend_box.align = "left" 
    
    # Add leg1 to the plot so it persists when we add leg2
    axes[1].add_artist(leg1)

    # 4. Add Second Legend (Outcomes) - Below the first
    leg2 = axes[1].legend(
        handles=outcome_handles, 
        title=r'$\bf{Outcome}$',
        loc='upper left', 
        bbox_to_anchor=(1.05, 0.45), # Position: Shifted down
        frameon=False, 
        fontsize=style['legend_fontsize'],
        title_fontsize=style['legend_fontsize'] + 1
    )
    leg2._legend_box.align = "left"

    if saveto:
        fig.savefig(saveto, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()



def plot_rate_behavior(
    model,
    params: dict,
    rates=(0.05, 0.10, 1/6, 0.30, 0.50),
    ssd=10,
    n_session_trials=1000,
    step_in_ms=25.0,
    saveto=None,
    rt_ylim_ms=(0, 1000),  # set None to auto-scale
):
    """
    Two-panel behavioral summary across stop rates:
    Left: Accuracy (Go & Stop) with SEM
    Right: RT (Go-success & Stop-error) [ms] with SEM

    Styling: all black, mean±SEM, no top/right spines.
    """

    # --- simulate sessions at each rate ---
    rows = []
    for r in rates:
        m = model.__class__(rate_stop_trial=float(r), **params)
        m.value_iteration_tensor()
        df = _simulate_mixed_session(m, n_trials=n_session_trials, ssd=ssd)
        stats = _summarize_behavior_with_sem(df, step_in_ms=step_in_ms)
        rows.append(dict(rate=float(r), **stats))
    beh = pd.DataFrame(rows).sort_values("rate")

    # --- figure ---
    fig, axes = plt.subplots(1, 2, figsize=(
        12, 3.5), constrained_layout=True)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="both",
                        direction="out", length=3, width=0.9)

    # common kwargs
    k_go = dict(color="black", marker="o", linestyle="-",
                linewidth=1.6, markersize=5, capsize=3)
    k_stop = dict(color="black", marker="o", linestyle="--",
                    linewidth=1.6, markersize=5, capsize=3)

    # ----- LEFT: Accuracy -----
    ax = axes[0]
    ax.errorbar(beh["rate"], beh["go_acc"],
                yerr=beh["go_acc_sem"],   label="Go",   **k_go)
    ax.errorbar(beh["rate"], beh["stop_succ"],
                yerr=beh["stop_succ_sem"], label="Stop", **k_stop)
    ax.set_xlabel(r"Stop Probability $\tilde{r}^{\nu}$", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.1)
    # ax.legend(frameon=False, ncol=2)
    ax.legend(
        frameon=False,
        ncol=1,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0)
    )

    # ----- RIGHT: RTs -----
    ax = axes[1]
    ax.errorbar(beh["rate"], beh["rt_go_succ"],
                yerr=beh["rt_go_succ_sem"],  label="Go (Success)",  **k_go)
    ax.errorbar(beh["rate"], beh["rt_stop_err"],
                yerr=beh["rt_stop_err_sem"], label="Stop (Error)", **k_stop)
    ax.set_xlabel(r"Stop Probability $\tilde{r}^{\nu}$", fontsize=12)
    ax.set_ylabel("Reaction Time (ms)", fontsize=12)
    if rt_ylim_ms is None:
        ymax = np.nanmax([beh["rt_go_succ"], beh["rt_stop_err"]]) * 1.1
        ax.set_ylim(0, max(600, float(ymax)))
    else:
        ax.set_ylim(*rt_ylim_ms)
    # ax.legend(frameon=False, ncol=1)
    ax.legend(
        frameon=False,
        ncol=1,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0)
    )

    # tidy
    for ax in axes:
        ax.margins(x=0.05)

    # save or show
    if saveto is not None:
        fig.canvas.draw()
        fig.savefig(saveto, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.show()
        plt.close(fig)
    else:
        plt.show()

    return saveto, beh