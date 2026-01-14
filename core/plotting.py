
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from core import simulation

def plot_simu_trial_batch_belief_5conditions(out_go, out_stop, show_cnt=5, saveto=None):
    # ==========================================
    # 🎨 TUNING SECTION: ADJUST STYLE HERE
    # ==========================================
    style = {
        # Mean Lines (The thick background curves)
        "mean_lw": 1.5,          # Thickness
        "mean_alpha": 1.0,       # Opacity (0.0 - 1.0)
        "sem_alpha": 0.4,        # Opacity of the shaded error band

        # Individual Samples (The wiggly lines)
        "sample_active_lw": 0.7,    # Thickness (Pre-decision)
        "sample_active_alpha": 0.9, # Opacity (Pre-decision)
        
        "sample_shadow_lw": 0.7,    # Thickness (Post-decision ghost)
        "sample_shadow_alpha": 0.2, # Opacity (Post-decision ghost)

        # Decision Dot (The marker at the decision point)
        "marker_size": 30,       # Size of the dot
        "marker_alpha": 1.0,     # Opacity
        "marker_edge_width": 0.5,# White border thickness around dot

        # SSD Vertical Lines
        "ssd_lw": 1.2,
        "ssd_alpha": 0.8,
    }
    # ==========================================

    colors = {
        "beta": "#4E7C59",
        "sigma": "#B75D69",
        "Q_L": "#E3B23C",
        "Q_R": "#3A6EA5",
        "Q_W": "#7A7A7A",
    }

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

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    x_range_full = np.arange(1, 41) 
    xticks = [1, 10, 20, 30, 40]

    for i, df in enumerate(df_list):
        row, col = divmod(i, 3)
        axis = ax[row, col]

        if len(df) == 0:
            axis.axis('off')
            continue

        # --- 1. Plot Means (Background) ---
        beta_mean = df['beta_seq'].apply(lambda x: pd.Series(x)).mean()
        beta_sem = df['beta_seq'].apply(lambda x: pd.Series(x)).sem()
        sigma_mean = df['sigma_seq'].apply(lambda x: pd.Series(x)).mean()
        sigma_sem = df['sigma_seq'].apply(lambda x: pd.Series(x)).sem()

        curr_len_mean = len(beta_mean)
        x_range_mean = x_range_full[:curr_len_mean]

        axis.plot(x_range_mean, beta_mean, label=r'$\beta^\nu_t$',
                  color=colors["beta"], linewidth=style['mean_lw'], alpha=style['mean_alpha'])
        axis.fill_between(x_range_mean, beta_mean - beta_sem, beta_mean + beta_sem,
                          color=colors["beta"], alpha=style['sem_alpha'])

        axis.plot(x_range_mean, sigma_mean, label=r'$\sigma^\nu_t$',
                  color=colors["sigma"], linewidth=style['mean_lw'], alpha=style['mean_alpha'])
        axis.fill_between(x_range_mean, sigma_mean - sigma_sem, sigma_mean + sigma_sem,
                          color=colors["sigma"], alpha=style['sem_alpha'])

        # --- 2. Plot Individual Samples ---
        for j in range(min(show_cnt, len(df))):
            row_data = df.iloc[j]
            beta_seq = row_data['beta_seq']
            sigma_seq = row_data['sigma_seq']
            policy_seq = row_data['policy_seq']

            # Find decision time
            decision_idx = len(policy_seq) - 1
            decision_made = False
            for t_idx, act in enumerate(policy_seq):
                if act != 2: 
                    decision_idx = t_idx
                    decision_made = True
                    break
            
            seq_len = len(beta_seq)
            x_vals = x_range_full[:seq_len]

            # --- Plot Beta ---
            # Active
            axis.plot(x_vals[:decision_idx+1], beta_seq[:decision_idx+1],
                      color=colors["beta"], linewidth=style['sample_active_lw'], alpha=style['sample_active_alpha'])
            # Shadow
            if decision_idx < seq_len - 1:
                axis.plot(x_vals[decision_idx:], beta_seq[decision_idx:],
                          color=colors["beta"], linewidth=style['sample_shadow_lw'], alpha=style['sample_shadow_alpha'])
            # Marker
            if decision_made:
                axis.scatter(x_vals[decision_idx], beta_seq[decision_idx], 
                             s=style['marker_size'], color=colors["beta"], alpha=style['marker_alpha'], 
                             zorder=3, edgecolors='white', linewidth=style['marker_edge_width'])

            # --- Plot Sigma ---
            # Active
            axis.plot(x_vals[:decision_idx+1], sigma_seq[:decision_idx+1],
                      color=colors["sigma"], linewidth=style['sample_active_lw'], alpha=style['sample_active_alpha'])
            # Shadow
            if decision_idx < seq_len - 1:
                axis.plot(x_vals[decision_idx:], sigma_seq[decision_idx:],
                          color=colors["sigma"], linewidth=style['sample_shadow_lw'], alpha=style['sample_shadow_alpha'])
            # Marker
            if decision_made:
                axis.scatter(x_vals[decision_idx], sigma_seq[decision_idx], 
                             s=style['marker_size'], color=colors["sigma"], alpha=style['marker_alpha'], 
                             zorder=3, edgecolors='white', linewidth=style['marker_edge_width'])

        # --- Formatting ---
        if condition_names[i].startswith('Stop') and ssd is not None:
            axis.axvline(x=ssd + 1, color='k', linestyle='--', linewidth=style['ssd_lw'], alpha=style['ssd_alpha'])
            axis.axvline(x=ssd + 12 + 1, color='k', linestyle='--', linewidth=style['ssd_lw'], alpha=style['ssd_alpha'])

        axis.set_xlim(1, 40)
        axis.set_ylim(0, 1.1)
        axis.set_xticks(xticks)
        axis.set_yticks(np.linspace(0, 1, 5))
        axis.set_title(f"{condition_names[i]}", pad=12, fontsize=14)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    ax[1, 2].axis("off")
    plt.subplots_adjust(hspace=0.5, wspace=0.3, top=0.92, bottom=0.08, left=0.08, right=0.98)

    # Legend
    handles, labels = ax[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    dummy = Line2D([], [], color='none', label='Belief State:')
    final_handles = [dummy] + [by_label[k] for k in [r'$\beta^\nu_t$', r'$\sigma^\nu_t$']]
    final_labels = ['Belief State'] + [r'$\beta^\nu_t$', r'$\sigma^\nu_t$']
    
    fig.legend(final_handles, final_labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False, fontsize=13)

    if saveto:
        fig.savefig(saveto, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_simu_trial_batch_action_value_5conditions(out_go, out_stop, show_cnt=5, saveto=None):
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    STYLE = {
        "mean_lw": 1.5, "mean_alpha": 1.0, "sem_alpha": 0.4,
        "sample_active_lw": 0.7, "sample_active_alpha": 0.9,
        "sample_shadow_lw": 0.7, "sample_shadow_alpha": 0.2,
        "marker_size": 30, "marker_alpha": 1.0, "marker_edge_width": 0.5,
        "ssd_lw": 1.2, "ssd_alpha": 0.8,
    }

    # Centralized config for the 3 curves (Column Name, Label, Color)
    METRICS = [
        ('value_left_seq',  r'$Q^\nu_t(\beta^\nu_t, L)$', "#E3B23C"),
        ('value_right_seq', r'$Q^\nu_t(\beta^\nu_t, R)$', "#3A6EA5"),
        ('value_wait_seq',  r'$Q^\nu_t(\beta^\nu_t, W)$', "#7A7A7A")
    ]

    # ==========================================
    # 2. DATA PREPARATION
    # ==========================================
    df_go = pd.DataFrame(out_go)
    df_stop = pd.DataFrame(out_stop)
    
    # Safely get SSD (handling empty stop df)
    ssd = df_stop.iloc[0]['ssd'] if not df_stop.empty else None

    # Define conditions and titles
    conditions = [
        (df_go, 'GS', 'Go Success'),
        (df_go, 'GE', 'Go Error'),
        (df_go, 'GM', 'Go Missing'),
        (df_stop, 'SS', 'Stop Success'),
        (df_stop, 'SE', 'Stop Error')
    ]

    # ==========================================
    # 3. PLOTTING LOOP
    # ==========================================
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    ax_flat = axes.flatten() # Flattens the 2x3 grid to 1D array for easier indexing
    
    x_range_full = np.arange(1, 41)
    
    for i, (source_df, result_code, title) in enumerate(conditions):
        ax = ax_flat[i]
        
        # Filter Data
        df = source_df[source_df['result'] == result_code]
        if df.empty:
            ax.axis('off')
            continue

        # --- A. Plot Means & SEM (Background) ---
        for col_name, label, color in METRICS:
            # OPTIMIZATION: Convert list-column to DataFrame efficiently
            expanded_data = pd.DataFrame(df[col_name].tolist())
            
            mean_seq = expanded_data.mean(axis=0)
            sem_seq = expanded_data.sem(axis=0)
            x_vals = x_range_full[:len(mean_seq)]

            ax.plot(x_vals, mean_seq, label=label, color=color, 
                    lw=STYLE['mean_lw'], alpha=STYLE['mean_alpha'])
            ax.fill_between(x_vals, mean_seq - sem_seq, mean_seq + sem_seq, 
                            color=color, alpha=STYLE['sem_alpha'])

        # --- B. Plot Individual Samples ---
        # Only take the first 'show_cnt' rows
        subset = df.iloc[:show_cnt]
        
        for _, row_data in subset.iterrows():
            policy_seq = row_data['policy_seq']
            
            # Find decision point (first index where action != 2)
            # np.where returns a tuple, we take the first element if it exists
            decision_indices = np.where(np.array(policy_seq) != 2)[0]
            if len(decision_indices) > 0:
                decision_idx = decision_indices[0]
                decision_made = True
            else:
                decision_idx = len(policy_seq) - 1
                decision_made = False

            # Inner helper to draw segments
            def draw_segment(vals, color):
                seq_len = len(vals)
                x_vals = x_range_full[:seq_len]
                
                # 1. Active Path (up to decision)
                ax.plot(x_vals[:decision_idx+1], vals[:decision_idx+1],
                        color=color, alpha=STYLE['sample_active_alpha'], lw=STYLE['sample_active_lw'])
                
                # 2. Decision Dot
                if decision_made:
                    ax.scatter(x_vals[decision_idx], vals[decision_idx],
                               s=STYLE['marker_size'], color=color, alpha=STYLE['marker_alpha'], 
                               zorder=3, edgecolors='white', lw=STYLE['marker_edge_width'])
                
                # 3. Shadow Path (after decision)
                if decision_idx < seq_len - 1:
                    ax.plot(x_vals[decision_idx:], vals[decision_idx:],
                            color=color, alpha=STYLE['sample_shadow_alpha'], lw=STYLE['sample_shadow_lw'])

            # Loop through metrics to plot segments
            for col_name, _, color in METRICS:
                draw_segment(row_data[col_name], color)

        # --- C. Aesthetics & SSD ---
        if title.startswith('Stop') and ssd is not None:
            for x_line in [ssd + 1, ssd + 12 + 1]:
                ax.axvline(x=x_line, color='k', ls='--', lw=STYLE['ssd_lw'], alpha=STYLE['ssd_alpha'])

        ax.set_xlim(1, 40)
        ax.set_xticks([1, 10, 20, 30, 40])
        ax.set_title(title, fontsize=14, pad=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Turn off the unused 6th subplot (index 5)
    ax_flat[5].axis("off")
    
    plt.subplots_adjust(hspace=0.5, wspace=0.3, top=0.92, bottom=0.08, left=0.08, right=0.98)

    # ==========================================
    # 4. ROBUST LEGEND
    # ==========================================
    # Gather handles from the first valid plot (ax_flat[0])
    handles, labels = ax_flat[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Extract just the labels we want from our METRICS config
    target_keys = [m[1] for m in METRICS] # Gets the LaTeX strings
    
    dummy = Line2D([], [], color='none')
    final_handles = [dummy]
    final_labels = ['Action Value']

    for k in target_keys:
        if k in by_label:
            final_handles.append(by_label[k])
            final_labels.append(k)
    
    fig.legend(
        final_handles, final_labels, 
        loc='upper center', bbox_to_anchor=(0.5, 1.05), 
        ncol=len(final_handles), frameon=False, fontsize=13
    )

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

def plot_rate_timecourses(
    model,
    params: dict,
    rates=(0.05, 1/6, 0.35, 0.55, 0.80),
    ssd=None,
    n_batch=300,
    t_max=40,
    saveto=None,
    trial_type: str = "stop",   # "stop" or "nonstop"
):
    """
    Across-rate within-trial dynamics.

    If trial_type == "stop":
        - Simulate stop trials with the given SSD (if None, default to 10).
        - Left: mean sigma_t (Stop belief), colored by stop-rate (darker = higher rate).
        - Right: gap Q(W) - min(Q(L), Q(R)), gray gradient, outcome styles SS(solid)/SE(dashed).
        - SSD markers (vertical) at ssd+1 and ssd+12+1 if ssd is not None.

    If trial_type == "nonstop":
        - Simulate go-only (nonstop) trials; SSD is ignored.
        - Left/Right panels as above; outcome styles GS(solid)/GE(dashed)/GM(dot-dash).
        - No SSD markers.

    Note:
        Color encodes the across-trial stop probability r~ even for nonstop,
        because it still shapes priors and value/belief dynamics.
    """
    import numpy as np
    from matplotlib.colors import to_rgb, rgb_to_hsv, hsv_to_rgb

    def desat_ramp(base="#7A1F2C", n=5, s_scale=0.55, v_low=0.42, v_high=0.88):
        b = np.array(to_rgb(base))
        h, s, v = rgb_to_hsv(b[np.newaxis, np.newaxis, :])[0, 0]
        s = np.clip(s * s_scale, 0, 1)
        vals = np.linspace(v_low, v_high, n)
        return [tuple(hsv_to_rgb([h, s, vv])) for vv in vals]

    def gray_ramp(n=5, g_low=0.18, g_high=0.70):
        return [(g, g, g) for g in np.linspace(g_high, g_low, n)]

    trial_type = str(trial_type).lower().strip()
    if trial_type not in ("stop", "nonstop"):
        raise ValueError("trial_type must be 'stop' or 'nonstop'.")

    # ---------- helpers ----------
    def _pad_to_matrix(seq_list, T):
        if len(seq_list) == 0:
            return np.full((0, T), np.nan)
        out = np.full((len(seq_list), T), np.nan)
        for i, s in enumerate(seq_list):
            if s is None:
                continue
            arr = np.asarray(s, dtype=float)
            out[i, :min(len(arr), T)] = arr[:min(len(arr), T)]
        return out

    def _mean_sem(mat):
        mean = np.nanmean(mat, axis=0)
        n = np.sum(~np.isnan(mat), axis=0)
        std = np.nanstd(mat, axis=0, ddof=1)
        sem = np.where(n > 1, std / np.sqrt(n), np.nan)  # require n>1
        return mean, sem

    def _sigma_mean_sem(df):
        if len(df) == 0 or "sigma_seq" not in df.columns:
            return np.full(t_max, np.nan), np.full(t_max, np.nan)
        mat = _pad_to_matrix(df["sigma_seq"].tolist(), t_max)
        return _mean_sem(mat)

    def _gap_mean_sem(df):
        if len(df) == 0:
            return np.full(t_max, np.nan), np.full(t_max, np.nan)
        vl = _pad_to_matrix(df.get("value_left_seq", []).tolist(), t_max)
        vr = _pad_to_matrix(df.get("value_right_seq", []).tolist(), t_max)
        vw = _pad_to_matrix(df.get("value_wait_seq", []).tolist(), t_max)
        if vl.shape[0] == 0 or vr.shape[0] == 0 or vw.shape[0] == 0:
            return np.full(t_max, np.nan), np.full(t_max, np.nan)
        min_go = np.nanmin(np.stack([vl, vr], axis=0), axis=0)
        gap = vw - min_go
        return _mean_sem(gap)

    # ---------- simulate per rate ----------
    series = []
    for r in rates:
        current_params = params.copy()
        current_params['rate_stop_trial'] = float(r)

        m = model.__class__(**current_params)
        m.value_iteration_tensor()

        # A=success, B=error, M=miss (GM only)
        rec_a, rec_b, rec_m = [], [], []
        n_iter = int(n_batch)

        for _ in range(n_iter):
            true_go = np.random.choice(["left", "right"])

            if trial_type == "stop":
                ssd_use = 10 if ssd is None else int(ssd)
                res = simulation.simu_trial(m, true_go, "stop", ssd_use, verbose=True)
                outcome = res["result"] if isinstance(res, dict) else (
                    res[0] if isinstance(res, (list, tuple)) else None)
                if outcome == "SS":
                    rec_a.append(res)
                elif outcome == "SE":
                    rec_b.append(res)
                # no "SM" in stop trials; keep two buckets
            else:
                res = simulation.simu_trial(m, true_go, "nonstop", None, verbose=True)
                outcome = res["result"] if isinstance(res, dict) else (
                    res[0] if isinstance(res, (list, tuple)) else None)
                if outcome == "GS":
                    rec_a.append(res)
                elif outcome == "GE":
                    rec_b.append(res)
                elif outcome == "GM":
                    rec_m.append(res)

        df_a, df_b, df_m = pd.DataFrame(
            rec_a), pd.DataFrame(rec_b), pd.DataFrame(rec_m)

        sigma_a_mean, sigma_a_sem = _sigma_mean_sem(df_a)
        sigma_b_mean, sigma_b_sem = _sigma_mean_sem(df_b)
        sigma_m_mean, sigma_m_sem = _sigma_mean_sem(df_m)

        gap_a_mean, gap_a_sem = _gap_mean_sem(df_a)
        gap_b_mean, gap_b_sem = _gap_mean_sem(df_b)
        gap_m_mean, gap_m_sem = _gap_mean_sem(df_m)

        series.append(dict(
            rate=float(r),
            sigma_a_mean=sigma_a_mean, sigma_a_sem=sigma_a_sem,
            sigma_b_mean=sigma_b_mean, sigma_b_sem=sigma_b_sem,
            sigma_m_mean=sigma_m_mean, sigma_m_sem=sigma_m_sem,
            gap_a_mean=gap_a_mean,   gap_a_sem=gap_a_sem,
            gap_b_mean=gap_b_mean,   gap_b_sem=gap_b_sem,
            gap_m_mean=gap_m_mean,   gap_m_sem=gap_m_sem
        ))

    # ---------- plotting ----------
    T = np.arange(1, t_max + 1)
    fig, axes = plt.subplots(1, 2, figsize=(
        12, 3.5), constrained_layout=True)
    lw = 1.5
    xticks = [1, max(1, t_max//4), max(1, t_max//2),
                max(1, 3*t_max//4), t_max]

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    n_rates = len(series)
    sorted_rates = sorted(rates)
    rate_ranks = {r: i for i, r in enumerate(sorted_rates)}
    reds = desat_ramp("#7A1F2C", n_rates, s_scale=0.55,
                        v_low=0.42, v_high=0.88)[::-1]
    grays = gray_ramp(n_rates, g_low=0.18, g_high=0.70)

    if trial_type == "stop":
        style_a_lbl, style_b_lbl, style_m_lbl = "SS", "SE", None
    else:
        style_a_lbl, style_b_lbl, style_m_lbl = "GS", "GE", "GM"

    # ----- Left panel: sigma_t -----
    ax = axes[0]
    rate_lines = []
    for s in series:
        r = s["rate"]
        rank = rate_ranks[r]
        col = reds[rank]
        # lw = 2.2 - 0.6*rank/(n_rates-1)
        # al = 0.95 - 0.35*rank/(n_rates-1)
        lw = 1.5
        al = 0.18

        ax.plot(T, s["sigma_a_mean"], color=col, lw=lw, linestyle='-')
        ax.fill_between(T, s["sigma_a_mean"] - s["sigma_a_sem"],
                        s["sigma_a_mean"] + s["sigma_a_sem"], color=col, alpha=al)

        ax.plot(T, s["sigma_b_mean"], color=col, lw=lw, linestyle='--')
        ax.fill_between(T, s["sigma_b_mean"] - s["sigma_b_sem"],
                        s["sigma_b_mean"] + s["sigma_b_sem"], color=col, alpha=al)

        if trial_type == "nonstop":
            ax.plot(T, s["sigma_m_mean"], color=col, lw=lw, linestyle='-.')
            ax.fill_between(T, s["sigma_m_mean"] - s["sigma_m_sem"],
                            s["sigma_m_mean"] + s["sigma_m_sem"], color=col, alpha=al)

        rate_lines.append(
            Line2D([0], [0], color=col, lw=lw, label=f"{r:.2f}"))

    style_lines = [
        Line2D([0], [0], color='k', lw=lw,
                linestyle='-',  label=style_a_lbl),
        Line2D([0], [0], color='k', lw=lw,
                linestyle='--', label=style_b_lbl),
    ]
    if trial_type == "nonstop":
        style_lines.append(
            Line2D([0], [0], color='k', lw=lw, linestyle='-.', label=style_m_lbl))

    if trial_type == "stop" and ssd is not None:
        ax.axvline(x=int(ssd) + 1,      color='k',
                    linestyle='--', linewidth=1)
        ax.axvline(x=int(ssd) + 12 + 1, color='k',
                    linestyle='--', linewidth=1)

    legend_title = (
        "Stop Prob. " + r"$\tilde{r}^{\nu}$" + " (color)\n" +
        ("Outcome (style)" if trial_type == "nonstop" else "Outcome (style)")
    )

    leg = ax.legend(
        handles=rate_lines + style_lines,
        frameon=False,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        title=legend_title,
        fontsize=12
    )
    leg.get_title().set_fontsize(12)

    ax.set_ylim(0, 1.1)
    ax.set_xlim(1, t_max)
    ax.set_xticks(xticks)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel(r"Belief State $\sigma^\nu_t$", fontsize=12)

    # ----- Right panel: gap -----
    ax = axes[1]
    rate_lines = []
    for s in series:
        r = s["rate"]
        rank = rate_ranks[r]
        col = grays[rank]
        # lw = 2.2 - 0.6*rank/(n_rates-1)
        # al = 0.95 - 0.35*rank/(n_rates-1)
        lw = 1.5
        al = 0.18

        ax.plot(T, s["gap_a_mean"], color=col, lw=lw, linestyle='-')
        ax.fill_between(T, s["gap_a_mean"] - s["gap_a_sem"],
                        s["gap_a_mean"] + s["gap_a_sem"], color=col, alpha=al)

        ax.plot(T, s["gap_b_mean"], color=col, lw=lw, linestyle='--')
        ax.fill_between(T, s["gap_b_mean"] - s["gap_b_sem"],
                        s["gap_b_mean"] + s["gap_b_sem"], color=col, alpha=al)

        if trial_type == "nonstop":
            ax.plot(T, s["gap_m_mean"], color=col, lw=lw, linestyle='-.')
            ax.fill_between(T, s["gap_m_mean"] - s["gap_m_sem"],
                            s["gap_m_mean"] + s["gap_m_sem"], color=col, alpha=al)

        rate_lines.append(
            Line2D([0], [0], color=col, lw=lw, label=f"{r:.2f}"))

    style_lines = [
        Line2D([0], [0], color='k', lw=lw,
                linestyle='-',  label=style_a_lbl),
        Line2D([0], [0], color='k', lw=lw,
                linestyle='--', label=style_b_lbl),
    ]
    if trial_type == "nonstop":
        style_lines.append(
            Line2D([0], [0], color='k', lw=lw, linestyle='-.', label=style_m_lbl))

    if trial_type == "stop" and ssd is not None:
        ax.axvline(x=int(ssd) + 1,      color='k',
                    linestyle='--', linewidth=1)
        ax.axvline(x=int(ssd) + 12 + 1, color='k',
                    linestyle='--', linewidth=1)

    leg = ax.legend(
        handles=rate_lines + style_lines,
        frameon=False,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        title=legend_title,
        fontsize=12
    )
    leg.get_title().set_fontsize(12)

    ax.set_xlim(1, t_max)
    ax.set_xticks(xticks)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel(
        r"$Q^\nu_t(W) - \min(Q^\nu_t(L), Q^\nu_t(R))$", fontsize=12)

    plt.tight_layout()
    if saveto:
        fig.savefig(saveto, dpi=300, bbox_inches='tight', pad_inches=0.03)
        plt.show()
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