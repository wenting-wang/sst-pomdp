import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, linregress
import warnings

# Suppress runtime warnings often caused by division by zero in empty categories
warnings.simplefilter("ignore", RuntimeWarning)

# --- 1. Basic Helpers ---

def get_percentage(df):
    """
    Computes proportions of response types:
    - GS: Go Success
    - GE: Go Error
    - GM: Go Miss
    - SS: Stop Success
    """
    stats = df['result'].value_counts()
    n_gs = stats.get("GS", 0)
    n_ge = stats.get("GE", 0)
    n_gm = stats.get("GM", 0)
    n_ss = stats.get("SS", 0)
    n_se = stats.get("SE", 0)
    
    n_go = n_gs + n_ge + n_gm
    n_stop = n_ss + n_se
    
    perc_gs = n_gs / n_go if n_go > 0 else 0
    perc_ge = n_ge / n_go if n_go > 0 else 0
    perc_gm = n_gm / n_go if n_go > 0 else 0
    perc_ss = n_ss / n_stop if n_stop > 0 else 0
    
    return perc_gs, perc_ge, perc_gm, perc_ss


# --- 2. Distance Metrics (Observed vs Simulated) ---

def get_distance(df_obs, df_sim):
    """
    Main objective function. Calculates a tuple of discrepancy metrics 
    between observed and simulated data.
    """
    # 1. Percentage Distances
    dis_perc = get_dis_perc(df_obs, df_sim)
    
    # 2. RT Wasserstein Distances (Normalized)
    dis_ws_gs = get_ws_distance(df_obs, df_sim, "GS")
    dis_ws_ge = get_ws_distance(df_obs, df_sim, "GE")
    dis_ws_se = get_ws_distance(df_obs, df_sim, "SE")

    # 3. KS Distances (Distribution Shape)
    dis_ks_gs = get_ks_distance(df_obs, df_sim, 'rt', 'GS')
    dis_ks_se = get_ks_distance(df_obs, df_sim, 'rt', 'SE')

    # 4. SSD Mean Difference
    dis_ssd_mean = get_dis_ssd_mean(df_obs, df_sim)

    # Return flattened tuple of 10 metrics
    return (
        *dis_perc,      # (perc_gs, perc_ge, perc_gm, perc_ss)
        dis_ws_gs, dis_ws_ge, dis_ws_se,
        dis_ks_gs, dis_ks_se,
        dis_ssd_mean
    )


def get_dis_perc(df_obs, df_sim):
    """Absolute difference in response type proportions."""
    p_obs = get_percentage(df_obs)
    p_sim = get_percentage(df_sim)
    
    # Calculate absolute difference for each component (GS, GE, GM, SS)
    return tuple(abs(o - s) for o, s in zip(p_obs, p_sim))


def get_ws_distance(df_obs, df_sim, result_type):
    """
    Wasserstein distance for RT distributions, normalized by a factor (40).
    """
    try:
        obs = df_obs[df_obs['result'] == result_type]['rt']
        sim = df_sim[df_sim['result'] == result_type]['rt']
        
        if len(obs) == 0 or len(sim) == 0:
            return 1.0
            
        # Normalization factor 40 is likely specific to model grid/resolution
        return wasserstein_distance(obs, sim) / 40.0
    except ValueError:
        return 1.0


def get_ks_distance(df_obs, df_sim, column, result_type):
    """
    Kolmogorov-Smirnov distance between two empirical RT distributions.
    """
    df_obs_f = df_obs[df_obs['result'] == result_type]
    df_sim_f = df_sim[df_sim['result'] == result_type]

    x = df_obs_f[column].dropna().values.astype(float)
    y = df_sim_f[column].dropna().values.astype(float)

    if x.size == 0 or y.size == 0:
        return np.nan

    # Compute KS statistic manually for efficiency on sorted arrays
    x.sort()
    y.sort()
    
    grid = np.unique(np.concatenate([x, y]))
    if grid.size == 0:
        return np.nan
        
    Fx = np.searchsorted(x, grid, side='right') / x.size
    Fy = np.searchsorted(y, grid, side='right') / y.size

    return float(np.max(np.abs(Fx - Fy)))


def get_dis_ssd_mean(df_obs, df_sim):
    """Absolute difference between mean SSDs."""
    # Filter for valid SSDs
    obs = df_obs[df_obs['ssd'] >= 0]['ssd']
    sim = df_sim[df_sim['ssd'] >= 0]['ssd']

    if obs.empty or sim.empty:
        return np.nan

    return abs(obs.mean() - sim.mean())


# --- 3. Descriptive Statistics (Analysis) ---

def get_stats_mean(df):
    """
    Compute summary stats for OBSERVED data (uses 'rt_real', 'ssd_real').
    """
    # Proportions
    percs = get_percentage(df)
    
    # Mean RTs (Observed data usually has 'rt_real' in ms)
    mrt_gs = df[df.result == "GS"].rt_real.mean()
    mrt_ge = df[df.result == "GE"].rt_real.mean()
    mrt_se = df[df.result == "SE"].rt_real.mean()
    
    # Stop Signal metrics
    mssd = df.ssd_real.mean() 
    ssrt = mrt_gs - mssd
    
    # Slopes (inhibition functions)
    rate_perc_ss = get_rate_perc_ss_ssd(df)
    rate_rt_se = get_rate_rt_se_ssd(df)
    
    return (*percs, mrt_gs, mrt_ge, mrt_se, ssrt, rate_perc_ss, rate_rt_se)


def get_stats_mean_sim(df):
    """
    Compute summary stats for SIMULATED data (uses 'rt' * 25).
    """
    percs = get_percentage(df)
    
    # Mean RTs (Simulated data usually has 'rt' in steps, needs scaling)
    mrt_gs = df[df.result == "GS"].rt.mean() * 25
    mrt_ge = df[df.result == "GE"].rt.mean() * 25
    mrt_se = df[df.result == "SE"].rt.mean() * 25
    
    mssd = df.ssd.mean() * 25
    ssrt = mrt_gs - mssd
    
    rate_perc_ss = get_rate_perc_ss_ssd(df)
    rate_rt_se = get_rate_rt_se_ssd(df)
    
    return (*percs, mrt_gs, mrt_ge, mrt_se, ssrt, rate_perc_ss, rate_rt_se)


# --- 4. Regression / Slope Helpers ---

def get_rate_perc_ss_ssd(df):
    """Slope of Stop Success Rate vs SSD."""
    try:
        df_stop = df[df['result'].isin(['SS', 'SE'])]
        if df_stop.empty: 
            return None
            
        # Group by SSD and calculate Success Rate
        counts = df_stop.groupby('ssd')['result'].value_counts().unstack(fill_value=0)
        
        # Ensure columns exist
        if 'SS' not in counts: counts['SS'] = 0
        if 'SE' not in counts: counts['SE'] = 0
        
        perc_ss = counts['SS'] / (counts['SS'] + counts['SE'])
        
        x = perc_ss.index.values
        y = perc_ss.values
        
        if len(x) < 2: 
            return None
            
        rate, _, _, _, _ = linregress(x, y)
        return rate
    except Exception:
        return None


def get_rate_rt_se_ssd(df):
    """Slope of Signal Respond RT vs SSD."""
    try:
        # Filter for Signal Errors (failed stops)
        df_se = df[df['result'] == 'SE']
        
        # Calculate mean RT per SSD
        means = df_se.groupby('ssd')['rt'].mean()
        
        x = means.index.values
        y = means.values
        
        if len(x) < 2:
            return None

        rate, _, _, _, _ = linregress(x, y)
        return rate
    except Exception:
        return None