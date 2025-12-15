import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

# --- Configuration & Paths ---

BASE_DIR = Path("/Users/w/Desktop/data")
WORK_DIR = Path("/Users/w/sst-abcd-adhd-mac/clean_version_within")
ARTIFACTS_DIR = Path("/Users/w/Desktop/artifacts_sbi_txf_823879")

# Inputs
PARAMS_CSV = ARTIFACTS_DIR / "posterior_summary_all.csv"
STATS_CSV = WORK_DIR / "stats_beh_full.csv"

# Output
OUTPUT_TEX = WORK_DIR / "lmm_results_table.tex"

# Analysis Config
# Parameters to analyze
PARAMS_OF_INTEREST = ['q_d_n', 'q_d', 'q_s_n', 'q_s', 'cost_stop_error', 'cost_time']

# Display names for LaTeX
PARAM_DISPLAY_NAMES = {
    'q_d_n': r"$\chi'$",
    'q_d': r"$\chi$",
    'q_s_n': r"$\sigma'$",
    'q_s': r"$\sigma$",
    'cost_stop_error': r"$c_{\mathrm{se}}$",
    'cost_time': r"$c_t$",
}


# --- Core Functions ---

def load_and_prep_data(params_path, stats_path):
    """
    Loads posterior parameters and demographic stats, merges them, 
    and filters for baseline data.
    """
    if not params_path.exists() or not stats_path.exists():
        raise FileNotFoundError("Input files not found.")

    # 1. Load Parameters (Long format -> Wide format)
    df_param = pd.read_csv(params_path)
    if 'subject_year' in df_param.columns:
        df_param = df_param.rename(columns={"subject_year": "year"})
    
    # Pivot: One row per subject/year, columns for each param
    df_pivot = df_param.pivot(
        index=['subject_id', 'year'], 
        columns='param', 
        values='mean'
    ).reset_index()
    df_pivot.columns.name = None

    # 2. Load Stats/Demographics
    df_stats = pd.read_csv(stats_path)
    
    # 3. Merge
    df_merged = pd.merge(df_stats, df_pivot, on=['subject_id', 'year'], how='inner')
    
    # 4. Filter for Baseline Only (as requested)
    df_clean = df_merged[df_merged['year'] == 'baseline'].dropna().copy()
    
    print(f"Data Loaded. N={len(df_clean)} subjects (Baseline).")
    return df_clean


def fit_mixed_models(df, params):
    """
    Fits Linear Mixed-Effects Models (random intercept by subject) for each parameter.
    """
    rows = []
    
    # Predictors: ADHD (main), Sex, IQ, Medication (covariates)
    formula_template = "{target} ~ adhd + C(sex) + iq + adhd_med_flag"

    print(f"Fitting MixedLM models for: {params}...")

    for target_param in params:
        if target_param not in df.columns:
            print(f"Skipping {target_param}: not found in data.")
            continue
            
        dsub = df.dropna(subset=[target_param]).copy()
        if dsub.empty:
            continue
        
        try:
            # MixedLM: random intercept per subject
            model = smf.mixedlm(
                formula=formula_template.format(target=target_param), 
                data=dsub, 
                groups=dsub["subject_id"], 
                re_formula="1"
            )
            
            # Fit model (using LBFGS for robustness)
            result = model.fit(method="lbfgs", reml=True, maxiter=200, disp=False)

            # Helper to safely extract terms
            def get_term(term):
                if term in result.params.index:
                    return (
                        result.params[term],
                        result.bse[term],
                        result.pvalues[term]
                    )
                else:
                    return (np.nan, np.nan, np.nan)

            # Extract Coefficients
            intercept, int_se, int_p = get_term('Intercept')
            adhd, adhd_se, adhd_p = get_term('adhd')
            sex, sex_se, sex_p = get_term('C(sex)[T.Male]')
            iq, iq_se, iq_p = get_term('iq')
            med, med_se, med_p = get_term('adhd_med_flag')

            rows.append({
                'param': target_param,
                'param_label': PARAM_DISPLAY_NAMES.get(target_param, target_param),
                # Intercept
                'Intercept': intercept, 'Intercept_se': int_se, 'Intercept_p': int_p,
                # ADHD
                'ADHD': adhd, 'ADHD_se': adhd_se, 'ADHD_p': adhd_p,
                # Sex
                'Sex': sex, 'Sex_se': sex_se, 'Sex_p': sex_p,
                # IQ
                'IQ': iq, 'IQ_se': iq_se, 'IQ_p': iq_p,
                # Medication
                'Medication': med, 'Medication_se': med_se, 'Medication_p': med_p,
                # Diagnostics
                'N_obs': int(result.nobs),
                'N_subjects': int(dsub['subject_id'].nunique()),
                'Var_intercept': float(result.cov_re.iloc[0,0]) if result.cov_re is not None else np.nan,
            })
            
        except Exception as e:
            print(f"Error fitting model for {target_param}: {e}")

    return pd.DataFrame(rows)


def build_latex_table(summary_df):
    """Generates the LaTeX table string from the results dataframe."""
    
    def p_to_stars(p):
        if pd.isna(p): return ""
        if p < 0.001: return r"\textbf{***}"
        if p < 0.01:  return r"\textbf{**}"
        if p < 0.05:  return r"\textbf{*}"
        return ""

    def fmt(beta, se, p, is_scientific=False):
        if pd.isna(beta) or pd.isna(se):
            return ""
        
        if is_scientific:
            val_str = f"{beta:.2e}"
            se_str = f"{se:.2e}"
        else:
            val_str = f"{beta:.4f}"
            se_str = f"{se:.4f}"
            
        stars = p_to_stars(p)
        return f"{val_str}{stars} ({se_str})"

    caption = "Linear mixed-effects results (ADHD main effect; random intercepts by participant; controlling sex, IQ, and medication)"
    label = "tab:lmm_pomdp_full"

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    # Columns: Param | Intercept | ADHD | Sex(Male) | IQ | Medication
    lines.append(r"\begin{tabularx}{\linewidth}{lXXXXX}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Parameter} & \textbf{Intercept} & \textbf{ADHD} & \textbf{Sex (Male)} & \textbf{IQ} & \textbf{Medication} \\")
    lines.append(r"\midrule")

    for _, r in summary_df.iterrows():
        is_sci = (r['param'] == 'cost_time') # Scientific notation for small parameter
        
        row = (
            f"{r['param_label']} & "
            f"{fmt(r['Intercept'], r['Intercept_se'], r['Intercept_p'], is_sci)} & "
            f"{fmt(r['ADHD'], r['ADHD_se'], r['ADHD_p'], is_sci)} & "
            f"{fmt(r['Sex'], r['Sex_se'], r['Sex_p'], is_sci)} & "
            f"{fmt(r['IQ'], r['IQ_se'], r['IQ_p'], is_sci)} & "
            f"{fmt(r['Medication'], r['Medication_se'], r['Medication_p'], is_sci)} \\\\"
        )
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\vspace{1mm}")
    lines.append(r"\raggedright")
    lines.append(r"\small")
    lines.append(
        r"\textbf{Note.} Linear mixed-effects models with random intercepts by participant. "
        r"Entries are fixed-effect coefficients with Wald SEs in parentheses. "
        r"Stars mark p-values: \textbf{*} $p<.05$, \textbf{**} $p<.01$, \textbf{***} $p<.001$. "
        r"ADHD is the main predictor; Sex, IQ, and Medication are covariates."
    )
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


# --- Main Execution ---

def main():
    try:
        # 1. Load Data
        df = load_and_prep_data(PARAMS_CSV, STATS_CSV)
        
        # 2. Fit MixedLM Models
        results_df = fit_mixed_models(df, PARAMS_OF_INTEREST)
        
        # 3. Generate LaTeX
        if not results_df.empty:
            latex_code = build_latex_table(results_df)
            
            print("\n--- LaTeX Table Output ---\n")
            print(latex_code)
            
            with open(OUTPUT_TEX, "w") as f:
                f.write(latex_code)
            print(f"\nSaved LaTeX table to: {OUTPUT_TEX}")
        else:
            print("No models were successfully fit.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()