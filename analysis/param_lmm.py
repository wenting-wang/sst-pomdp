import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# --- Configuration & Paths ---

# Update these paths if necessary
BASE_DIR = Path('/kyb/agpd/wwang/data/')
OUT_DIR = Path('/kyb/agpd/wwang/sst-pomdp/outputs/')

# Inputs
PARAMS_CSV = OUT_DIR / "posterior_summary.csv"
STATS_CSV = OUT_DIR / "stats_beh_full.csv"

# Output
OUTPUT_TEX = OUT_DIR / "lmm_results.tex"

# Analysis Config
PARAMS_OF_INTEREST = ['q_d_n', 'q_d', 'q_s_n', 'q_s', 'cost_stop_error', 'cost_time']

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

    # 1. Load Parameters
    df_param = pd.read_csv(params_path)
    if 'subject_year' in df_param.columns:
        df_param = df_param.rename(columns={"subject_year": "year"})
    
    # Pivot
    df_pivot = df_param.pivot(
        index=['subject_id', 'year'], 
        columns='param', 
        values='mean'
    ).reset_index()
    df_pivot.columns.name = None

    # 2. Load Stats
    df_stats = pd.read_csv(stats_path)
    
    # 3. Merge
    df_merged = pd.merge(df_stats, df_pivot, on=['subject_id', 'year'], how='inner')
    
    # 4. Filter for Baseline
    df_clean = df_merged[df_merged['year'] == 'baseline'].dropna().copy()
    
    print(f"Data Loaded. N={len(df_clean)} subjects (Baseline).")
    return df_clean


def fit_linear_models(df, params):
    """
    Fits Multiple Linear Regression (OLS) models.
    """
    rows = []
    formula_template = "{target} ~ adhd + C(sex) + iq + adhd_med_flag"

    print(f"Fitting OLS models for: {params}...")

    for target_param in params:
        if target_param not in df.columns:
            print(f"Skipping {target_param}: not found in data.")
            continue
            
        dsub = df.dropna(subset=[target_param]).copy()
        if dsub.empty:
            continue
        
        try:
            # OLS Model
            model = smf.ols(
                formula=formula_template.format(target=target_param), 
                data=dsub
            )
            result = model.fit()

            # Helper
            def get_term(term):
                if term in result.params.index:
                    return (result.params[term], result.bse[term], result.pvalues[term])
                else:
                    return (np.nan, np.nan, np.nan)

            # Extract terms
            intercept, int_se, int_p = get_term('Intercept')
            adhd, adhd_se, adhd_p = get_term('adhd')
            sex, sex_se, sex_p = get_term('C(sex)[T.Male]')
            iq, iq_se, iq_p = get_term('iq')
            med, med_se, med_p = get_term('adhd_med_flag')

            rows.append({
                'param': target_param,
                'param_label': PARAM_DISPLAY_NAMES.get(target_param, target_param),
                'Intercept': intercept, 'Intercept_se': int_se, 'Intercept_p': int_p,
                'ADHD': adhd, 'ADHD_se': adhd_se, 'ADHD_p': adhd_p,
                'Sex': sex, 'Sex_se': sex_se, 'Sex_p': sex_p,
                'IQ': iq, 'IQ_se': iq_se, 'IQ_p': iq_p,
                'Medication': med, 'Medication_se': med_se, 'Medication_p': med_p,
            })
            
        except Exception as e:
            print(f"Error fitting model for {target_param}: {e}")

    return pd.DataFrame(rows)


def build_latex_table(summary_df):
    """Generates the LaTeX table string with the caption at the bottom."""
    
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

    # --- Table Construction ---
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    
    # Note: Caption removed from top, will be added at bottom
    
    lines.append(r"\begin{tabularx}{\linewidth}{lXXXXX}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Parameter} & \textbf{Intercept} & \textbf{ADHD} & \textbf{Sex (Male)} & \textbf{IQ} & \textbf{Medication} \\")
    lines.append(r"\midrule")

    for _, r in summary_df.iterrows():
        is_sci = (r['param'] == 'cost_time')
        
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
    
    # --- Modified End Section ---
    lines.append(r"\vspace{1mm}")
    lines.append(r"\raggedright")
    lines.append(r"\small")
    lines.append(r"\captionsetup{labelfont=bf}")
    
    caption_text = (
        r"\textbf{Multiple linear regression results of parameters with ADHD and covariates.} "
        r"Multiple linear regression models ($N=5,918$). Entries are fixed-effect coefficients with Wald SEs in parentheses. "
        r"Stars denote p-values: \textbf{*} $p<.05$, \textbf{**} $p<.01$, \textbf{***} $p<.001$. "
        r"ADHD was the main predictor, with Sex, IQ, and Medication included as covariates."
    )
    
    lines.append(rf"\caption{{{caption_text}}}")
    lines.append(r"\label{tab:lm_pomdp}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


# --- Main Execution ---

def main():
    try:
        df = load_and_prep_data(PARAMS_CSV, STATS_CSV)
        results_df = fit_linear_models(df, PARAMS_OF_INTEREST)
        
        if not results_df.empty:
            latex_code = build_latex_table(results_df)
            with open(OUTPUT_TEX, "w") as f:
                f.write(latex_code)
            print(f"\nSaved LaTeX table to: {OUTPUT_TEX}")
        else:
            print("No models were successfully fit.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()