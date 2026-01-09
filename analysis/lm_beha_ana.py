import pandas as pd
import statsmodels.formula.api as smf
import os
import numpy as np
from pathlib import Path

# ===== 1. Configuration & Paths =====

# Update these paths if necessary
OUT_DIR = Path('/kyb/agpd/wwang/sst-pomdp/outputs/')

# Inputs
STATS_CSV = OUT_DIR / "stats_beh_full.csv"

# Output
OUTPUT_TEX = OUT_DIR / "linear_model_behavior.tex"

DVS = ["mrt_gs", "ssrt", "perc_gs", "perc_ss"]

SUB_CAP = {
    "mrt_gs": "Go success reaction times",
    "ssrt":   "Stop signal reaction times",
    "perc_gs": "Go success rates",
    "perc_ss": "Stop success rates",
}

# Mapping statsmodels names to readable labels
PREDICTOR_MAPPING = {
    "Intercept": "Intercept",
    "adhd": "ADHD",
    "C(sex, Treatment('Female'))[T.Male]": "Sex (Male)",
    "iq": "IQ",
    "adhd_med_flag": "Medication",
}

# ===== 2. Helper Functions (Reference: lm_param.py) =====

def p_to_stars(p):
    if pd.isna(p): return ""
    if p < 0.001: return r"\textbf{***}"
    if p < 0.01:  return r"\textbf{**}"
    if p < 0.05:  return r"\textbf{*}"
    return ""

def fmt(beta, se, p, is_scientific=False):
    """Combines Coef, SE, and P-stars into one string."""
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

def fit_models(df):
    """Fits OLS models and returns a summary dictionary."""
    results = {}
    formula_template = "{dv} ~ adhd + C(sex, Treatment('Female')) + iq + adhd_med_flag"
    
    for dv in DVS:
        model = smf.ols(formula=formula_template.format(dv=dv), data=df)
        fit = model.fit()
        
        # Extract into a clean dictionary
        summary = {}
        for predictor, label in PREDICTOR_MAPPING.items():
            if predictor in fit.params:
                summary[label] = {
                    'beta': fit.params[predictor],
                    'se': fit.bse[predictor],
                    'p': fit.pvalues[predictor]
                }
        results[dv] = summary
    return results

# ===== 3. Table Construction =====

def build_latex_table(results_dict):
    """Generates a four-panel LaTeX table with caption at bottom."""
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    
    # Use tabularx or standard tabular based on preference; sticking to layout from lm_beha
    # but using the formatting style of lm_param.
    
    order = ["mrt_gs", "ssrt", "perc_gs", "perc_ss"]
    
    lines.append(r"\begin{minipage}{\linewidth}")
    
    # Helper for sub-panels
    def get_subtable(dv):
        sub_lines = []
        sub_lines.append(r"    \begin{subtable}[t]{0.48\linewidth}")
        sub_lines.append(r"    \centering")
        sub_lines.append(f"    \caption{{{SUB_CAP[dv]}}}")
        sub_lines.append(r"    \begin{tabular}{lc}")
        sub_lines.append(r"    \toprule")
        sub_lines.append(r"    \textbf{Predictor} & \textbf{Coef (SE)} \\")
        sub_lines.append(r"    \midrule")
        
        # Determine if we should use scientific notation (e.g., for percentages)
        use_sci = (dv == "perc_ss") 
        
        for _, label in PREDICTOR_MAPPING.items():
            res = results_dict[dv].get(label, {})
            formatted_val = fmt(res.get('beta'), res.get('se'), res.get('p'), use_sci)
            sub_lines.append(f"    {label} & {formatted_val} \\\\")
            
        sub_lines.append(r"    \bottomrule")
        sub_lines.append(r"    \end{tabular}")
        sub_lines.append(r"    \end{subtable}")
        return "\n".join(sub_lines)

    # Row 1
    lines.append(get_subtable(order[0]))
    lines.append(r"\hfill")
    lines.append(get_subtable(order[1]))
    lines.append("\n" + r"\vspace{0.3cm}" + "\n")
    # Row 2
    lines.append(get_subtable(order[2]))
    lines.append(r"\hfill")
    lines.append(get_subtable(order[3]))
    
    lines.append(r"\end{minipage}")
    
    # Bottom Section (Reference: lm_param.py)
    lines.append(r"\vspace{2mm}")
    lines.append(r"\raggedright")
    lines.append(r"\small")
    
    note = (
        r"\textbf{Linear model results for model-agnostic behavioral measures.} "
        r"Entries are fixed-effect coefficients with SEs in parentheses. "
        r"Stars denote p-values: \textbf{*} $p<.05$, \textbf{**} $p<.01$, \textbf{***} $p<.001$."
    )
    lines.append(rf"\caption{{{note}}}")
    lines.append(r"\label{tab:lm_beha_all}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)

# ===== 4. Execution =====

def main():
    if not os.path.exists(STATS_CSV):
        print(f"File not found: {STATS_CSV}")
        return

    # Load and clean data
    df = pd.read_csv(STATS_CSV)
    df = df[df['year'] == 'baseline'].dropna()
    print(f"Data Loaded. N={len(df)} subjects.")

    # Fit models
    results = fit_models(df)

    # Generate and Save LaTeX
    latex_code = build_latex_table(results)
    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(latex_code)
    
    print(f"Successfully saved LaTeX table to {OUTPUT_TEX}")

if __name__ == "__main__":
    main()