import pandas as pd
import statsmodels.formula.api as smf
import os
import numpy as np
from pathlib import Path

# ===== 1. Configuration & Paths =====

# Update these paths if necessary
# OUT_DIR = Path('/kyb/agpd/wwang/sst-pomdp/outputs/')
OUT_DIR = Path('/Users/w/sst-abcd-adhd-mac/sst-pomdp-main/outputs/')


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

# ===== 2. Helper Functions =====

def format_num(x):
    """
    Formats Coef, SE, t:
    1. Default to 3 decimal places.
    2. If result is '0.000' or '-0.000', use 4 decimal places.
    """
    if pd.isna(x):
        return ""
    
    s = f"{x:.3f}"
    # If the result is visually zero, switch to 4 decimal places
    if s == "0.000" or s == "-0.000":
        return f"{x:.4f}"
    return s

def format_p(p):
    """
    Formats p-value:
    1. If p < 0.001, return $<0.001$$^{***}$
    2. Otherwise, return 3 decimals with stars as superscript if significant.
    """
    if pd.isna(p):
        return ""
    
    # Determine stars
    stars = ""
    if p < 0.001: stars = "***"
    elif p < 0.01: stars = "**"
    elif p < 0.05: stars = "*"
    
    # Formatting
    if p < 0.001:
        # User requested specific format for very small p-values
        return r"$<0.001$$^{***}$"
    else:
        # Standard 3 decimal formatting for larger p-values
        s = f"{p:.3f}"
        if stars:
            return f"{s}$^{{{stars}}}$"
        return s

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
                    't': fit.tvalues[predictor],
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
    
    order = ["mrt_gs", "ssrt", "perc_gs", "perc_ss"]
    
    lines.append(r"\begin{minipage}{\linewidth}")
    
    # Helper for sub-panels
    def get_subtable(dv):
        sub_lines = []
        sub_lines.append(r"    \begin{subtable}[t]{0.48\linewidth}")
        sub_lines.append(r"    \centering")
        sub_lines.append(f"    \caption{{{SUB_CAP[dv]}}}")
        
        # 5 columns: Predictor, Coef, SE, t, p
        sub_lines.append(r"    \begin{tabular}{lcccc}")
        sub_lines.append(r"    \toprule")
        sub_lines.append(r"    \textbf{Predictor} & \textbf{Coef.} & \textbf{SE} & \textbf{t} & \textbf{p} \\")
        sub_lines.append(r"    \midrule")
        
        for _, label in PREDICTOR_MAPPING.items():
            res = results_dict[dv].get(label, {})
            
            beta_str = format_num(res.get('beta'))
            se_str   = format_num(res.get('se'))
            t_str    = format_num(res.get('t'))
            # Use specific p-value formatter
            p_str    = format_p(res.get('p'))
            
            sub_lines.append(f"    {label} & {beta_str} & {se_str} & {t_str} & {p_str} \\\\")
            
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
    
    # Bottom Section
    lines.append(r"\vspace{2mm}")
    lines.append(r"\raggedright")
    lines.append(r"\small")
    
    note = (
        r"\textbf{Linear model results for model-agnostic behavioral measures.} "
        r"Table shows fixed-effect coefficients (Coef.), standard errors (SE), t-values (t), and p-values (p). "
        r"Stars denote significance levels: \textbf{*} $p<.05$, \textbf{**} $p<.01$, \textbf{***} $p<.001$."
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