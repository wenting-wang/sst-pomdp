import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from pathlib import Path
import warnings
import argparse

warnings.filterwarnings("ignore")

# =============================================================================
# 0. ARGUMENT PARSING (DEMO vs PROD MODE)
# =============================================================================
parser = argparse.ArgumentParser(description="Generate Linear Models Table for Behavior")
parser.add_argument(
    '--prod', 
    action='store_true', 
    help='Run in PROD mode using local raw data. Usage: run `python script.py --prod`. Defaults to DEMO mode using example data if omitted.'
)
args, unknown = parser.parse_known_args()

DEMO_MODE = not args.prod

# =============================================================================
# 1. CONFIGURATION & PATHS
# =============================================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if DEMO_MODE:
    print(">>> RUNNING IN DEMO MODE (Using example data) <<<")
    OUT_DIR = PROJECT_ROOT / "outputs_example"
    STATS_CSV = PROJECT_ROOT / "data" / "example_clinical_behavior.csv"
else:
    print(">>> RUNNING IN PROD MODE (Using local raw data) <<<")
    OUT_DIR = PROJECT_ROOT / "outputs"
    STATS_CSV = Path("/Users/w/sst-pomdp/data/clinical_behavior.csv")

# Ensure output directory exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Output
OUTPUT_TEX = OUT_DIR / "tab_lm_behavior.tex"

# =============================================================================
# 2. MODEL CONFIGURATIONS
# =============================================================================
DVS = ["mrt_gs", "ssrt", "perc_gs", "perc_ss"]

DV_LABELS = {
    "mrt_gs": "GSRT",
    "ssrt":   "SSRT",
    "perc_gs": "GS rate",
    "perc_ss": "SS rate",
}

PREDICTORS = {
    'ADHD': {
        'term': 'adhd', 
        'extract': 'adhd', 
        'label': 'ADHD'
    },
    'Sex': {
        'term': "C(sex, Treatment('Female'))", 
        'extract': "C(sex, Treatment('Female'))[T.Male]", 
        'label': 'Sex (Male)'
    },
    'IQ': {
        'term': 'iq', 
        'extract': 'iq', 
        'label': 'IQ'
    },
    'Medication': {
        'term': 'adhd_med_flag', 
        'extract': 'adhd_med_flag', 
        'label': 'Medication'
    }
}

# ===== 2. Statistical Analysis =====

def fit_models(df):
    """Fits OLS models, calculates effect sizes, and returns structured results."""
    results = []
    
    # Formula for the full model
    terms_list = [p['term'] for p in PREDICTORS.values()]
    full_rhs = " + ".join(terms_list)
    
    for dv in DVS:
        if dv not in df.columns:
            print(f"Warning: Dependent variable '{dv}' not found in data. Skipping.")
            continue
            
        # Fit full model
        full_formula = f"{dv} ~ {full_rhs}"
        full_model = smf.ols(formula=full_formula, data=df).fit()
        r2_full = full_model.rsquared
        
        row_data = {
            'dv': dv,
            'dv_label': DV_LABELS.get(dv, dv)
        }
        
        # Iterate over predictors to extract stats and calculate effect sizes
        for pred_key, p_info in PREDICTORS.items():
            extract_term = p_info['extract']
            formula_term = p_info['term']
            
            if extract_term in full_model.params.index:
                beta = full_model.params[extract_term]
                se = full_model.bse[extract_term]
                p_val = full_model.pvalues[extract_term]
                
                # Fit reduced model to get Partial R2
                reduced_terms = [t for t in terms_list if t != formula_term]
                reduced_rhs = " + ".join(reduced_terms) if reduced_terms else "1"
                reduced_model = smf.ols(formula=f"{dv} ~ {reduced_rhs}", data=df).fit()
                r2_reduced = reduced_model.rsquared
                
                # Calculate Partial R2 and Cohen's f2
                partial_r2 = max(0.0, (r2_full - r2_reduced) / (1 - r2_reduced))
                cohens_f2 = partial_r2 / max(1 - partial_r2, 1e-10)
                
            else:
                beta, se, p_val, partial_r2, cohens_f2 = np.nan, np.nan, np.nan, np.nan, np.nan
                
            row_data[f'{pred_key}_beta'] = beta
            row_data[f'{pred_key}_se'] = se
            row_data[f'{pred_key}_p'] = p_val
            row_data[f'{pred_key}_pr2'] = partial_r2
            row_data[f'{pred_key}_f2'] = cohens_f2
            
        results.append(row_data)
        
    return pd.DataFrame(results)

# ===== 3. Table Construction =====

def build_latex_table(df_res, n_subjects):
    """Generates a grouped longtable for behavioral measures matching the params table format."""
    def fmt_num(val):
        if pd.isna(val): return "---"
        res = f"{val:.4f}"
        if res == "0.0000" or res == "-0.0000": return "$<0.0001$"
        return res

    def p_fmt(p):
        if pd.isna(p): return "---"
        if p < 0.0001: base = "$<0.0001$"
        else: base = f"{p:.4f}"
        if p < 0.001: return base + r"$^{***}$"
        elif p < 0.01: return base + r"$^{**}$"
        elif p < 0.05: return base + r"$^{*}$"
        else: return base

    lines = [
        r"\begingroup",
        r"    \renewcommand{\arraystretch}{1.0}",
        r"    \footnotesize",
        r"    \begin{longtable}{l r r r r r}",
        r"        \captionsetup{labelfont=bf}",
        r"        \caption{\textbf{Linear model results for model-agnostic behavioral metrics}}",
        r"        \label{tab:lm_behavior_main} \\",
        r"        \toprule",
        r"        \textbf{Behavioral metrics} & ",
        r"        \multicolumn{1}{c}{\textbf{Coef.}} & ",
        r"        \multicolumn{1}{c}{\textbf{SE}} & ",
        r"        \multicolumn{1}{c}{\textbf{$p$-value}} & ",
        r"        \multicolumn{1}{c}{\textbf{Partial $R^2$}} & ",
        r"        \multicolumn{1}{c}{\textbf{$f^2$}} \\",
        r"        \midrule",
        r"        \endfirsthead",
        r"",
        r"        \multicolumn{6}{c}{{\tablename\ \thetable{} -- continued from previous page}} \\",
        r"        \toprule",
        r"        \textbf{Behavioral metrics} & ",
        r"        \multicolumn{1}{c}{\textbf{Coef.}} & ",
        r"        \multicolumn{1}{c}{\textbf{SE}} & ",
        r"        \multicolumn{1}{c}{\textbf{$p$-value}} & ",
        r"        \multicolumn{1}{c}{\textbf{Partial $R^2$}} & ",
        r"        \multicolumn{1}{c}{\textbf{$f^2$}} \\",
        r"        \midrule",
        r"        \endhead",      
        r"",
        r"        \midrule",
        r"        \multicolumn{6}{r}{{Continued on next page}} \\",
        r"        \endfoot",      
        r"",
        r"        \bottomrule",
        rf"        \multicolumn{{6}}{{l}}{{\footnotesize \rule{{0pt}}{{3ex}}Note: $N={n_subjects:,}$. $^* p<.05$, $^{{**}} p<.01$, $^{{***}} p<.001$.}} \\",        r"        \endlastfoot"   
    ]

    first_group = True
    for pred_key, p_info in PREDICTORS.items():
        pred_label = p_info['label']
        
        # Check if there's any data for this predictor
        has_data = any(not pd.isna(r[f'{pred_key}_beta']) for _, r in df_res.iterrows())
        if not has_data: continue

        if not first_group:
            lines.append(r"        \midrule")
        first_group = False
        
        lines.append(rf"        \textbf{{{pred_label}}} & & & & & \\")
        
        for _, r in df_res.iterrows():
            if pd.isna(r[f'{pred_key}_beta']): continue 
            lines.append(
                rf"        \hspace{{4mm}} {r['dv_label']} & "
                rf"{fmt_num(r[f'{pred_key}_beta'])} & {fmt_num(r[f'{pred_key}_se'])} & "
                rf"{p_fmt(r[f'{pred_key}_p'])} & {fmt_num(r[f'{pred_key}_pr2'])} & {fmt_num(r[f'{pred_key}_f2'])} \\"
            )
        
    lines.extend([r"    \end{longtable}", r"\endgroup"])
    return "\n".join(lines)

# ===== 4. Execution =====

def main():
    def fmt_print(val):
        if pd.isna(val): return "     ---"
        s = f"{val:8.4f}"
        if s.strip() == "0.0000" or s.strip() == "-0.0000": return "<0.0001".rjust(8)
        return s

    try:
        if not STATS_CSV.exists():
            raise FileNotFoundError(
                f"Data file not found at {STATS_CSV}. "
                "Please place your dataset in the 'data' folder or update the STATS_CSV path."
            )

        # Load data
        df = pd.read_csv(STATS_CSV)
        
        # Filter for baseline
        if 'year' in df.columns:
            df = df[df['year'].astype(str) == 'baseline']
            
        req_cols = DVS + [p['extract'] for p in PREDICTORS.values() if ':' not in p['extract'] and '[' not in p['extract']]
        req_cols = DVS + ['adhd', 'sex', 'iq', 'adhd_med_flag'] # Manual list to handle categorical strings
        
        available_cols = [col for col in req_cols if col in df.columns]
        df_clean = df[available_cols].dropna().copy()
        
        n_subjects = len(df_clean)
        print(f"Data Loaded and Cleaned. N={n_subjects} valid subjects.")

        if df_clean.empty:
            print("Warning: Dataset is empty after filtering for 'baseline'. Table will not be generated.")
            return

        # Fit models and extract results
        df_results = fit_models(df_clean)

        if not df_results.empty:
            # Print to console for quick review
            print("\n" + "="*85)
            print(" [MAIN RESULTS] BEHAVIORAL EFFECT SIZES")
            print("="*85)
            
            for pred_key, p_info in PREDICTORS.items():
                print(f"\n--- {p_info['label'].upper()} ---")
                print(f"{'Measure':<28} | {'Beta':>8} | {'SE':>8} | {'p-value':>8} | {'Part_R2':>8} | {'f2':>8}")
                print("-" * 85)
                for _, r in df_results.iterrows():
                    if pd.isna(r[f'{pred_key}_beta']): continue
                    print(f"{r['dv_label']:<28} | "
                          f"{fmt_print(r[f'{pred_key}_beta'])} | {fmt_print(r[f'{pred_key}_se'])} | "
                          f"{fmt_print(r[f'{pred_key}_p'])} | {fmt_print(r[f'{pred_key}_pr2'])} | {fmt_print(r[f'{pred_key}_f2'])}")
            print("\n" + "="*85)

            # Generate and Save LaTeX
            latex_code = build_latex_table(df_results, n_subjects)
            with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
                f.write(latex_code)
            print(f"\nSuccessfully saved LaTeX table to {OUTPUT_TEX}")
        else:
            print("No models were successfully fitted.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()