import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import traceback
import warnings
import argparse

warnings.filterwarnings("ignore")

# =============================================================================
# 0. ARGUMENT PARSING (DEMO vs PROD MODE)
# =============================================================================
parser = argparse.ArgumentParser(description="Generate Linear Models Table for Parameters")
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
    OUT_DIR = PROJECT_ROOT / "outputs"
    PARAMS_CSV = PROJECT_ROOT / "data" / "example_params_posteriors.csv"
    STATS_CSV = PROJECT_ROOT / "data" / "example_clinical_behavior.csv"
else:
    print(">>> RUNNING IN PROD MODE (Using local raw data) <<<")
    OUT_DIR = PROJECT_ROOT / "outputs"
    PARAMS_CSV = Path("/Users/w/sst-hdbm-mac/sst-pomdp-main/data/params_posteriors_6p_v1.csv")
    STATS_CSV = Path("/Users/w/sst-hdbm-mac/sst-pomdp-main/data/clinical_behavior.csv")

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MAIN_TEX = OUT_DIR / "tab_lm_params_main.tex"
OUTPUT_SUPP_TEX = OUT_DIR / "tab_lm_params_supp.tex"

# =============================================================================
# 2. MODEL CONFIGURATIONS
# =============================================================================
PARAMS_OF_INTEREST = [
    'q_d', 'q_s', 'cost_stop_error', 'cost_go_error', 'cost_time', 'tau'
]

PARAM_DISPLAY_NAMES = {
    'q_d': r"$\chi$",
    'q_s': r"$\delta$",
    'cost_stop_error': r"$c_{\text{se}}$",
    'cost_go_error': r"$c_{\text{ge}}$",
    'cost_time': r"$c_{\text{t}}$",
    'tau': r"$\tau$"
}

# --- Define building blocks for Candidate Models ---
BASE_RHS = "C(sex, Treatment('Female')) + iq + adhd_med_flag"
MAIN_RHS = f"adhd + {BASE_RHS}"

INT_SEX_RHS = f"{MAIN_RHS} + adhd:C(sex, Treatment('Female'))"
INT_IQ_RHS  = f"{MAIN_RHS} + adhd:iq"
INT_MED_RHS = f"{MAIN_RHS} + adhd:adhd_med_flag"

FULL_RHS = f"{MAIN_RHS} + adhd:C(sex, Treatment('Female')) + adhd:iq + adhd:adhd_med_flag"

CANDIDATE_MODELS = {
    'Main': MAIN_RHS,
    '+Int_Sex': INT_SEX_RHS,
    '+Int_IQ': INT_IQ_RHS,
    '+Int_Med': INT_MED_RHS,
    'Full': FULL_RHS        
}

PREDICTORS_FORMULA = {
    'ADHD': 'adhd',
    'Sex': "C(sex, Treatment('Female'))",
    'IQ': 'iq',
    'Medication': 'adhd_med_flag',
    'ADHD_x_Sex': r"adhd:C(sex, Treatment('Female'))",
    'ADHD_x_IQ': r"adhd:iq",
    'ADHD_x_Med': r"adhd:adhd_med_flag"
}

PREDICTORS_EXTRACT = {
    'ADHD': 'adhd',
    'Sex': "C(sex, Treatment('Female'))[T.Male]",
    'IQ': 'iq',
    'Medication': 'adhd_med_flag',
    'ADHD_x_Sex': "adhd:C(sex, Treatment('Female'))[T.Male]",
    'ADHD_x_IQ': "adhd:iq",
    'ADHD_x_Med': "adhd:adhd_med_flag"
}

# --- Core Functions ---
def load_and_prep_data(params_path, stats_path, params_to_keep):
    try:
        df_param = pd.read_csv(params_path)
        df_stats = pd.read_csv(stats_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing file: {e.filename}")

    if 'subject_year' in df_param.columns:
        df_param = df_param.rename(columns={"subject_year": "year"})
    
    df_param['year'] = df_param['year'].astype(str)
    df_pivot = df_param.pivot_table(index=['subject_id', 'year'], columns='index', values='mean').reset_index()
    df_pivot.columns.name = None
    
    df_stats['year'] = df_stats['year'].astype(str)
    df_pivot['subject_id'] = df_pivot['subject_id'].str.replace('NDAR_', '', regex=False)
    
    df_merged = pd.merge(df_stats, df_pivot, on=['subject_id', 'year'], how='inner')
    df_baseline = df_merged[df_merged['year'] == 'baseline'].copy()
    
    req_cols = ['adhd', 'sex', 'iq', 'adhd_med_flag'] + [p for p in params_to_keep if p in df_baseline.columns]
    avail_cols = [c for c in req_cols if c in df_baseline.columns]
    df_clean = df_baseline.dropna(subset=avail_cols).copy()
    
    continuous_vars = ['adhd', 'iq'] + [p for p in params_to_keep if p in df_clean.columns]
    for col in continuous_vars:
        if df_clean[col].std() > 0:
            df_clean[col] = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
            
    return df_clean

def fit_all_effect_sizes(df, params):
    main_rows = []
    supp_rows = []

    for target_param in params:
        if target_param not in df.columns: continue
        dsub = df.dropna(subset=[target_param]).copy()
        if dsub.empty: continue
        
        try:
            models_fit = {}
            bics = {}
            dfs = {}
            for m_name, rhs in CANDIDATE_MODELS.items():
                m_fit = smf.ols(formula=f"{target_param} ~ {rhs}", data=dsub).fit()
                models_fit[m_name] = m_fit
                bics[m_name] = m_fit.bic
                dfs[m_name] = int(m_fit.df_model) 
                
            best_model_name = min(bics, key=bics.get)
            best_model = models_fit[best_model_name]
            best_rhs = CANDIDATE_MODELS[best_model_name]
            r2_best = best_model.rsquared
            min_bic = bics[best_model_name]
            
            for m_name in CANDIDATE_MODELS.keys():
                supp_rows.append({
                    'param_label': PARAM_DISPLAY_NAMES.get(target_param, target_param),
                    'model': m_name,
                    'df': dfs[m_name],
                    'bic': bics[m_name],
                    'delta_bic': bics[m_name] - min_bic
                })
            
            row_data = {
                'param': target_param, 
                'param_label': PARAM_DISPLAY_NAMES.get(target_param, target_param),
                'best_model': best_model_name
            }
            
            best_terms_list = best_rhs.split(' + ')
            
            for pred_name, formula_term in PREDICTORS_FORMULA.items():
                extract_term = PREDICTORS_EXTRACT[pred_name]
                
                if extract_term not in best_model.params.index and ':' in extract_term:
                    parts = extract_term.split(':')
                    alt_term = f"{parts[1]}:{parts[0]}"
                    if alt_term in best_model.params.index:
                        extract_term = alt_term
                
                if extract_term in best_model.params.index:
                    beta = best_model.params[extract_term]
                    se = best_model.bse[extract_term]
                    p_val = best_model.pvalues[extract_term]
                    
                    if formula_term in best_terms_list:
                        reduced_terms = [t for t in best_terms_list if t != formula_term]
                        reduced_rhs = " + ".join(reduced_terms) if reduced_terms else "1"
                        model_reduced = smf.ols(formula=f"{target_param} ~ {reduced_rhs}", data=dsub).fit()
                        r2_reduced = model_reduced.rsquared
                        
                        partial_r2 = max(0.0, (r2_best - r2_reduced) / (1 - r2_reduced))
                        cohens_f2 = partial_r2 / max(1 - partial_r2, 1e-10)
                    else:
                        partial_r2, cohens_f2 = np.nan, np.nan
                else:
                    beta, se, p_val, partial_r2, cohens_f2 = np.nan, np.nan, np.nan, np.nan, np.nan
                
                row_data[f'{pred_name}_beta'] = beta
                row_data[f'{pred_name}_se'] = se
                row_data[f'{pred_name}_p'] = p_val
                row_data[f'{pred_name}_pr2'] = partial_r2
                row_data[f'{pred_name}_f2'] = cohens_f2
                
            main_rows.append(row_data)
        except Exception as e:
            print(f"Error on {target_param}: {e}")
            
    return pd.DataFrame(main_rows), pd.DataFrame(supp_rows)

def build_main_latex_table(df_res, n_subjects):
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
        r"        \caption{\textbf{Parameter Regression Coefficients and Effect Sizes}}",
        r"        \label{tab:param_regression_main} \\",
        r"        \toprule",
        r"        \textbf{Parameters} & ",
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
        r"        \textbf{Parameters} & ",
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
        rf"       \multicolumn{{6}}{{l}}{{\footnotesize \rule{{0pt}}{{3ex}}Note: $N={n_subjects:,}$. $^* p<.05$, $^{{**}} p<.01$, $^{{***}} p<.001$.}} \\",
        r"        \endlastfoot"   
    ]

    predictor_groups = [
        ('ADHD', 'ADHD'),
        ('Sex', 'Sex (Male)'),
        ('IQ', 'IQ'),
        ('Medication', 'Medication'),
        ('ADHD_x_Sex', r'ADHD $\times$ Sex'),
        ('ADHD_x_IQ', r'ADHD $\times$ IQ'),
        ('ADHD_x_Med', r'ADHD $\times$ Medication')
    ]

    first_group = True
    for pred_key, pred_label in predictor_groups:
        has_data = any(not pd.isna(r[f'{pred_key}_beta']) for _, r in df_res.iterrows())
        if not has_data: continue

        if not first_group:
            lines.append(r"        \midrule")
        first_group = False
        
        lines.append(rf"        \textbf{{{pred_label}}} & & & & & \\")
        
        for _, r in df_res.iterrows():
            if pd.isna(r[f'{pred_key}_beta']): continue 
            lines.append(
                rf"        \hspace{{4mm}} {r['param_label']} & "
                rf"{fmt_num(r[f'{pred_key}_beta'])} & {fmt_num(r[f'{pred_key}_se'])} & "
                rf"{p_fmt(r[f'{pred_key}_p'])} & {fmt_num(r[f'{pred_key}_pr2'])} & {fmt_num(r[f'{pred_key}_f2'])} \\"
            )
        
    lines.extend([r"    \end{longtable}", r"\endgroup"])
    return "\n".join(lines)


def build_supp_latex_table(df_supp):
    lines = [
        r"\begingroup",
        r"    \renewcommand{\arraystretch}{1.0}",
        r"    \footnotesize",
        r"    \begin{longtable}{l l c r r}",
        r"        \caption{\textbf{Supplementary: BIC Model Comparisons}}",
        r"        \label{tab:bic_comparison} \\",
        r"        \toprule",
        r"        \textbf{Parameter} & \textbf{Model} & \textbf{$df$} & \textbf{BIC} & \textbf{$\Delta$BIC} \\",
        r"        \midrule",
        r"        \endfirsthead",
        r"        \toprule",
        r"        \textbf{Parameter} & \textbf{Model} & \textbf{$df$} & \textbf{BIC} & \textbf{$\Delta$BIC} \\",
        r"        \midrule",
        r"        \endhead",      
        r"        \midrule",
        r"        \multicolumn{5}{r}{{Continued on next page}} \\",
        r"        \endfoot",      
        r"        \bottomrule",
        r"        \multicolumn{5}{l}{\footnotesize \rule{0pt}{3ex}Note: $\Delta$BIC = 0 indicates the preferred model.} \\",
        r"        \endlastfoot"   
    ]

    current_param = None
    for _, r in df_supp.iterrows():
        param = r['param_label']
        if param != current_param:
            if current_param is not None:
                lines.append(r"        \midrule")
            current_param = param
            param_display = param
        else:
            param_display = ""
            
        model_name = r['model']
        if model_name == '+Int_Sex': model_name = r'+ ADHD $\times$ Sex'
        elif model_name == '+Int_IQ': model_name = r'+ ADHD $\times$ IQ'
        elif model_name == '+Int_Med': model_name = r'+ ADHD $\times$ Med'

        if r['delta_bic'] == 0:
            model_name = f"\\textbf{{{model_name}}}"
            
        lines.append(
            rf"        {param_display} & {model_name} & {r['df']} & "
            rf"{r['bic']:.1f} & {r['delta_bic']:.1f} \\"
        )

    lines.extend([r"    \end{longtable}", r"\endgroup"])
    return "\n".join(lines)


# --- Main ---
def main():
    def fmt_print(val):
        if pd.isna(val): return "     ---"
        s = f"{val:8.4f}"
        if s.strip() == "0.0000" or s.strip() == "-0.0000": return "<0.0001".rjust(8)
        return s

    try:
        df = load_and_prep_data(PARAMS_CSV, STATS_CSV, PARAMS_OF_INTEREST)
        n_sub = len(df)
        if df.empty: return
        
        df_main, df_supp = fit_all_effect_sizes(df, PARAMS_OF_INTEREST)
        
        print("\n" + "="*65)
        print(" [SUPPLEMENTARY] BIC MODEL COMPARISON (STEPWISE)")
        print("="*65)
        print(f"{'Parameter':<15} | {'Model':<15} | {'df':>2} | {'BIC':>8} | {'Delta BIC':>9}")
        print("-" * 65)
        
        current_param = None
        for _, r in df_supp.iterrows():
            param_str = r['param_label'] if r['param_label'] != current_param else ""
            current_param = r['param_label']
            
            model_str_console = r['model']
            if model_str_console == '+Int_Sex': model_str_console = '+ ADHD*Sex'
            elif model_str_console == '+Int_IQ': model_str_console = '+ ADHD*IQ'
            elif model_str_console == '+Int_Med': model_str_console = '+ ADHD*Med'

            model_str = f"*{model_str_console}*" if r['delta_bic'] == 0 else model_str_console
            print(f"{param_str:<15} | {model_str:<15} | {r['df']:>2} | {r['bic']:8.1f} | {r['delta_bic']:9.1f}")


        print("\n" + "="*85)
        print(" [MAIN RESULTS] FINAL EFFECT SIZES (MAIN EFFECTS ONLY)")
        print("="*85)
        
        predictors_to_print = ['ADHD', 'Sex', 'IQ', 'Medication', 'ADHD_x_Sex', 'ADHD_x_IQ', 'ADHD_x_Med']
        
        for pred in predictors_to_print:
            has_data = any(not pd.isna(r[f'{pred}_beta']) for _, r in df_main.iterrows())
            if not has_data: continue

            print(f"\n--- {pred.replace('_', ' ').upper()} ---")
            print(f"{'Param':<15} | {'Beta':>8} | {'SE':>8} | {'p-value':>8} | {'Part_R2':>8} | {'f2':>8}")
            print("-" * 85)
            for _, r in df_main.iterrows():
                if pd.isna(r[f'{pred}_beta']): continue
                print(f"{r['param_label']:<15} | "
                      f"{fmt_print(r[f'{pred}_beta'])} | {fmt_print(r[f'{pred}_se'])} | "
                      f"{fmt_print(r[f'{pred}_p'])} | {fmt_print(r[f'{pred}_pr2'])} | {fmt_print(r[f'{pred}_f2'])}")
        
        print("\n" + "="*85)

        with open(OUTPUT_MAIN_TEX, "w") as f:
            f.write(build_main_latex_table(df_main, n_sub))
        print(f"\nSaved Main Text Table to: {OUTPUT_MAIN_TEX}")
        
        with open(OUTPUT_SUPP_TEX, "w") as f:
            f.write(build_supp_latex_table(df_supp))
        print(f"Saved Supplementary Table to: {OUTPUT_SUPP_TEX}")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()