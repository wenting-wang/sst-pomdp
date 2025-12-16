import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterable, Optional, Union, List, Tuple

# --- Configuration & Paths ---

BASE_DIR = Path('/kyb/agpd/wwang/data/')
OUT_DIR = Path('/kyb/agpd/wwang/sst-pomdp/outputs/')

# Inputs (This file already contains merged demographics and stats)
DATA_CSV = OUT_DIR / "stats_beh_full.csv"

# Output
OUTPUT_TEX = OUT_DIR / "counts_demo.tex"


# --- Helper Functions ---

def normalize_group_column(series: pd.Series, kind: str) -> pd.Series:
    """
    Standardizes categorical values for display in the LaTeX table.
    - Maps 0/1 medication flags to 'Unmedicated'/'Medicated'.
    - Ensures Sex is standardized (though input is likely already clean).
    """
    s = series.copy()
    
    if kind == "sex":
        # Input is likely already 'Male'/'Female', but this ensures consistency
        sex_map = {
            "F": "Female", "Female": "Female", "female": "Female", "f": "Female",
            "M": "Male", "Male": "Male", "male": "Male", "m": "Male"
        }
        return s.map(lambda x: sex_map.get(str(x), str(x)))
    
    if kind == "med":
        # Map binary flags (0/1) to readable labels
        med_map = {
            0: "Unmedicated", 1: "Medicated", "0": "Unmedicated", "1": "Medicated",
            "no": "Unmedicated", "yes": "Medicated", 
            False: "Unmedicated", True: "Medicated"
        }
        return s.map(lambda x: med_map.get(x, med_map.get(str(x), str(x))))
        
    return s


def bin_numeric(series: pd.Series,
                labels: Iterable[str] = ("Low IQ", "Mid IQ", "High IQ"),
                bins: Optional[Iterable[Union[int, float]]] = None,
                q: Iterable[float] = (0, 1/3, 2/3, 1)) -> pd.Series:
    """
    Bins a numeric series (like IQ) into categorical labels (e.g., Tertiles).
    """
    ser = pd.to_numeric(series, errors="coerce")
    ser = ser[ser.notna()]
    
    if ser.empty:
        return pd.Categorical([np.nan] * len(series), categories=labels)

    # Prepare bin edges
    smin, smax = float(ser.min()), float(ser.max())
    span = max(smax - smin, 1.0)
    tiny = max(1e-12, 1e-9 * span)

    # Strategy 1: User provided explicit bins
    if bins is not None:
        edges = list(map(float, bins))
        edges.sort()
        if edges[0] > smin: edges[0] = smin - tiny
        if edges[-1] < smax: edges[-1] = smax + tiny
        
        # Ensure monotonicity
        for i in range(1, len(edges)):
            if edges[i] <= edges[i-1]:
                edges[i] = np.nextafter(edges[i-1], float('inf'))
                
        return pd.cut(pd.to_numeric(series, errors="coerce"),
                      bins=edges, include_lowest=True, labels=labels)
        
    # Strategy 2: Quantiles (Default: Tertiles)
    qs = np.array(ser.quantile(q, interpolation="linear"), dtype=float)
    
    # Ensure monotonicity
    for i in range(1, len(qs)):
        if qs[i] <= qs[i-1]:
            qs[i] = np.nextafter(qs[i-1], float('inf'))
            
    if qs[-1] < smax:  
        qs[-1] = smax + tiny
    qs[-1] = np.nextafter(qs[-1], float('inf')) 
    
    return pd.cut(pd.to_numeric(series, errors="coerce"),
                  bins=qs, include_lowest=True, labels=labels)


def generate_counts_table(
    df: pd.DataFrame,
    *,
    adhd_col: str = "adhd",
    scores: Iterable[int] = range(0, 15),
    groups: List[Tuple[str, str, str]] = None,
    caption: str = "Counts by Sex, IQ tertiles, and Medication across ADHD raw scores (0--14).",
    label: str = "tab:counts_sex_iq_med",
    outfile: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generates the summary dataframe and writes the LaTeX table.
    """
    if groups is None:
        groups = [
            ("Sex", "sex", "sex"),
            ("IQ", "iq", "numeric"),
            ("Medication", "adhd_med_flag", "med"),
        ]

    # Select and Clean Data
    use_cols = [adhd_col] + [col for _, col, _ in groups]
    sub = df[use_cols].copy()
    
    # Ensure ADHD column is integer
    sub[adhd_col] = pd.to_numeric(sub[adhd_col], errors="coerce")
    sub = sub.dropna(subset=[adhd_col])
    sub = sub[sub[adhd_col].isin(scores)]
    sub[adhd_col] = sub[adhd_col].astype(int)
    
    score_cols = list(scores)

    # --- Part 1: Overall "Count" Row ---
    overall_ct = sub[adhd_col].value_counts().reindex(score_cols, fill_value=0).astype(int)
    total_count = overall_ct.sum()
    
    count_df = pd.DataFrame([list(overall_ct.values) + [total_count]],
                            columns=score_cols + ["Total"])
    count_df.insert(0, "ADHD Score", "Count")
    count_df.insert(0, "Group", "")

    parts = [count_df]

    # --- Part 2: Grouped Rows (Sex, IQ, Meds) ---
    for (disp_name, col_name, kind) in groups:
        tmp = sub[[adhd_col, col_name]].copy()

        # Apply Normalization / Binning
        if kind in {"sex", "med"}:
            tmp[col_name] = normalize_group_column(tmp[col_name], kind)
        elif kind == "numeric":
            tmp[col_name] = bin_numeric(tmp[col_name], labels=("Low IQ", "Mid IQ", "High IQ"))
        
        tmp = tmp.dropna(subset=[col_name])

        # Cross-tabulation
        ct = pd.crosstab(tmp[col_name].astype(str), tmp[adhd_col])
        ct = ct.reindex(columns=score_cols, fill_value=0).astype(int)
        ct["Total"] = ct.sum(axis=1)

        # Format block
        block_df = ct.reset_index().rename(columns={col_name: "ADHD Score"})
        block_df.insert(0, "Group", "")
        if not block_df.empty:
            block_df.iloc[0, 0] = disp_name  # Set Group label on first row
            
        parts.append(block_df)

    combined = pd.concat(parts, axis=0, ignore_index=True)

    # --- Part 3: LaTeX Formatting ---
    colspec = "ll" + "r" * len(score_cols) + "r"
    header = " & ".join(["Group", "ADHD Score"] + [str(s) for s in score_cols] + ["Total"]) + r" \\"

    lines = []
    prev_group = None
    
    for i, row in combined.iterrows():
        g = str(row["Group"])
        lvl = str(row["ADHD Score"])
        vals = [str(int(row[s])) for s in score_cols] + [str(int(row["Total"]))]
        line = " & ".join([g, lvl] + vals) + r" \\"

        # Add horizontal rules
        if i == 1: lines.append(r"\midrule") # After Count row
        elif g != "" and prev_group == "": lines.append(r"\midrule") # Between groups

        lines.append(line)
        prev_group = g

    latex_str = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"\\begin{{tabular}}{{{colspec}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        + "\n".join(lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )

    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(latex_str)
        print(f"LaTeX table saved to: {outfile}")

    return combined


# --- Main Execution ---

def main():
    try:
        if not DATA_CSV.exists():
            raise FileNotFoundError(f"Data file not found: {DATA_CSV}")

        # 1. Load Data
        df = pd.read_csv(DATA_CSV, index_col=False)
        
        # 2. Filter for Baseline
        df_baseline = df[df['year'] == 'baseline'].copy()
        print(f"Processing {len(df_baseline)} baseline subjects.")

        # 3. Generate Table
        _ = generate_counts_table(
            df_baseline,
            outfile=OUTPUT_TEX
        )

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()