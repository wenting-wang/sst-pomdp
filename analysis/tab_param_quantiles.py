import pandas as pd
from pathlib import Path
import argparse
import re

# =============================================================================
# 0. ARGUMENT PARSING (DEMO vs PROD MODE)
# =============================================================================
parser = argparse.ArgumentParser(description="Analyze parameter posteriors across multiple models")
parser.add_argument(
    '--prod', 
    action='store_true', 
    help='Run in PROD mode using local raw data. Usage: run `python script.py --prod`.'
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
    INPUT_DIR = PROJECT_ROOT / "outputs_example" / "tesbi" / "model_posteriors"
    OUT_CSV = PROJECT_ROOT / "outputs_example" / "tab_param_quantiles.csv"
else:
    print(">>> RUNNING IN PROD MODE (Using local raw data) <<<")
    INPUT_DIR = Path("/Users/w/sst-pomdp/outputs/tesbi/model_posteriors")
    OUT_DIR = PROJECT_ROOT / "outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV = OUT_DIR / "tab_param_quantiles.csv"

# =============================================================================
# Selected models to analyze
MODELS = ['10p_v1', '8p_v2', '7p_v3', '6p_v1']

# =============================================================================
# 2. BATCH PROCESSING LOOP
# =============================================================================
def main():
    if not INPUT_DIR.exists():
        print(f"Error: Input directory {INPUT_DIR} does not exist.")
        return

    # Find all parameter posterior CSVs in the target directory
    csv_files = list(INPUT_DIR.glob("*posteriors_*.csv"))
    csv_files = [f for f in csv_files if any(model in f.name for model in MODELS)]
    
    if not csv_files:
        print(f"No '*posteriors_*.csv' files found in {INPUT_DIR}")
        return

    print(f"Found {len(csv_files)} model posterior files to process.\n")

    # Helper function to extract the integer parameter count for correct sorting
    def extract_p_count(filepath):
        match = re.search(r'_(\d+)p_', filepath.name)
        return int(match.group(1)) if match else 0

    # Sort files in descending order based on the numeric value (e.g., 10, 8, 7, 6)
    sorted_csvs = sorted(csv_files, key=extract_p_count, reverse=True)

    summary_dict = {}
    all_params = set()
    ordered_models = []

    for file_path in sorted_csvs:
        # Extract clean model name
        model_name = file_path.stem.replace("params_posteriors_", "").replace("example_posteriors_", "")
        ordered_models.append(model_name)
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            continue

        # Calculate quantiles
        quantiles_df = df.groupby('index')['mean'].quantile([0.2, 0.5, 0.8]).unstack()

        for param in quantiles_df.index:
            q20 = quantiles_df.loc[param, 0.2]
            q50 = quantiles_df.loc[param, 0.5]
            q80 = quantiles_df.loc[param, 0.8]
            
            # Format: Median [20%, 80%]
            val_str = f"{q50:.3f} [{q20:.3f}, {q80:.3f}]"

            if param not in summary_dict:
                summary_dict[param] = {}
            summary_dict[param][model_name] = val_str
            all_params.add(param)

    # ---------------------------------------------------------
    # Build the combined wide-format DataFrame
    # ---------------------------------------------------------
    df_rows = []
    # Sort parameters alphabetically for consistency
    for param in sorted(list(all_params)):
        row = {'Parameter': param}
        for model in ordered_models:
            row[model] = summary_dict[param].get(model, "-") # Use "-" if param doesn't exist in that model
        df_rows.append(row)

    final_df = pd.DataFrame(df_rows)

    # Save to CSV
    final_df.to_csv(OUT_CSV, index=False)
    print(f"Saved combined CSV table to: {OUT_CSV}")

if __name__ == "__main__":
    main()