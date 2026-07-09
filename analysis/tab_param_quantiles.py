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
else:
    print(">>> RUNNING IN PROD MODE (Using local raw data) <<<")
    INPUT_DIR = Path("/Users/w/sst-pomdp/outputs/tesbi/model_posteriors")

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
        print(f"No 'params_posteriors_*.csv' files found in {INPUT_DIR}")
        return

    print(f"Found {len(csv_files)} model posterior files to process.\n")

    # Helper function to extract the integer parameter count for correct sorting
    def extract_p_count(filepath):
        match = re.search(r'_(\d+)p_', filepath.name)
        return int(match.group(1)) if match else 0

    # Sort files in descending order based on the numeric value (e.g., 10, 8, 7, 6)
    sorted_csvs = sorted(csv_files, key=extract_p_count, reverse=True)

    for file_path in sorted_csvs:
        # Extract model name from filename (e.g., "10p_v1" from "params_posteriors_10p_v1.csv")
        model_name = file_path.stem.replace("params_posteriors_", "")
        
        print("="*70)
        print(f" MODEL CONFIGURATION: {model_name}")
        print("="*70)
        print(f"Reading file: {file_path}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            continue

        # ---------------------------------------------------------
        # Calculate and Print Quantiles
        # ---------------------------------------------------------
        print(f"\nParameter Mean Distributions (Quantiles) for {model_name}:")
        
        # Group by parameter index and calculate 20%, 50%, 80% quantiles
        quantiles_df = df.groupby('index')['mean'].quantile([0.2, 0.5, 0.8]).unstack()

        print(f"{'Parameter':<18} | {'20% Quantile':<12} | {'50% (Median)':<12} | {'80% Quantile':<12}")
        print("-" * 65)

        for param in quantiles_df.index:
            q20 = quantiles_df.loc[param, 0.2]
            q50 = quantiles_df.loc[param, 0.5]
            q80 = quantiles_df.loc[param, 0.8]
            print(f"{param:<18} | {q20:<12.4f} | {q50:<12.4f} | {q80:<12.4f}")
        print("-" * 65 + "\n")

if __name__ == "__main__":
    main()

