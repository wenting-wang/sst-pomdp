import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
import re
from pathlib import Path
from utils.preprocessing import preprocessing
from utils.metrics import get_stats_mean

# --- Configuration & Constants ---

# Paths
BASE_DIR = Path('/kyb/agpd/wwang/data/')
OUT_DIR = Path('/kyb/agpd/wwang/sst-pomdp/outputs/')
FILE_LIST_PATH = BASE_DIR / 'sst_valid_filenames.txt'
DATA_DIR = BASE_DIR / 'sst_valid_base'

# Input Data Files
CBCL_FILE = BASE_DIR / 'mh_p_cbcl.csv'
DEMO_FILE = BASE_DIR / 'abcd_p_demo.csv'
MEDS_FILE = BASE_DIR / 'ph_p_meds.csv'
IQ_FILE = BASE_DIR / 'nc_y_wisc.csv'

# Output Files
OUT_FINAL_CSV = OUT_DIR /  'stats_beh_full.csv'     # Behavioral stats + Demographics + Clinical

# Constants
YEAR_MAP = {
    'baseline_year_1_arm_1': 'baseline',
    '1_year_follow_up_y_arm_1': '1',
    '2_year_follow_up_y_arm_1': '2',
    '3_year_follow_up_y_arm_1': '3',
    '4_year_follow_up_y_arm_1': '4'
}

ADHD_MEDS = [
    "Adderall", "Concerta", "Methylphenidate", "Ritalin", "Focalin", "Strattera", "Amphetamine",
    "Quillivant", "Guanfacine", "Evekeo", "Atomoxetine", "Lisdexamfetamine", "Dexedrine", "Dynavel",
    "Adzenys", "Metadate", "Kapvay", "Clonidine", "Intuniv", "Daytrana", "Methylin", "Dextrostat",
    "Zenzedi", "Tenex", "Catapres", "Aptensio", "Cotempla", "Quillichew", "Bupropion", "Wellbutrin",
    "Norpramin", "Desipramine", "Impiprmine", "Tofranil", "Nortriptyline", "Aventyl", "Pamelor"
]

# --- Helper Functions ---

def clean_subject_id(raw_id):
    """Extracts 'INV...' from 'NDAR_INV...'."""
    return raw_id.split('_')[1] if '_' in raw_id else raw_id

def clean_year(raw_year):
    """Maps raw event names to simplified years."""
    return YEAR_MAP.get(raw_year, raw_year.split('_')[0])

# --- Data Loading Functions ---

def get_behavioral_stats(file_list_path, data_dir):
    """Iterates through raw files and computes behavioral statistics."""
    results = []
    failures = []

    with file_list_path.open('r') as f:
        filenames = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(filenames)} behavioral files...")

    for fname in filenames:
        # Expected format: NDAR_INVXJK60DCE_baseline_year_1_arm_1_sst.csv.zip
        parts = fname.split('_')
        if len(parts) < 3 or parts[0] != 'NDAR':
            failures.append((fname, 'unexpected filename pattern'))
            continue

        subject_id = parts[1]
        year = parts[2] # 'baseline' usually
        
        file_path = data_dir / fname
        
        if not file_path.exists():
            failures.append((fname, 'file not found'))
            continue

        try:
            # Process using external modules
            df_obs = preprocessing(str(file_path))
            stats = get_stats_mean(df_obs)
            
            # Unpack stats (ensure get_stats_mean returns exactly these 10 values)
            (perc_gs, perc_ge, perc_gm, perc_ss,
             mrt_gs, mrt_ge, mrt_se, ssrt,
             rate_perc_ss_ssd, rate_rt_se_ssd) = stats

            results.append({
                'subject_id': subject_id,
                'year': year,
                'filename': fname,
                'perc_gs': perc_gs, 'perc_ge': perc_ge, 'perc_gm': perc_gm, 'perc_ss': perc_ss,
                'mrt_gs': mrt_gs, 'mrt_ge': mrt_ge, 'mrt_se': mrt_se, 'ssrt': ssrt,
                'rate_perc_ss_ssd': rate_perc_ss_ssd, 'rate_rt_se_ssd': rate_rt_se_ssd,
            })

        except Exception as e:
            failures.append((fname, f'processing error: {e}'))

    if failures:
        print(f"Encountered {len(failures)} failures.")
    
    return pd.DataFrame(results)

def get_cbcl_data(path):
    """Loads and cleans CBCL (ADHD Score) data."""
    df = pd.read_csv(path)
    df['subject_id'] = df['src_subject_id'].apply(clean_subject_id)
    df['year'] = df['eventname'].apply(clean_year)
    
    df = df[['subject_id', 'year', 'cbcl_scr_dsm5_adhd_r']]
    df = df.rename(columns={'cbcl_scr_dsm5_adhd_r': 'adhd'})
    return df

def get_demo_data(path):
    """Loads and cleans Demographic (Sex) data."""
    df = pd.read_csv(path)
    df['subject_id'] = df['src_subject_id'].apply(clean_subject_id)
    df['year'] = df['eventname'].apply(clean_year)
    
    # 1=Male, 2=Female. Filter out Intersex/Unknown if needed (original script removed 3)
    df = df[df['demo_sex_v2'] != 3] 
    df['sex'] = df['demo_sex_v2'].replace({1: 'Male', 2: 'Female'})
    
    df = df[['subject_id', 'year', 'sex']]
    
    # Fill missing sex within subject groups
    df['sex'] = df.groupby('subject_id')['sex'].transform(lambda s: s.ffill().bfill())
    return df

def get_med_data(path):
    """Loads Medication data and flags ADHD med usage."""
    df = pd.read_csv(path)
    
    # Identify medication columns
    med_cols = [col for col in df.columns if "rxnorm_p" in col]
    
    # Regex search for ADHD meds
    pattern = re.compile("|".join(ADHD_MEDS), flags=re.IGNORECASE)
    
    df["adhd_med_flag"] = df[med_cols].apply(
        lambda row: int(any(bool(pattern.search(str(v))) for v in row)), axis=1
    )
    
    df['subject_id'] = df['src_subject_id'].apply(clean_subject_id)
    df['year'] = df['eventname'].apply(clean_year)
    
    return df[['subject_id', 'year', 'adhd_med_flag']]

def get_iq_data(path):
    """Loads IQ data."""
    df = pd.read_csv(path)
    df['subject_id'] = df['src_subject_id'].apply(clean_subject_id)
    df['year'] = df['eventname'].apply(clean_year)
    
    df = df[['subject_id', 'year', 'pea_wiscv_trs']]
    df = df.rename(columns={'pea_wiscv_trs': 'iq'})
    return df

# --- Main Execution ---

def main():
    # 1. Compute Behavioral Stats (Time Consuming)
    print("Computing behavioral stats from raw files...")
    df_stats = get_behavioral_stats(FILE_LIST_PATH, DATA_DIR)

    # 2. Load Metadata
    print("Loading Demographics, CBCL, Meds, and IQ...")
    df_cbcl = get_cbcl_data(CBCL_FILE)
    df_demo = get_demo_data(DEMO_FILE)
    df_meds = get_med_data(MEDS_FILE)
    df_iq   = get_iq_data(IQ_FILE)

    # 3. Merge DataFrames
    # Start with stats, join everything on subject_id and year
    print("Merging datasets...")

    # Inner join for critical demographics
    df_merged = pd.merge(df_stats, df_cbcl, on=['subject_id', 'year'], how='inner')
    df_merged = pd.merge(df_merged, df_demo, on=['subject_id', 'year'], how='inner')
    
    # Left join for Meds and IQ (we will dropna later)
    df_merged = pd.merge(df_merged, df_meds, on=['subject_id', 'year'], how='left')
    df_merged = pd.merge(df_merged, df_iq, on=['subject_id', 'year'], how='left')

    # 4. Final Cleanup
    initial_len = len(df_merged)
    df_final = df_merged.dropna()
    print(f"Dropped {initial_len - len(df_final)} rows with missing data.")
    
    # 5. Save Final Output
    df_final.to_csv(OUT_FINAL_CSV, index=False)
    print(f"Done. Final dataset saved to: {OUT_FINAL_CSV}")

if __name__ == "__main__":
    main()