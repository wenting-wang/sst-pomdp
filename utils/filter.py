# Criteria details: https://nda.nih.gov/data-structure/abcd_sst02
# Details: https://docs.abcdstudy.org/latest/documentation/imaging/type_trial.html##stop-signal-task-sst

import shutil
import pandas as pd
from pathlib import Path
import zipfile
import numpy as np
import os

# --- Configuration ---
BASE_DIR = Path("/Users/w/Desktop/data")
INPUT_CSV = BASE_DIR / "mri_y_tfmr_sst_beh.csv"
RAW_SOURCE_DIR = BASE_DIR / "abcd_sst_tlb01"
CLEAN_DATA_DIR = BASE_DIR / "sst_valid_base"
FOUND_FILES_TXT = BASE_DIR / "sst_valid_filenames.txt"

# QC Thresholds
RT_FAST_THRESHOLD = 200
MAX_FAST_TRIALS = 3  # 0, 1, 2, 3 are OK. 4 is a fail. 

def filter_metadata(file_path):
    print(f"Loading metadata from {file_path}...")
    df = pd.read_csv(file_path, index_col=False, low_memory=False)

    mask = (
        (df['eventname'].str.contains('baseline', case=False, na=False)) & 
        (df['tfmri_sst_beh_performflag'] == 1) &
        (df['tfmri_sst_beh_switchflag'] == 0) & 
        (df['tfmri_sst_beh_glitchflag'] == 0) &
        (df['tfmri_sst_beh_0ssdcount'] <= 3) &
        (df['tfmri_sst_beh_violatorflag'] == 0) &
        (df['tfmri_sst_all_beh_total_nt'] == 360) &
        (df['tfmri_sst_all_beh_s_nt'] == 60)
    )
    return df[mask].copy()

def check_rt_quality(zip_path):
    """Returns True (Pass) or False (Fail) based on RT < 200ms rule."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_files = [f for f in z.namelist() if f.endswith('sst.csv') and '__MACOSX' not in f]
            if not csv_files: return False
            
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f, usecols=['sst_primaryrt'])
            
            valid_rts = df[df['sst_primaryrt'] > 0]['sst_primaryrt'].values
            n_fast = np.sum(valid_rts < RT_FAST_THRESHOLD) # Strictly < 200
            
            return n_fast <= MAX_FAST_TRIALS

    except Exception:
        return False

def main():
    # 1. Metadata Filter
    df_filtered = filter_metadata(INPUT_CSV)
    print(f"Subjects passing metadata: {len(df_filtered)}")

    # 2. File Copy & Content Check
    if not CLEAN_DATA_DIR.exists():
        CLEAN_DATA_DIR.mkdir(parents=True)
    
    stats = {'copied': 0, 'missing_file': 0, 'bad_rt': 0}
    
    print(f"Processing files from {RAW_SOURCE_DIR}...")
    
    for _, row in df_filtered.iterrows():
        filename = f"{row['src_subject_id']}_{row['eventname']}_sst.csv.zip"
        src = RAW_SOURCE_DIR / filename
        dest = CLEAN_DATA_DIR / filename
        
        if not src.exists():
            stats['missing_file'] += 1
            continue
            
        if check_rt_quality(src):
            shutil.copy2(src, dest)
            stats['copied'] += 1
        else:
            stats['bad_rt'] += 1

    # 3. Final Report
    print("\n" + "="*40)
    print("FINAL FILTERING REPORT")
    print("="*40)
    print(f"Metadata Valid Subjects : {len(df_filtered)}")
    print(f" - Missing Raw Files    : {stats['missing_file']}")
    print(f" - Failed RT Quality    : {stats['bad_rt']}")
    print(f" - Successfully Copied  : {stats['copied']}")
    print("="*40)

    # 4. Save Log
    with open(FOUND_FILES_TXT, 'w') as f:
        for file_path in sorted(CLEAN_DATA_DIR.glob('*.zip')):
            f.write(f"{file_path.name}\n")
    print(f"List saved to {FOUND_FILES_TXT}")

if __name__ == "__main__":
    main()

# ========================================
# FINAL FILTERING REPORT
# ========================================
# Metadata Valid Subjects : 6773
#  - Missing Raw Files    : 682
#  - Failed RT Quality    : 746
#  - Successfully Copied  : 5345
# ========================================