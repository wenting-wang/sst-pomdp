import shutil
import pandas as pd
from pathlib import Path
import zipfile
import numpy as np

# --- Configuration & Paths ---
BASE_DIR = Path("/kyb/agpd/wwang/data")

# Input
INPUT_CSV = BASE_DIR / "mri_y_tfmr_sst_beh.csv"
RAW_SOURCE_DIR = BASE_DIR / "abcd_sst_tlb01"

# Output (Data)
CLEAN_DATA_DIR = BASE_DIR / "sst_valid_base"

# Output (Log)
FOUND_FILES_TXT = BASE_DIR / "sst_valid_filenames.txt"

# New Filtering Criteria
RT_FAST_THRESHOLD = 200  # ms
MAX_FAST_RT_PCT = 1.0    # Drop subject if > 1.0% of valid trials are < 200ms

# --- Core Functions ---

def filter_data(file_path):
    """
    Loads data and filters subjects based on behavioral performance flags.
    Criteria details: https://nda.nih.gov/data-structure/abcd_sst02
    Details: https://docs.abcdstudy.org/latest/documentation/imaging/type_trial.html##stop-signal-task-sst
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_csv(file_path, index_col=False)

    # Filter Conditions:
    # 0. eventname: Must be baseline
    # 1. performflag: Acceptable performance (>3 events for pos/neg feedback)
    # 2. switchflag: Box was not switched
    # 3. glitchflag: No task coding error
    # 4. 0ssdcount: Number of Stop trials with 0ms SSD (Keep <= 3)
    # 5. violatorflag: No race model violators
    # 6. total_nt: Total trials must be 360
    # 7. s_nt: Stop trials must be 60
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
    
    filtered_df = df[mask].copy()
    print(f"Data filtered. Rows remaining: {len(filtered_df)} (Original: {len(df)})")
    
    return filtered_df


def get_valid_filenames(df_filtered):
    """Generates the expected filenames based on ID and event name."""
    valid_id = df_filtered['src_subject_id'].astype(str)
    valid_year = df_filtered['eventname'].astype(str)
    
    # Create filenames: {id}_{year}_sst.csv.zip
    # Since we filtered for 'baseline', 'year' will contain 'baseline'
    return [f'{id}_{year}_sst.csv.zip' for id, year in zip(valid_id, valid_year)]

def check_rt_quality(zip_path):
    """
    Opens the zip file and checks for anticipatory responses.
    Returns True if the subject passes the quality check (<= 1% fast RTs).
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Find the CSV file inside (ignore __MACOSX artifacts)
            csv_files = [f for f in z.namelist() if f.endswith('sst.csv') and '__MACOSX' not in f]
            
            if not csv_files:
                return False # Missing CSV is a fail
            
            # Read only the reaction time column
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f, usecols=['sst_primaryrt'])
            
            # Get valid reaction times (RT > 0)
            valid_rts = df[df['sst_primaryrt'] > 0]['sst_primaryrt'].values
            
            if len(valid_rts) == 0:
                return False # No valid responses to analyze
            
            # Calculate percentage of fast RTs
            n_fast = np.sum(valid_rts < RT_FAST_THRESHOLD)
            pct_fast = (n_fast / len(valid_rts)) * 100
            
            # Check threshold
            if pct_fast <= MAX_FAST_RT_PCT:
                return True
            else:
                return False # Too many fast responses

    except Exception as e:
        print(f"Error checking {zip_path.name}: {e}")
        return False

def copy_files(file_list, src_dir, dest_dir):
    """
    Copies files from src_dir to dest_dir ONLY if they:
    1. Are in the valid file_list (passed metadata filter)
    2. Pass the check_rt_quality (passed content filter)
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert list to set for faster lookup
    valid_files_set = set(file_list)
    
    copied_count = 0
    rejected_count = 0
    
    print(f"Scanning {src_dir} and validating RT data...")
    
    for file_path in src_dir.iterdir():
        # Check if file is in our valid list (which now guarantees 'baseline')
        if file_path.is_file() and file_path.name in valid_files_set:
            
            # --- Check Content Quality Before Copying ---
            if check_rt_quality(file_path):
                shutil.copy(file_path, dest_dir)
                # print(f"Copied: {file_path.name}")
                copied_count += 1
            else:
                rejected_count += 1
                # print(f"Skipped (Bad RT Data): {file_path.name}")
            
    print(f"\nFinal Summary:")
    print(f"  - Metadata valid files found: {copied_count + rejected_count}")
    print(f"  - Rejected for RT quality (>1% fast): {rejected_count}")
    print(f"  - Successfully copied: {copied_count}")
            
    return copied_count

# --- Main Execution ---

def main():
    # 1. Load and Filter Data (Metadata + Baseline check)
    df_filtered = filter_data(INPUT_CSV)
    
    # 2. Generate Valid File List
    valid_files = get_valid_filenames(df_filtered)

    # 3. Copy Files (includes Content check)
    copy_files(valid_files, RAW_SOURCE_DIR, CLEAN_DATA_DIR)

    # 4. Verify and Create Log (list files that actually made it)
    with open(FOUND_FILES_TXT, 'w') as f:
        for file_path in CLEAN_DATA_DIR.iterdir():
            if file_path.name.endswith('.zip'):
                f.write(f"{file_path.name}\n")
    
    print(f"Log saved to {FOUND_FILES_TXT}")

if __name__ == "__main__":
    main()