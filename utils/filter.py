import shutil
import pandas as pd
from pathlib import Path

# --- Configuration & Paths ---
BASE_DIR = Path("/kyb/agpd/wwang/data")

# Input
INPUT_CSV = BASE_DIR / "mri_y_tfmr_sst_beh.csv"
RAW_SOURCE_DIR = BASE_DIR / "abcd_sst_tlb01"

# Output (Data)
CLEAN_DATA_DIR = BASE_DIR / "sst_valid_base"

# Output (Log)
FOUND_FILES_TXT = BASE_DIR / "sst_valid_filenames.txt"


# --- Core Functions ---

def filter_data(file_path):
    """
    Loads data and filters subjects based on behavioral performance flags.
    Criteria details: https://nda.nih.gov/data-structure/abcd_sst02
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_csv(file_path, index_col=False)

    # Filter Conditions:
    # 1. performflag: Acceptable performance (>3 events for pos/neg feedback)
    # 2. switchflag: Box was not switched
    # 3. glitchflag: No task coding error
    # 4. 0ssdcount: Number of Stop trials with 0ms SSD (Keep <= 3)
    # 5. violatorflag: No race model violators
    # 6. total_nt: Total trials must be 360
    # 7. s_nt: Stop trials must be 60
    mask = (
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


def get_valid_filenames(df):
    """Generates the expected filenames based on subject ID and event name."""
    valid_id = df['src_subject_id'].to_list()
    valid_year = df['eventname'].to_list()
    
    # Create filenames: {id}_{year}_sst.csv.zip
    return [f'{id}_{year}_sst.csv.zip' for id, year in zip(valid_id, valid_year)]


def copy_files(file_list, src_dir, dest_dir):
    """Copies files present in file_list from src_dir to dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert list to set for faster lookup
    valid_files_set = set(file_list)
    copied_count = 0

    print(f"Scanning {src_dir}...")
    
    for file_path in src_dir.iterdir():
        if file_path.is_file() and file_path.name in valid_files_set:
            shutil.copy(file_path, dest_dir)
            print(f"Copied: {file_path.name}")
            copied_count += 1
            
    return copied_count


# --- Main Execution ---

def main():
    # 1. Load and Filter Data (In Memory)
    df_filtered = filter_data(INPUT_CSV)
    
    # 2. Generate Valid File List (In Memory)
    valid_files = get_valid_filenames(df_filtered)

    # 3. Copy Files
    total_copied = copy_files(valid_files, RAW_SOURCE_DIR, CLEAN_DATA_DIR)
    print(f"Total files copied: {total_copied}")

    # 4. Verify and Create Log
    # Lists files that now physically exist in the destination folder
    files_in_dest = [f.name for f in CLEAN_DATA_DIR.glob('*') if f.is_file()]
    
    with open(FOUND_FILES_TXT, 'w') as f:
        for item in files_in_dest:
            f.write(f"{item}\n")
            
    print(f"Verification log (found files) saved to: {FOUND_FILES_TXT}")


if __name__ == "__main__":
    main()