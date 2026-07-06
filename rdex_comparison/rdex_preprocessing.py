import os
import zipfile
import pandas as pd
import numpy as np

def process_abcd_sst_for_dmc(subject_id, input_dir, output_dir):
    # Construct the expected zip filename based on standard ABCD naming conventions
    zip_filename = f"{subject_id}_baseline_year_1_arm_1_sst.csv.zip"
    zip_path = os.path.join(input_dir, zip_filename)

    if not os.path.exists(zip_path):
        print(f"Error: Zip file not found at {zip_path}")
        return None

    # Read the CSV directly from the zip file
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        if not csv_files:
            print("Error: No CSV file found inside the zip archive.")
            return None
        
        target_csv = csv_files[0]
        with z.open(target_csv) as f:
            df = pd.read_csv(f)

    # 1. Filter task trials only
    df = df[df['sst_expcon'].isin(['GoTrial', 'VariableStopTrial'])].copy()

    # 2. Extract Stimulus (S): 'left' or 'right'
    df['S'] = df['sst_stim'].apply(
        lambda x: 'left' if 'left' in str(x).lower() else ('right' if 'right' in str(x).lower() else np.nan)
    )

    # 3. Define Trial Type (SS): 'GO' or 'SS'
    df['SS'] = df['sst_expcon'].apply(
        lambda x: 'GO' if x == 'GoTrial' else 'SS'
    )

    # 4. Extract Stop Signal Delay (SSD): Convert to seconds, fill GO trials with np.inf
    df['SSD'] = np.where(df['SS'] == 'SS', df['sst_ssd_dur'] / 1000.0, np.inf)

    # 5. Extract Reaction Time (RT): Convert to seconds, 0 becomes NaN
    df['RT'] = df['sst_primaryrt'] / 1000.0
    df.loc[df['RT'] == 0, 'RT'] = np.nan

    # 6. Apply Data Cleaning (Excluding fast guesses < 0.15s and abnormally slow > 1.5s)
    valid_rt_mask = df['RT'].isna() | ((df['RT'] >= 0.15) & (df['RT'] <= 1.5))
    df_clean = df[valid_rt_mask].copy()

    # 7. Map Response (R) specifically for DMC: 'LEFT', 'RIGHT', or 'NR'
    def map_response(resp):
        resp_str = str(resp).lower()
        if 'left' in resp_str:
            return 'LEFT'
        elif 'right' in resp_str:
            return 'RIGHT'
        else:
            return 'NR'
            
    df_clean['R'] = df_clean['sst_primaryresp'].apply(map_response)

    # 8. Rename subject column to 's' to match R scripts
    df_clean.rename(columns={'subject': 's'}, inplace=True)

    # 9. Select the final columns ordered exactly as needed for DMC
    final_cols = ['s', 'S', 'R', 'SS', 'SSD', 'RT']
    df_final = df_clean[final_cols]

    # Create output directory and save
    os.makedirs(output_dir, exist_ok=True)
    out_filename = f"{subject_id}_dmc_ready.csv"
    out_path = os.path.join(output_dir, out_filename)
    
    df_final.to_csv(out_path, index=False)
    print(f"Successfully processed {subject_id}. Kept {len(df_final)} trials.")
    print(f"Saved to: {out_path}")
    
    return df_final


if __name__ == "__main__":
    # SUBJECT_ID = 'NDAR_INVRUVFBX1L'
    # SUBJECT_ID = 'NDAR_INVPWL6YB2C'
    SUBJECT_ID = 'NDAR_INVRH6B02GB'
    INPUT_DIRECTORY = '/Users/w/Desktop/data/sst_valid_base'
    OUTPUT_DIRECTORY = 'rdex_processed'

    processed_data = process_abcd_sst_for_dmc(
        subject_id=SUBJECT_ID,
        input_dir=INPUT_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY
    )