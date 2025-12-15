import pandas as pd
import numpy as np
from typing import Union
from pathlib import Path

# --- Constants ---
TIME_STEP_MS = 25  # Resolution for discretizing time (ms per step)

def preprocessing(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Preprocesses the raw ABCD SST trial-level data.

    Tasks:
    1. Categorizes trials into outcomes: 
       GS (Go Success), GE (Go Error), GM (Go Miss), SS (Stop Success), SE (Stop Error).
    2. Discretizes Reaction Time (RT) and Stop Signal Delay (SSD) into 25ms steps.
    3. Encodes trial states (Left/Right, Stop/Non-Stop).

    Args:
        file_path: Path to the raw subject CSV file.

    Returns:
        pd.DataFrame: Cleaned dataframe with columns:
            ['true_go_state', 'true_stop_state', 'ssd', 'ssd_real', 
             'rt', 'rt_real', 'rt_gs_real', 'result', 'sequence', 'response']
    """
    # Ensure file_path is a string for read_csv compatibility (though pandas handles Path usually)
    df = pd.read_csv(file_path)

    # --- 1. Define Trial Outcomes ---
    # GS: Go Trial, Correct Accuracy, Valid RT (<= 1000ms)
    is_gs = (df['sst_expcon'] == 'GoTrial') & \
            (df['sst_choiceacc'] == 1) & \
            (df['sst_primaryrt'] <= 1000)

    # GE: Go Trial, Incorrect Accuracy, Valid RT
    is_ge = (df['sst_expcon'] == 'GoTrial') & \
            (df['sst_choiceacc'] == 0) & \
            (df['sst_primaryrt'] <= 1000)

    # GM: Go Trial, No valid response (Response codes 1 or 2 usually indicate button press)
    # Note: Checks if response is NOT 1 AND NOT 2.
    is_gm = (df['sst_expcon'] == 'GoTrial') & \
            (df['sst_go_resp'] != 1) & \
            (df['sst_go_resp'] != 2)

    # SS: Stop Trial, Successful Inhibition
    is_ss = (df['sst_expcon'] == 'VariableStopTrial') & \
            (df['sst_inhibitacc'] == 1)

    # SE: Stop Trial, Failed Inhibition
    is_se = (df['sst_expcon'] == 'VariableStopTrial') & \
            (df['sst_inhibitacc'] == 0)

    # Consolidate into a single 'result' column
    # Priority is implicitly handled by the mutually exclusive nature of masks above.
    conditions = [is_gs, is_ge, is_gm, is_ss, is_se]
    choices = ['GS', 'GE', 'GM', 'SS', 'SE']
    df['result'] = np.select(conditions, choices, default=np.nan)

    # --- 2. Process Time Variables (RT & SSD) ---
    
    # Rename SSD and RT for clarity
    df = df.rename(columns={'sst_ssd_dur': 'ssd_real'})
    df['rt_real'] = df['sst_primaryrt']

    # Discretize time (ms -> steps)
    # SSD: stop signal delay
    df['ssd'] = df['ssd_real'] // TIME_STEP_MS
    
    # RT: response time
    df['rt'] = df['rt_real'] // TIME_STEP_MS

    # Calculate specific RTs (e.g., RT for Go Successes only)
    df["rt_gs_real"] = np.where(df["result"] == "GS", df["rt_real"], np.nan)

    # Log transformations (handle 0 or NaN safely)
    df['lg_rt_real'] = np.log(df['rt_real'].replace(0, np.nan))
    df['lg_rt_gs_real'] = np.log(df['rt_gs_real'].replace(0, np.nan))

    # --- 3. Encode Task States ---
    
    # Trial Type: nonstop (Go) vs stop (Stop)
    df['true_stop_state'] = df['sst_expcon'].map({
        'GoTrial': 'nonstop',
        'VariableStopTrial': 'stop'
    })

    # Stimulus Direction
    df['true_go_state'] = df['sst_stim'].map({
        'left_arrow': 'left', 
        'right_arrow': 'right'
    })

    # Sequence ID (0 for Go, 1 for Stop)
    df['sequence'] = df['sst_expcon'].map({
        'GoTrial': 0, 
        'VariableStopTrial': 1
    })

    # Binary Response: 0 = No Response (Inhibition), 1 = Response (Action)
    # GS, GE, SE -> Responded (1)
    # GM, SS    -> Did not respond (0)
    df['response'] = df['result'].map({
        'GS': 1, 'GE': 1, 'SE': 1, 
        'GM': 0, 'SS': 0
    })

    # --- 4. Select Final Columns ---
    columns_to_keep = [
        'true_go_state', 
        'true_stop_state', 
        'ssd', 
        'ssd_real', 
        'rt', 
        'rt_real', 
        'rt_gs_real', 
        'lg_rt_real', 
        'lg_rt_gs_real',
        'result', 
        'sequence', 
        'response'
    ]
    
    return df[columns_to_keep]