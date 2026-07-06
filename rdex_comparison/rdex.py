import numpy as np

class RDEX:
    def __init__(self, params=None):
        """
        Initialize the RDEX-ABCD model.
        If no params are provided, it defaults to the 600-person group average 
        reported in Weigard et al. (2023).
        NOTE: These base parameters are calibrated in SECONDS.
        """
        if params is None:
            self.params = {
                't0': 0.137,      # Go non-decision time (s)
                'B': 1.417,       # Go evidence threshold
                'v_plus': 3.285,  # Asymptotic Go match rate
                'v_minus': 0.401, # Asymptotic Go mismatch rate
                'v0': 2.713,      # Go processing speed at SSD = 0
                'g': 2.953,       # Perceptual growth rate (linear slope)
                'mu': 0.229,      # Stop ex-Gaussian normal mean (s)
                'sigma': 0.076,   # Stop ex-Gaussian normal SD (s)
                'tau': 0.030,     # Stop ex-Gaussian exponential mean (s)
                'p_gf': 0.027,    # Probability of go failure
                'p_tf': 0.037     # Probability of trigger failure
            }
        else:
            self.params = params

    def simu_trial(self, true_go_state, true_stop_state, ssd_ms=None, verbose=False):
        """
        Simulate a single trial of the RDEX-ABCD task.
        
        Returns:
            tuple: (result, rt_ms, ssd_ms, is_correct)
                   RT is returned in ms. Missing responses are marked as np.nan.
        """
        p = self.params
        
        # 1. Context Independence Violation Mechanism
        if true_stop_state == 'stop' and ssd_ms is not None:
            ssd_sec = ssd_ms / 1000.0
            v_m = min(p['v0'] + p['g'] * ssd_sec, p['v_plus'])
            v_mm = max(p['v0'] - p['g'] * ssd_sec, p['v_minus'])
        else:
            ssd_sec = np.inf
            v_m = p['v_plus']
            v_mm = p['v_minus']
            
        # 2. Go Process
        if np.random.rand() < p['p_gf']:
            go_time_m, go_time_mm = np.inf, np.inf
        else:
            go_time_m = np.random.wald(p['B'] / v_m, p['B']**2)
            go_time_mm = np.random.wald(p['B'] / v_mm, p['B']**2)
        
        go_rt_raw = min(go_time_m, go_time_mm)
        
        if go_time_m < go_time_mm:
            is_correct = True
        elif go_time_mm < go_time_m:
            is_correct = False
        else:
            is_correct = False # Tie / Omission
            
        final_go_rt = p['t0'] + go_rt_raw if go_rt_raw != np.inf else np.inf
        
        # Preprocessing exclusion rule (> 1s is treated as an omission)
        if final_go_rt > 1.0:
            final_go_rt = np.inf
            is_correct = False
            
        # 3. Stop Process & The Race
        if true_stop_state == 'stop':
            if np.random.rand() < p['p_tf']:
                stop_runner_time = np.inf
            else:
                exg_sample = np.random.normal(p['mu'], p['sigma']) + np.random.exponential(p['tau'])
                stop_runner_time = max(0.05, exg_sample)
            
            if final_go_rt <= ssd_sec + stop_runner_time:
                result = 'SE' # Stop Error (Signal Respond)
                rt_sec = final_go_rt
            else:
                result = 'SS' # Stop Success (Inhibited)
                rt_sec = np.nan
        else:
            if final_go_rt == np.inf:
                result = 'GM' # Go Miss
                rt_sec = np.nan
            elif is_correct:
                result = 'GS' # Go Success
                rt_sec = final_go_rt
            else:
                result = 'GE' # Go Error
                rt_sec = final_go_rt

        rt_ms = rt_sec * 1000.0 if not np.isnan(rt_sec) else np.nan
        
        if verbose:
            return (result, rt_ms, ssd_ms, is_correct, go_time_m, go_time_mm)
        else:
            return (result, rt_ms, ssd_ms, is_correct)

    def simu_task(self, verbose=False):
        """
        Simulate a full SST session of 360 trials with dynamic SSD staircase adjustment.
        """
        n_trial = 360
        outcomes = []
        current_ssd_ms = 50
        first_ssd = True
        last_stop_result = None

        for _ in range(n_trial):
            true_go_state = np.random.choice(['left', 'right'])
            true_stop_state = np.random.choice(['nonstop', 'stop'], p=[5/6, 1/6])

            if true_stop_state == 'stop':
                if first_ssd:
                    current_ssd_ms = 50
                    first_ssd = False
                elif last_stop_result == 'SS':
                    current_ssd_ms = min(current_ssd_ms + 50, 850)
                elif last_stop_result == 'SE':
                    current_ssd_ms = max(current_ssd_ms - 50, 50)
                trial_ssd_ms = current_ssd_ms
            else:
                trial_ssd_ms = None

            trial_outcome = self.simu_trial(true_go_state, true_stop_state, ssd_ms=trial_ssd_ms, verbose=verbose)
            outcomes.append(trial_outcome)

            if true_stop_state == 'stop':
                last_stop_result = trial_outcome[0] 

        return outcomes

# if __name__ == '__main__':
#     model = RDEX()
    
#     print("--- Test 1: simu_trial ---")
#     go_out = model.simu_trial(true_go_state='left', true_stop_state='nonstop')
#     print(f"Go trial: {go_out}")
    
#     stop_out = model.simu_trial(true_go_state='right', true_stop_state='stop', ssd_ms=250)
#     print(f"Stop trial (SSD=250ms): {stop_out}")
    
#     print("\n--- Test 2: simu_task ---")
#     task_outcomes = model.simu_task()
#     print(f"Total trials simulated: {len(task_outcomes)}")
    
#     print("\nFirst 5 Stop trials (Checking SSD staircase):")
#     stop_count = 0
#     for trial in task_outcomes:
#         result, rt_ms, ssd_ms, is_correct = trial
#         # if result in ['SS', 'SE']:
#         if result in ['GS', 'GE', 'GM', 'SS', 'SE']:
#             print(f"Result: {result:2s} | RT: {str(rt_ms)[:6]:>6s} | SSD: {ssd_ms}")
#             stop_count += 1
#             if stop_count >= 20:
#                 break