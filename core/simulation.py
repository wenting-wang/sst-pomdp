import numpy as np


def simu_trial(model, true_go_state, true_stop_state, ssd, verbose=False):
    def generate_obs_sequences(true_go_state, true_stop_state, ssd):
        if true_stop_state == 'stop': # stop trial
            if ssd is None or np.isnan(ssd):
                raise ValueError("SSD must be specified for stop trials.")
            ssd = int(ssd)

            y_pre = np.random.choice([0, 1, 2], ssd, p=model.p_y_z0_pre)
            y_on = np.random.choice([0, 1, 2], mu, p=model.p_y_z1_on)
            y_post = np.random.choice([0, 1, 2], max(
                horizon - ssd - mu, 0), p=model.p_y_z0_post)
            y_seq = np.concatenate((y_pre, y_on, y_post))

            p_x_z0, p_x_z1 = {"left": (model.p_x_dL_z0, model.p_x_dL_z1),
                                "right": (model.p_x_dR_z0, model.p_x_dR_z1)}[true_go_state]
            x_pre = np.random.choice([0, 1, 2], ssd, p=p_x_z0)
            x_on_post = np.random.choice(
                [0, 1, 2], horizon - ssd, p=p_x_z1)
            x_seq = np.concatenate((x_pre, x_on_post))
        elif true_stop_state == 'nonstop': # go trial
            if ssd is not None and not np.isnan(ssd):
                raise ValueError("SSD must be None or NaN for go trials.")

            y_seq = np.random.choice([0, 1, 2], horizon, p=model.p_y_z0_pre)
            x_seq = np.random.choice([0, 1, 2], horizon,
                                        p={"left": model.p_x_dL_z0,
                                        "right": model.p_x_dR_z0}[true_go_state])
        return x_seq, y_seq, ssd

    def get_action_and_qvals(beta, sigma, t_step):
        beta_idx = np.searchsorted(model.beta_space, beta, side="left")
        zeta_idx = np.searchsorted(model.zeta_space, sigma, side="left")
        idx = (t_step, beta_idx, zeta_idx)

        v_left = model.value_left[idx]
        v_right = model.value_right[idx]
        v_wait = model.value_wait[idx]

        # softmax action selection
        q_vals = np.array([v_left, v_right, v_wait])
        q_vals = -q_vals  # cost
        q_vals -= np.max(q_vals)
        probs = np.exp(model.inv_temp * q_vals)
        probs /= np.sum(probs)
        action = np.random.choice([0, 1, 2], p=probs)

        return action, q_vals, (v_left, v_right, v_wait)

    # --- Constants ---
    horizon = model.horizon
    mu = 12  # stop signal duration in time steps
    x_seq, y_seq, ssd = generate_obs_sequences(
        true_go_state, true_stop_state, ssd)

    # --- Initial beliefs ---
    beta, zeta, sigma = 0.5, 0.0, model.update_sigma(0.0, 0)
    beta_seq, zeta_seq, sigma_seq = [beta], [zeta], [sigma]
    action, q_vals, (v_left, v_right, v_wait) = get_action_and_qvals(
        beta, sigma, 0)

    if verbose:
        value_seq = [q_vals[action]]
        value_left_seq = [v_left]
        value_right_seq = [v_right]
        value_wait_seq = [v_wait]
    policy_seq = [action]

    rt, simu_action, ever_act = None, None, False

    for t_step, (x, y) in enumerate(zip(x_seq, y_seq)):
        if t_step >= horizon - 2:
            break

        zeta = model.update_zeta(zeta, y, t_step)
        sigma = model.update_sigma(zeta, t_step)
        beta = model.update_beta(beta, zeta, x)

        beta_seq.append(beta)
        zeta_seq.append(zeta)
        sigma_seq.append(sigma)

        action, q_vals, (v_left, v_right, v_wait) = get_action_and_qvals(
            beta, sigma, t_step)

        if verbose:
            value_seq.append(q_vals[action])
            value_left_seq.append(v_left)
            value_right_seq.append(v_right)
            value_wait_seq.append(v_wait)
        policy_seq.append(action)

        if action != 2 and not ever_act:
            simu_action = action
            rt = t_step
            ever_act = True

    # --- Outcome decoding ---
    simu_action = {0: 'left', 1: 'right', 2: 'wait'}.get(simu_action)
    if true_stop_state == 'nonstop':
        if simu_action == true_go_state:
            result = 'GS'
        elif simu_action in ['left', 'right']:
            result = 'GE'
        else:
            result = 'GM'
    else:
        result = 'SS' if not ever_act else 'SE'

    # --- Non-Decision Time ---
    if rt is not None:
        rt += model.ndt

    # --- Output ---
    if verbose:
        return {
            'true_go_state': true_go_state,
            'true_stop_state': true_stop_state,
            'ssd': ssd,
            'beta_seq': beta_seq,
            'zeta_seq': zeta_seq,
            'sigma_seq': sigma_seq,
            'value_seq': value_seq,
            'value_left_seq': value_left_seq,
            'value_right_seq': value_right_seq,
            'value_wait_seq': value_wait_seq,
            'policy_seq': policy_seq,
            'result': result,
            'rt': rt
        }
    else:
        return (result, rt, ssd)

def simu_trial_batch(model, true_go_state, true_stop_state, ssd, batch_size, verbose=False):
    """
    Simulate multiple repeated Stop Signal Task (SST) trials under identical conditions.

    Parameters:
    - true_go_state (str): Direction of the go signal ('left' or 'right').
    - true_stop_state (str): Trial type ('nonstop' or 'stop').
    - ssd (int or None): Stop-signal delay (only required for stop trials).
    - batch_size (int): Number of trials to simulate.
    - verbose (bool): If True, returns full belief and value traces per trial.

    Returns:
    - list: If verbose=False: list of (result, rt, ssd) tuples.
            If verbose=True: list of detailed dicts for each trial (see `simu_trial`).
    """
    trials = [
        simu_trial(model, true_go_state, true_stop_state,
                        ssd, verbose=verbose)
        for _ in range(batch_size)
    ]
    return trials

def simu_task(model):
    """
    Simulate a full SST session consisting of 360 trials, including dynamic SSD staircase adjustment.

    Stop trials occur with base probability (1/6). The SSD (stop-signal delay) is updated only
    after stop trials, based on performance:

    - If previous stop trial was 'SS' (stop success), SSD increases by 2 steps (50ms).
    - If previous stop trial was 'SE' (stop error), SSD decreases by 2 steps.
    - SSD is constrained between 2 and 36 steps (50ms - 900ms).

    Returns:
    - list of tuples: Each trial as (result, rt, ssd)
    """
    n_trial = 360
    outcomes = []
    ssd_step = 2
    first_ssd = True
    last_stop_result = None

    for _ in range(n_trial):
        true_go_state = np.random.choice(['left', 'right'])
        true_stop_state = np.random.choice(
            ['nonstop', 'stop'], p=[5/6, 1/6])

        if true_stop_state == 'stop':
            if first_ssd:
                ssd_step = 2
                first_ssd = False
            elif last_stop_result == 'SS':
                ssd_step = min(ssd_step + 2, 34)
            elif last_stop_result == 'SE':
                ssd_step = max(ssd_step - 2, 2)
            ssd = ssd_step
        else:
            ssd = None

        trial_outcome = simu_trial(model,
            true_go_state, true_stop_state, ssd)
        outcomes.append(trial_outcome)

        if true_stop_state == 'stop':
            last_stop_result = trial_outcome[0]

    return outcomes