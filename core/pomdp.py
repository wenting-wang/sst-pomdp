# ABCD SST

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D


class POMDP:
    """
    Partially Observable Markov Decision Process (POMDP) for the Stop Signal Task (SST).

    This model simulates within-trial belief dynamics under uncertainty about sensory inputs
    and trial type. It supports dynamic hazard modeling and cost-sensitive decision policy.
    """

###################### Model ######################

    def __init__(self, q_d_n, q_d, q_s_n, q_s, cost_go_error, cost_go_missing,
                 cost_stop_error, cost_time, inv_temp, rate_stop_trial=1/6):
        """
        Initialize model with perceptual noise and cost parameters.

        Parameters:
        - q_d_n (float): Null level for go-signal. (chi')
        - q_d (float): Precision level of go-signal. (chi)
        - q_s_n (float): Null level for stop-signal. (delta')
        - q_s (float): Precision level of stop-signal. (delta)
        - cost_go_error (float): Cost of incorrect directional response.
        - cost_go_missing (float): Cost of missing response on go trials.
        - cost_stop_error (float): Cost of inhibition failure (stop error).
        - cost_time (float): Per-step time cost (e.g., delay penalty).
        - rate_stop_trial (float): Probability of stop trials (default: 1/6).
        """
        # Hazard function parameter
        self.lamb = 0.2
        # r=p(stop trial), from across-trial model
        self.rate_stop_trial = rate_stop_trial

        # Observation parameters
        self.q_d_n = q_d_n
        self.q_d = q_d
        self.q_s_n = q_s_n
        self.q_s = q_s

       # Cost parameters
        self.cost_go_error = cost_go_error
        self.cost_go_missing = cost_go_missing
        self.cost_stop_error = cost_stop_error
        self.cost_time = cost_time

        # Softmax inverse temperature
        self.inv_temp = inv_temp

        # Discretization
        self.bins = 80
        self.horizon = 41
        self.beta_space = np.linspace(0, 1, self.bins + 1)  # go signal belief
        self.zeta_space = np.linspace(
            0, 1, self.bins + 1)  # stop signal belief
        self.sigma_space = np.linspace(
            0, 1, self.bins + 1)  # stop trial belief
        self.shape = (self.horizon, self.bins + 1, self.bins + 1)

        # Time-varying hazard
        self.hazard = np.array([self._generate_hazard(t_step)
                               for t_step in range(self.horizon)])

        # Observation likelihoods
        # p(x | d, z)
        self.p_x_dL_z0, self.p_x_dR_z0 = self._generate_observation_probs(
            q_d, q_d_n)  # stop signal pre
        self.p_x_dL_z1, self.p_x_dR_z1 = self._generate_observation_probs(
            q_d, q_d_n, reverse=True)  # stop signal on and post

        # p(y | z)
        self.p_y_z0_pre, self.p_y_z1_pre = self._generate_observation_probs(
            q_s, q_s_n)
        self.p_y_z0_on, self.p_y_z1_on = self._generate_observation_probs(
            q_s, q_s_n)
        self.p_y_z0_post, self.p_y_z1_post = self._generate_observation_probs(
            q_s, q_s_n, reverse=True)

        # Action space: 0 = left, 1 = right, 2 = wait
        self.action_space = [0, 1, 2]

        # Value and policy tables
        self.policy = np.full(self.shape, np.nan)
        self.value = np.full(self.shape, np.nan)
        self.value_wait = np.full(self.shape, np.nan)
        self.value_left = np.full(self.shape, np.nan)
        self.value_right = np.full(self.shape, np.nan)

    def _generate_observation_probs(self, precision, null, reverse=False):
        """
        Generate observation probabilities P(obs | signal), with optional reverse (post-signal) mode.

        Parameters:
        - precision (float): Precision level of signal.
        - null (float): Null (uninformative) level of signal.
        - reverse (bool): If True, simulates degraded observation after signal disappers.

        Returns:
        - (list, list): Observation probabilities.
            For go-signal: p(x | d, z), where x in {left, right, null} is [0, 1, 2]
            For stop-signal: p(y | z), where y in {absent, present, null} is [0, 1, 2]
        """
        if not reverse:
            p0 = [(1 - null) * precision, (1 - null) * (1 - precision), null]
            p1 = [(1 - null) * (1 - precision), (1 - null) * precision, null]
        else:
            # Post-signal: signal disappears, residual effect remains (null influence more)
            p0 = [null * precision, null * (1 - precision), 1 - null]
            p1 = [null * (1 - precision), null * precision, 1 - null]

        return p0, p1

    def _generate_hazard(self, t_step):
        """
        Compute stop-signal onset probability (hazard) at time t_step.

        Parameters:
        - t_step (int): Time step in the trial.

        Returns:
        - float: Probability that stop signal appears at time `t_step`.
        """
        survival_prob = (1 - self.q_s) ** (t_step - 1)
        numerator = self.rate_stop_trial * survival_prob * self.lamb
        denominator = self.rate_stop_trial * \
            survival_prob + (1 - self.rate_stop_trial)

        return numerator / denominator

    def update_beta(self, beta, zeta, x):
        """
        Update belief over go signal direction (beta), given current belief about stop onset (zeta) and observation.

        Parameters:
        - beta (float): Prior belief that direction is "right" (d = R).
        - zeta (float): Belief that stop signal has already appeared (z = 1).
        - x (int): Observed direction cue (0 = left, 1 = right, 2 = noise).

        Returns:
        - float: Updated belief over right direction.
        """
        # Likelihoods: marginalize over stop signal status z
        # p(x|d=R,z=0) + p(x|d=R,z=1)
        p_dR = (1 - zeta) * self.p_x_dR_z0[x] + zeta * self.p_x_dR_z1[x]
        # p(x|d=L,z=0) + p(x|d=L,z=1)
        p_dL = (1 - zeta) * self.p_x_dL_z0[x] + zeta * self.p_x_dL_z1[x]

        numerator = beta * p_dR
        denominator = beta * p_dR + (1 - beta) * p_dL

        return numerator / denominator if denominator > 0 else beta

    def update_zeta(self, zeta, y, t_step):
        """
        Update belief over stop signal onset (zeta), given stop signal observation at time `t`.

        Parameters:
        - zeta (float): Prior belief that stop signal has occurred (z = 1).
        - y (int): Stop signal observation (0 = absent, 1 = present, 2 = noise).
        - t_step (int): Current time step.

        Returns:
        - float: Updated belief over stop onset.
        """
        # Assume subject imagine stop signal remains active since onset -> two phases only
        h = self.hazard[t_step]

        # Compute belief updates
        zeta_already = self.p_y_z1_on[y] * (zeta + (1 - zeta) * h)
        zeta_future = self.p_y_z0_pre[y] * (1 - zeta) * (1 - h)

        denominator = zeta_already + zeta_future
        return zeta_already / denominator

    def update_sigma(self, zeta, t_step):
        """
        Compute belief that current trial is a stop trial (sigma), given zeta and time.

        Parameters:
        - zeta (float): Belief over stop signal onset (z = 1).
        - t_step (int): Current time step.

        Returns:
        - float: Belief that this trial is a stop trial.
        """
        # Prior probability of being a stop trial, conditional on not having seen the stop signal yet
        numerator = (1 - self.lamb) ** t_step * self.rate_stop_trial
        denominator = numerator + (1 - self.rate_stop_trial)

        sigma_future = numerator / denominator

        # Compute updated belief
        return zeta + (1 - zeta) * sigma_future

    def immediate_cost(self, action, beta, zeta, t_step):
        """
        Compute immediate cost of taking action at time `t_step`, given current beliefs.

        Parameters:
        - action (int): 0 = left, 1 = right, 2 = wait.
        - beta (float): Belief that direction is 'right' (P(d = R)).
        - zeta (float): Belief that there is a stop signal (P(z_t = 1)) at time `t_step`.
        - t_step (int): Current time step.

        Returns:
        - float: Immediate cost of the action.
        """
        # Convert zeta to sigma
        sigma = self.update_sigma(zeta, t_step)

        # Cost from elapsing time
        time_cost = self.cost_time * t_step

        # Expected cost of go error (action != true direction, only on go trials)
        if action in [0, 1]:  # only left/right incurs go error cost
            p_error = beta if action == 0 else 1 - beta  # wrong side
            go_error_cost = (1 - sigma) * p_error * self.cost_go_error
        else:
            go_error_cost = 0

        # Expected cost of violating stop instruction
        # (i.e., making any response when it's a stop trial)
        stop_error_cost = sigma * \
            self.cost_stop_error if action in [0, 1] else 0

        return time_cost + go_error_cost + stop_error_cost

    def terminal_cost(self, zeta):
        """
        Compute terminal cost at t_step = horizon - 1 if agent keeps waiting (no response).

        Parameters:
        - zeta (float): Belief that trial is a stop trial.

        Returns:
        - float: Final cost for not responding.
        """
        # Convert zeta to sigma
        sigma = self.update_sigma(zeta, self.horizon - 1)

        # Time cost
        time_cost = self.cost_time * (self.horizon - 1)

        # If agent waits until the end and it's not a stop trial: "go missing"
        go_missing_cost = (1 - sigma) * self.cost_go_missing

        return time_cost + go_missing_cost

    def p_trans_x_(self, beta, x, z):
        """
        Compute the observation likelihood P(x | beta, z), where x is the observed go-signal cue.

        Parameters:
        - beta (float or ndarray): Belief that the go signal is "right" (P(d = R)).
        - x (int): Observed cue (0 = left, 1 = right, 2 = null).
        - z (int): Stop-signal status (0 = not yet appeared, 1 = appeared).

        Returns:
        - float or ndarray: Marginalized observation probability P(x | beta, z),
                            computed as a weighted average over hidden states d in {L, R}.
        """
        if z == 0:
            return beta * self.p_x_dR_z0[x] + (1 - beta) * self.p_x_dL_z0[x]
        elif z == 1:
            return beta * self.p_x_dR_z1[x] + (1 - beta) * self.p_x_dL_z1[x]

    def p_trans_y_(self, zeta, y, t_step, z):
        """
        Compute the joint transition probability P(y | z) · P(z | zeta),
        representing the likelihood of observing stop-signal cue `y` at time `t_step`.

        Parameters:
        - zeta (float or ndarray): Belief that stop signal has already occurred (P(z = 1)).
        - y (int): Observed stop signal (0 = absent, 1 = present, 2 = noise).
        - t_step (int): Current time step.
        - z (int): Latent stop onset state to evaluate (0 = pre-stop, 1 = post-stop).

        Returns:
        - float or ndarray: Transition probability for P(y | z) · P(z | zeta)
        """
        # Hazard probability
        hazard_prob = self.hazard[t_step + 1]

        if z == 0:
            return (1 - zeta) * (1 - hazard_prob) * self.p_y_z0_pre[y]
        elif z == 1:
            # Approximate: assume subject treats stop signal as sustained active once triggered
            return (zeta + (1 - zeta) * hazard_prob) * self.p_y_z1_on[y]

    # def future_value_tensor_(self, beta, zeta, t_step):
    #     """
    #     Retrieve the future value matrix V_{t+1}(beta, zeta) indexed by belief state pairs.

    #     This function supports matrix-level value iteration for the "wait" action,
    #     using discretized and symmetric-indexed belief states.

    #     Parameters:
    #     - beta (ndarray): Belief over go-signal direction (shape: [81, 3])
    #     - zeta (ndarray): Belief over stop-signal onset (shape: [81, 3])
    #     - t_step (int): Current time step t (to access V_{t+1})

    #     Returns:
    #     - ndarray: Future value matrix with shape (81, 81, 3, 1)
    #     """
    #     zeta_idx = (zeta * self.bins).astype(int)  # (81, 3) -> (1,81,1,3)

    #     # Center belief around 0.5 (symmetric discretization for beta)
    #     # (81, 81, 3)
    #     beta_idx = np.where(beta < 0.5,
    #                         self.bins - ((1 - beta) * self.bins).astype(int),
    #                         (beta * self.bins).astype(int))
    #     # (81, 81, 3, 3) -> (81, 81, 3, 1)
    #     return self.value[t_step,
    #                       beta_idx[:, :, :, None],
    #                       zeta_idx[None, :, None, :]]

    def future_value_tensor_(self, beta, zeta, t_step):
        """
        Bilinear-interpolate V_{t+1}(beta, zeta) on the grid and return (nbeta, nzeta, 3, 3)
        where the last two axes correspond to the 3 beta-branches × 3 zeta-branches.
        beta:  (nbeta, nzeta, 3)  belief after transition for beta-branches
        zeta:  (nzeta, 3)         belief after transition for zeta-branches
        """
        V = self.value[t_step]                          # (bins+1, bins+1)
        nbeta = beta.shape[0]
        nzeta = zeta.shape[0]

        # --- coords in [0, bins] ---
        B = np.clip(beta, 0.0, 1.0) * self.bins        # (nbeta, nzeta, 3)
        Z = np.clip(zeta, 0.0, 1.0) * self.bins       # (nzeta, 3)

        # expand to (nbeta, nzeta, 3, 1) and (1, nzeta, 1, 3) for broadcasting
        B = B[..., None]                                # (nbeta, nzeta, 3, 1)
        Z = Z[None, :, None, :]                         # (1, nzeta, 1, 3)

        # cell corners (indices)
        b0 = np.floor(B).astype(int)
        z0 = np.floor(Z).astype(int)
        b1 = np.clip(b0 + 1, 0, self.bins)
        z1 = np.clip(z0 + 1, 0, self.bins)

        # weights
        wb = B - b0                                     # in [0,1)
        wz = Z - z0

        # four-corner values -> broadcast to (nbeta, nzeta, 3, 3)
        V00 = V[b0, z0]
        V01 = V[b0, z1]
        V10 = V[b1, z0]
        V11 = V[b1, z1]

        # bilinear interpolation, result (nbeta, nzeta, 3, 3)
        Vinterp = ((1 - wb) * (1 - wz) * V00 +
                   (1 - wb) * wz * V01 +
                   wb * (1 - wz) * V10 +
                   wb * wz * V11)

        return Vinterp

    def value_iteration_tensor(self):
        """
        Perform value iteration using full tensorized belief space.

        This method computes the optimal policy and value function over time by iteratively
        updating state-action values backward from the final trial step (t = horizon - 1).

        It accounts for:
        - Per-step immediate cost (for all actions)
        - Future value under 'wait' action based on belief updates and transition probabilities
        - Uncertainty in both go-signal direction and stop-signal onset
        """
        # Initialize: terminal value at t = T-1: only cost of doing nothing (wait until end)
        self.value[-1, :, :] = self.terminal_cost(self.zeta_space)

        # Backward iteration over time steps
        for t_step in range(self.horizon - 2, -1, -1):
            # Precompute zeta index
            zeta_idx = (self.zeta_space * self.bins).astype(int)

            # ----------------------
            # Belief state updates
            # ----------------------
            # zeta(t+1) under all y. new_zeta.shape -> (81,3)
            new_zeta = np.array([[self.update_zeta(zeta, y, t_step)
                                  for y in range(3)] for zeta in self.zeta_space])

            # beta(t+1) under all x. new_beta.shape -> (81,81,3)
            new_beta = np.array([[[self.update_beta(beta, zeta, x)
                                   for x in range(3)]
                                  for zeta in self.zeta_space]
                                 for beta in self.beta_space])

            # ----------------------
            # Transition probabilities P(x, y | beta, zeta)
            # ----------------------
            # Marginalize over z in {0,1}
            p_trans_x_z0 = np.array([[self.p_trans_x_(beta, x, 0)
                                     for x in range(3)] for beta in self.beta_space])
            p_trans_y_z0 = np.array([[self.p_trans_y_(zeta, y, t_step, 0)
                                     for y in range(3)] for zeta in self.zeta_space])
            p_trans_z0 = np.einsum('ik,jl->ijkl', p_trans_x_z0, p_trans_y_z0)

            p_trans_x_z1 = np.array([[self.p_trans_x_(beta, x, 1)
                                     for x in range(3)] for beta in self.beta_space])
            p_trans_y_z1 = np.array([[self.p_trans_y_(zeta, y, t_step, 1)
                                     for y in range(3)] for zeta in self.zeta_space])
            p_trans_z1 = np.einsum('ik,jl->ijkl', p_trans_x_z1, p_trans_y_z1)

            p_trans = p_trans_z0 + p_trans_z1  # p_trans.shape -> (81,81,3,3)

            # ----------------------
            # Compute expected value of WAIT action
            # ----------------------
            # future_values.shape -> (81,81,3,3)
            future_value = self.future_value_tensor_(
                new_beta, new_zeta, t_step + 1)

            # expected_value.shape -> (81,81)
            expected_value = (p_trans * future_value).sum(axis=(-1, -2))

            # value_wait.shape at t_step -> (81,81)
            self.value_wait[t_step, np.arange(
                self.bins + 1)[:, None], zeta_idx] = expected_value

            # ------------------
            # Compute costs for LEFT and RIGHT actions
            # ----------------------
            action_costs = np.array([[[self.immediate_cost(action, beta, zeta, t_step)
                                     for zeta in self.zeta_space]
                                     for beta in self.beta_space]
                                     for action in [0, 1]])
            self.value_left[t_step] = action_costs[0]
            self.value_right[t_step] = action_costs[1]

            # ----------------------
            # Select optimal action
            # ----------------------
            # all_action_values = np.stack([
            #     self.value_left[t_step],
            #     self.value_right[t_step],
            #     self.value_wait[t_step]
            # ])
            # min_values = np.min(all_action_values, axis=0)
            # best_action = np.argmin(all_action_values, axis=0)

            # # Tie-breaking: prefer wait if left == right
            # tie_counts = np.sum(all_action_values == min_values, axis=0)
            # best_action[tie_counts > 1] = 2

            # self.value[t_step] = min_values
            # self.policy[t_step] = best_action

            ##############################
            all_action_values = np.stack(
                [self.value_left[t_step], self.value_right[t_step], self.value_wait[t_step]])

            minv = np.min(all_action_values, axis=0)
            tol = 1e-9
            is_best = np.isclose(
                all_action_values, minv[None, ...], atol=tol, rtol=0)
            best = np.argmin(all_action_values, axis=0)

            # L & R tie without Wait -> use beta side
            lr_tie = (is_best[0] & is_best[1]) & (~is_best[2])
            beta_grid = self.beta_space[:, None]
            best[(beta_grid < 0.5) & lr_tie] = 0
            best[(beta_grid > 0.5) & lr_tie] = 1

            # Only if Wait is also best, set Wait
            best[(is_best[2]) & (is_best.sum(axis=0) > 1)] = 2

            self.value[t_step] = minv
            self.policy[t_step] = best


###################### Simulation ######################

    def simu_trial(self, true_go_state, true_stop_state, ssd, verbose=False):
        def generate_obs_sequences(true_go_state, true_stop_state, ssd):
            if true_stop_state == 'stop': # stop trial
                if ssd is None or np.isnan(ssd):
                    raise ValueError("SSD must be specified for stop trials.")
                ssd = int(ssd)

                y_pre = np.random.choice([0, 1, 2], ssd, p=self.p_y_z0_pre)
                y_on = np.random.choice([0, 1, 2], mu, p=self.p_y_z1_on)
                y_post = np.random.choice([0, 1, 2], max(
                    horizon - ssd - mu, 0), p=self.p_y_z0_post)
                y_seq = np.concatenate((y_pre, y_on, y_post))

                p_x_z0, p_x_z1 = {"left": (self.p_x_dL_z0, self.p_x_dL_z1),
                                  "right": (self.p_x_dR_z0, self.p_x_dR_z1)}[true_go_state]
                x_pre = np.random.choice([0, 1, 2], ssd, p=p_x_z0)
                x_on_post = np.random.choice(
                    [0, 1, 2], horizon - ssd, p=p_x_z1)
                x_seq = np.concatenate((x_pre, x_on_post))
            elif true_stop_state == 'nonstop': # go trial
                if ssd is not None and not np.isnan(ssd):
                    raise ValueError("SSD must be None or NaN for go trials.")

                y_seq = np.random.choice([0, 1, 2], horizon, p=self.p_y_z0_pre)
                x_seq = np.random.choice([0, 1, 2], horizon,
                                         p={"left": self.p_x_dL_z0,
                                            "right": self.p_x_dR_z0}[true_go_state])
            return x_seq, y_seq, ssd

        def get_action_and_qvals(beta, sigma, t_step):
            beta_idx = np.searchsorted(self.beta_space, beta, side="left")
            zeta_idx = np.searchsorted(self.zeta_space, sigma, side="left")
            idx = (t_step, beta_idx, zeta_idx)

            v_left = self.value_left[idx]
            v_right = self.value_right[idx]
            v_wait = self.value_wait[idx]

            q_vals = np.array([v_left, v_right, v_wait])
            q_vals = -q_vals  # cost
            q_vals -= np.max(q_vals)
            probs = np.exp(self.inv_temp * q_vals)
            probs /= np.sum(probs)
            action = np.random.choice([0, 1, 2], p=probs)

            return action, q_vals, (v_left, v_right, v_wait)

        # --- Constants ---
        horizon = self.horizon
        mu = 12  # stop signal duration in time steps
        x_seq, y_seq, ssd = generate_obs_sequences(
            true_go_state, true_stop_state, ssd)

        # --- Initial beliefs ---
        beta, zeta, sigma = 0.5, 0.0, self.update_sigma(0.0, 0)
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

            zeta = self.update_zeta(zeta, y, t_step)
            sigma = self.update_sigma(zeta, t_step)
            beta = self.update_beta(beta, zeta, x)

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

    def simu_trial_batch(self, true_go_state, true_stop_state, ssd, batch_size, verbose=False):
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
            self.simu_trial(true_go_state, true_stop_state,
                            ssd, verbose=verbose)
            for _ in range(batch_size)
        ]
        return trials

    def simu_task(self):
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
                    ssd_step = min(ssd_step + 2, 36)
                elif last_stop_result == 'SE':
                    ssd_step = max(ssd_step - 2, 2)
                ssd = ssd_step
            else:
                ssd = None

            trial_outcome = self.simu_trial(
                true_go_state, true_stop_state, ssd)
            outcomes.append(trial_outcome)

            if true_stop_state == 'stop':
                last_stop_result = trial_outcome[0]

        return outcomes

###################### Visualization ######################

    def plot_simu_trial_batch_belief_5conditions(self, out_go, out_stop, show_cnt=5, saveto=None):

        colors = {
            "beta": "#4E7C59",
            "sigma": "#B75D69",
            "Q_L": "#E3B23C",
            "Q_R": "#3A6EA5",
            "Q_W": "#7A7A7A",
        }

        df_go = pd.DataFrame(out_go)
        df_stop = pd.DataFrame(out_stop)
        ssd = df_stop.iloc[0]['ssd'] if len(df_stop) > 0 else None

        df_gs = df_go[df_go['result'] == 'GS']
        df_ge = df_go[df_go['result'] == 'GE']
        df_gm = df_go[df_go['result'] == 'GM']
        df_ss = df_stop[df_stop['result'] == 'SS']
        df_se = df_stop[df_stop['result'] == 'SE']

        df_list = [df_gs, df_ge, df_gm, df_ss, df_se]
        condition_names = [
            'Go Success',
            'Go Error',
            'Go Missing',
            'Stop Success',
            'Stop Error'
        ]

        fig, ax = plt.subplots(2, 3, figsize=(12, 7))
        x_range = np.arange(1, 41)
        xticks = [1, 10, 20, 30, 40]

        for i, df in enumerate(df_list):
            row, col = divmod(i, 3)
            axis = ax[row, col]

            if len(df) == 0:
                axis.axis('off')
                continue

            # Compute means and SEM
            beta_mean = df['beta_seq'].apply(lambda x: pd.Series(x)).mean()
            beta_sem = df['beta_seq'].apply(lambda x: pd.Series(x)).sem()
            sigma_mean = df['sigma_seq'].apply(lambda x: pd.Series(x)).mean()
            sigma_sem = df['sigma_seq'].apply(lambda x: pd.Series(x)).sem()

            # Plot main curves
            axis.plot(x_range, beta_mean, label=r'$\beta^\nu_t$',
                      color=colors["beta"], linewidth=1.5)
            axis.fill_between(x_range, beta_mean - beta_sem, beta_mean + beta_sem,
                              color=colors["beta"], alpha=0.5)

            axis.plot(x_range, sigma_mean, label=r'$\sigma^\nu_t$',
                      color=colors["sigma"], linewidth=1.5)
            axis.fill_between(x_range, sigma_mean - sigma_sem, sigma_mean + sigma_sem,
                              color=colors["sigma"], alpha=0.5)

            # Plot individual samples
            for j in range(min(show_cnt, len(df))):
                axis.plot(x_range, df.iloc[j]['beta_seq'],
                          color=colors["beta"], linewidth=0.8, alpha=0.6)
                axis.plot(x_range, df.iloc[j]['sigma_seq'],
                          color=colors["sigma"], linewidth=0.8, alpha=0.6)

            # Add SSD marker lines
            if condition_names[i].startswith('Stop') and ssd is not None:
                axis.axvline(x=ssd + 1, color='k', linestyle='--', linewidth=1)
                axis.axvline(x=ssd + 12 + 1, color='k',
                             linestyle='--', linewidth=1)

            axis.set_xlim(1, 40)
            axis.set_ylim(0, 1)
            axis.set_xticks(xticks)
            axis.set_yticks(np.linspace(0, 1, 5))

            axis.set_xlabel("Time Step", fontsize=12)
            axis.set_ylabel(f"Belief State", fontsize=12)
            axis.set_title(f"{condition_names[i]}", pad=12, fontsize=14)

            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(axis='both', which='both',
                             direction='out', length=3, width=0.9)

        # Hide unused subplot
        ax[1, 2].axis("off")

        # plt.tight_layout()
        plt.subplots_adjust(
            hspace=0.5,
            wspace=0.3,
            top=0.92,
            bottom=0.08,
            left=0.08,
            right=0.98
        )

        # Shared legend
        handles, labels = ax[0, 0].get_legend_handles_labels()
        dummy = Line2D([], [], color='none', label='Belief State:')
        handles = [dummy] + handles
        labels = ['Belief State'] + labels
        fig.legend(
            handles, labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.05),
            ncol=len(handles),
            frameon=False,
            fontsize=12
        )
        # Save or show
        if saveto is not None:
            # Make sure layout is applied before saving
            fig.canvas.draw()
            fig.savefig(saveto, dpi=300, bbox_inches='tight', pad_inches=0.02)
            plt.close(fig)
        else:
            plt.show()

    def plot_simu_trial_batch_action_value_5conditions(self, out_go, out_stop, show_cnt=5, saveto=None):

        colors = {
            "beta": "#4E7C59",
            "sigma": "#B75D69",
            "Q_L": "#E3B23C",
            "Q_R": "#3A6EA5",
            "Q_W": "#7A7A7A",
        }

        df_go = pd.DataFrame(out_go)
        df_stop = pd.DataFrame(out_stop)
        ssd = df_stop.iloc[0]['ssd'] if len(df_stop) > 0 else None

        df_gs = df_go[df_go['result'] == 'GS']
        df_ge = df_go[df_go['result'] == 'GE']
        df_gm = df_go[df_go['result'] == 'GM']
        df_ss = df_stop[df_stop['result'] == 'SS']
        df_se = df_stop[df_stop['result'] == 'SE']

        df_list = [df_gs, df_ge, df_gm, df_ss, df_se]
        condition_names = [
            'Go Success',
            'Go Error',
            'Go Missing',
            'Stop Success',
            'Stop Error'
        ]

        fig, ax = plt.subplots(2, 3, figsize=(12, 7))
        x_range = np.arange(1, 41)
        xticks = [1, 10, 20, 30, 40]

        for i, df in enumerate(df_list):
            row, col = divmod(i, 3)
            axis = ax[row, col]

            if len(df) == 0:
                axis.axis('off')
                continue

            # Mean & SEM
            left_mean = df['value_left_seq'].apply(
                lambda x: pd.Series(x)).mean()
            left_sem = df['value_left_seq'].apply(lambda x: pd.Series(x)).sem()
            right_mean = df['value_right_seq'].apply(
                lambda x: pd.Series(x)).mean()
            right_sem = df['value_right_seq'].apply(
                lambda x: pd.Series(x)).sem()
            wait_mean = df['value_wait_seq'].apply(
                lambda x: pd.Series(x)).mean()
            wait_sem = df['value_wait_seq'].apply(lambda x: pd.Series(x)).sem()

            # Main value lines
            axis.plot(x_range, left_mean, label=r'$Q^\nu_t(b^\nu_t, L)$',
                      color=colors["Q_L"], linewidth=1.5)
            axis.fill_between(x_range, left_mean - left_sem,
                              left_mean + left_sem, color=colors["Q_L"], alpha=0.5)

            axis.plot(x_range, right_mean,
                      label=r'$Q^\nu_t(b^\nu_t, R)$', color=colors["Q_R"], linewidth=1.5)
            axis.fill_between(x_range, right_mean - right_sem,
                              right_mean + right_sem, color=colors["Q_R"], alpha=0.5)

            axis.plot(x_range, wait_mean, label=r'$Q^\nu_t(b^\nu_t, W)$',
                      color=colors["Q_W"], linewidth=1.5)
            axis.fill_between(x_range, wait_mean - wait_sem,
                              wait_mean + wait_sem, color=colors["Q_W"], alpha=0.5)

            # Trial samples
            for j in range(min(show_cnt, len(df))):
                axis.plot(x_range, df.iloc[j]['value_left_seq'],
                          color=colors["Q_L"], alpha=0.6, linewidth=0.8)
                axis.plot(
                    x_range, df.iloc[j]['value_right_seq'], color=colors['Q_R'], alpha=0.6, linewidth=0.8)
                axis.plot(x_range, df.iloc[j]['value_wait_seq'],
                          color=colors['Q_W'], alpha=0.6, linewidth=0.8)

            # SSD markers for stop trials
            if condition_names[i].startswith('Stop') and ssd is not None:
                axis.axvline(x=ssd + 1, color='k', linestyle='--', linewidth=1)
                axis.axvline(x=ssd + 12 + 1, color='k',
                             linestyle='--', linewidth=1)

            axis.set_xlim(1, 40)
            # axis.set_ylim(0, 2.5)
            axis.set_xticks(xticks)
            axis.set_xlabel("Time Step", fontsize=12)
            axis.set_ylabel(f"Action Value", fontsize=12)
            axis.set_title(f"{condition_names[i]}", fontsize=14, pad=12)

            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(axis='both', which='both',
                             direction='out', length=3, width=0.9)

        # Hide unused subplot
        ax[1, 2].axis("off")

        # plt.tight_layout()
        plt.subplots_adjust(
            hspace=0.5,
            wspace=0.3,
            top=0.92,
            bottom=0.08,
            left=0.08,
            right=0.98
        )

        # Legend
        handles, labels = ax[0, 0].get_legend_handles_labels()
        dummy = Line2D([], [], color='none', label='Action Value:')
        handles = [dummy] + handles
        labels = ['Action Value'] + labels
        fig.legend(
            handles, labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.05),
            ncol=len(handles),
            frameon=False,
            fontsize=12
        )        # Save or show
        if saveto is not None:
            # Make sure layout is applied before saving
            fig.canvas.draw()
            fig.savefig(saveto, dpi=300, bbox_inches='tight', pad_inches=0.02)
            plt.close(fig)
        else:
            plt.show()

    def plot_policy(self, saveto, dpi=150):
        """
        Publication-quality plot of the optimal policy over all time steps.
        """
        colors = {
            "beta": "#4E7C59",
            "sigma": "#B75D69",
            "Q_L": "#E3B23C",
            "Q_R": "#3A6EA5",
            "Q_W": "#7A7A7A",
        }

        policy = self.policy[:-1, :, :].copy()
        num_timesteps = policy.shape[0]

        # Mask undefined beliefs
        beta = 0.5
        zeta = 0
        for i in range(num_timesteps):
            beta = self.update_beta(beta, zeta=0, x=1)
            beta_idx = int(beta * self.bins)
            beta_idx = np.clip(beta_idx, 0, self.bins - 1)
            policy[i, beta_idx + 2:, :] = np.nan
            policy[i, :(self.bins - beta_idx) - 1, :] = np.nan

            zeta = self.update_zeta(zeta, 1, i)
            sigma = self.update_sigma(zeta, i)
            sigma_idx = int(sigma * self.bins)
            sigma_idx = np.clip(sigma_idx, 0, self.bins - 1)
            policy[i, :, sigma_idx + 2:] = np.nan

        # Define color mapping
        cmap = ListedColormap(
            [colors['Q_L'], colors['Q_R'], "lightgray", "white"])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = BoundaryNorm(bounds, cmap.N)

        # Layout grid
        cols = min(6, num_timesteps)
        rows = (num_timesteps + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5),
                                 dpi=dpi, constrained_layout=True)
        axes = np.array(axes).reshape(-1)

        for i in range(num_timesteps):
            matrix_plot = np.where(np.isnan(policy[i]), 3, policy[i])
            ax = axes[i]
            img = ax.imshow(matrix_plot, cmap=cmap,
                            norm=norm, extent=[0, 1, 0, 1])
            ax.set_title(f"t = {i + 1}", fontsize=14)
            ax.set_xlabel(r'$\sigma^\nu_t$', fontsize=14)
            ax.set_ylabel(r'$\beta^\nu_t$', fontsize=14)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis='both', which='both',
                           direction='out', length=4, width=0.8)

        for i in range(num_timesteps, len(axes)):
            axes[i].axis("off")

        # Shared colorbar
        cbar = fig.colorbar(
            img, ax=axes, orientation='horizontal', shrink=1, aspect=50, pad=0.02)
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(["Left", "Right", "Wait", "None"])
        cbar.ax.tick_params(labelsize=13)
        cbar.ax.set_title("Action", fontsize=14, pad=10,
                          rotation=0, loc='left')

        # Optional: title
        # fig.suptitle("Optimal Policy Over Time", fontsize=16, y=1.03)

        plt.savefig(saveto, dpi=dpi, bbox_inches="tight")
        plt.show()

    def plot_policy_simple(self, saveto, dpi=150, selected_timesteps=None):
        """
        Simplified publication-quality policy plot.

        Parameters:
            selected_timesteps: list of int, time step indices to plot.
        """
        colors = {
            "beta": "#4E7C59",
            "sigma": "#B75D69",
            "Q_L": "#E3B23C",
            "Q_R": "#3A6EA5",
            "Q_W": "#7A7A7A",
        }

        policy = self.policy[:-1, :, :].copy()
        num_timesteps = policy.shape[0]

        if selected_timesteps is None:
            selected_timesteps = list(range(min(4, num_timesteps)))
        else:
            selected_timesteps = [
                t for t in selected_timesteps if 0 <= t < num_timesteps]

        # mask for policy according to belief constraints
        beta = 0.5
        zeta = 0
        for i in range(num_timesteps):
            beta = self.update_beta(beta, zeta=0, x=1)
            beta_idx = int(beta * self.bins)
            beta_idx = np.clip(beta_idx, 0, self.bins - 1)
            policy[i, beta_idx + 2:, :] = np.nan
            policy[i, :(self.bins - beta_idx) - 1, :] = np.nan

            zeta = self.update_zeta(zeta, 1, i)
            sigma = self.update_sigma(zeta, i)
            sigma_idx = int(sigma * self.bins)
            sigma_idx = np.clip(sigma_idx, 0, self.bins - 1)
            policy[i, :, sigma_idx + 2:] = np.nan

        # Set colormap
        cmap = ListedColormap(
            [colors['Q_R'], colors['Q_L'], "lightgray", "white"])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = BoundaryNorm(bounds, cmap.N)

        cols = min(4, len(selected_timesteps))
        rows = (len(selected_timesteps) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5),
                                 dpi=dpi, constrained_layout=True)
        axes = np.array(axes).reshape(-1)

        for i, t in enumerate(selected_timesteps):
            matrix_plot = np.where(np.isnan(policy[t]), 3, policy[t])
            ax = axes[i]
            img = ax.imshow(matrix_plot, cmap=cmap,
                            norm=norm, extent=[0, 1, 0, 1])

            ax.set_title(f"t = {t + 1}", fontsize=14)
            ax.set_xlabel(r'$\sigma^\nu_t$', fontsize=14)
            ax.set_ylabel(r'$\beta^\nu_t$', fontsize=14)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis='both', which='both',
                           direction='out', length=4, width=0.8)

        for i in range(len(selected_timesteps), len(axes)):
            axes[i].axis("off")

        cbar = fig.colorbar(
            img, ax=axes, orientation='vertical', shrink=0.8, pad=0.02)
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(["Right", "Left", "Wait", "None"])
        cbar.ax.tick_params(labelsize=13)
        cbar.ax.set_title("Action", fontsize=14, pad=10,
                          rotation=0, loc='left')
        plt.savefig(saveto, dpi=dpi, bbox_inches="tight")
        plt.show()

###################### Viz Across-trial ######################

    def _pad_to_matrix(self, list_of_seqs, t_max):
        import numpy as np
        arr = np.full((len(list_of_seqs), t_max), np.nan, dtype=float)
        for i, s in enumerate(list_of_seqs):
            L = min(len(s), t_max)
            arr[i, :L] = np.asarray(s[:L], dtype=float)
        return arr

    def _mean_sem_over_trials(self, mat):
        import numpy as np
        mean = np.nanmean(mat, axis=0)
        n_eff = np.sum(~np.isnan(mat), axis=0).astype(float)
        std = np.nanstd(mat, axis=0, ddof=1)
        sem = np.divide(std, np.sqrt(np.maximum(n_eff, 1.0)),
                        out=np.zeros_like(std), where=n_eff > 1)
        return mean, sem

    def _simulate_mixed_session(self, n_trials=2000, ssd=12):
        import numpy as np
        import pandas as pd
        out = []
        for _ in range(n_trials):
            is_stop = np.random.rand() < self.rate_stop_trial
            gs = np.random.choice(["left", "right"])
            if is_stop:
                r = self.simu_trial(gs, "stop", ssd, verbose=False)
            else:
                r = self.simu_trial(gs, "nonstop", None, verbose=False)
            out.append(r)
        return pd.DataFrame(out, columns=["result", "rt", "ssd"])

    def _summarize_behavior_with_sem(self, df, step_in_ms=25.0):
        import numpy as np
        is_go = df["result"].isin(["GS", "GE", "GM"])
        go_df = df.loc[is_go]
        n_go = len(go_df)
        n_gs = (go_df["result"] == "GS").sum()
        go_acc = (n_gs / n_go) if n_go > 0 else np.nan
        go_acc_sem = np.sqrt(go_acc * (1.0 - go_acc) /
                             n_go) if n_go > 0 else np.nan

        is_stop = df["result"].isin(["SS", "SE"])
        stop_df = df.loc[is_stop]
        n_stop = len(stop_df)
        n_ss = (stop_df["result"] == "SS").sum()
        stop_succ = (n_ss / n_stop) if n_stop > 0 else np.nan
        stop_succ_sem = np.sqrt(
            stop_succ * (1.0 - stop_succ) / n_stop) if n_stop > 0 else np.nan

        rt_gs = df.loc[df["result"] == "GS", "rt"].to_numpy() * step_in_ms
        rt_se = df.loc[df["result"] == "SE", "rt"].to_numpy() * step_in_ms
        rt_go_succ = np.nanmean(rt_gs) if rt_gs.size > 0 else np.nan
        rt_go_succ_sem = (np.nanstd(rt_gs, ddof=1) /
                          np.sqrt(rt_gs.size)) if rt_gs.size > 1 else np.nan
        rt_stop_err = np.nanmean(rt_se) if rt_se.size > 0 else np.nan
        rt_stop_err_sem = (np.nanstd(rt_se, ddof=1) /
                           np.sqrt(rt_se.size)) if rt_se.size > 1 else np.nan

        return dict(
            go_acc=go_acc, go_acc_sem=go_acc_sem,
            stop_succ=stop_succ, stop_succ_sem=stop_succ_sem,
            rt_go_succ=rt_go_succ, rt_go_succ_sem=rt_go_succ_sem,
            rt_stop_err=rt_stop_err, rt_stop_err_sem=rt_stop_err_sem
        )

    def plot_rate_timecourses(
        self,
        params: dict,
        rates=(0.05, 1/6, 0.35, 0.55, 0.80),
        ssd=None,
        n_batch=300,
        t_max=40,
        saveto=None,
        base_color=(0.70, 0.10, 0.15),
        trial_type: str = "stop",   # "stop" or "nonstop"
    ):
        """
        Across-rate within-trial dynamics.

        If trial_type == "stop":
            - Simulate stop trials with the given SSD (if None, default to 10).
            - Left: mean sigma_t (Stop belief), colored by stop-rate (darker = higher rate).
            - Right: gap Q(W) - min(Q(L), Q(R)), gray gradient, outcome styles SS(solid)/SE(dashed).
            - SSD markers (vertical) at ssd+1 and ssd+12+1 if ssd is not None.

        If trial_type == "nonstop":
            - Simulate go-only (nonstop) trials; SSD is ignored.
            - Left/Right panels as above; outcome styles GS(solid)/GE(dashed)/GM(dot-dash).
            - No SSD markers.

        Note:
            Color encodes the across-trial stop probability r~ even for nonstop,
            because it still shapes priors and value/belief dynamics.
        """
        import numpy as np
        from matplotlib.colors import to_rgb, rgb_to_hsv, hsv_to_rgb

        def desat_ramp(base="#7A1F2C", n=5, s_scale=0.55, v_low=0.42, v_high=0.88):
            b = np.array(to_rgb(base))
            h, s, v = rgb_to_hsv(b[np.newaxis, np.newaxis, :])[0, 0]
            s = np.clip(s * s_scale, 0, 1)
            vals = np.linspace(v_low, v_high, n)
            return [tuple(hsv_to_rgb([h, s, vv])) for vv in vals]

        def gray_ramp(n=5, g_low=0.18, g_high=0.70):
            return [(g, g, g) for g in np.linspace(g_high, g_low, n)]

        trial_type = str(trial_type).lower().strip()
        if trial_type not in ("stop", "nonstop"):
            raise ValueError("trial_type must be 'stop' or 'nonstop'.")

        # ---------- helpers ----------
        def _pad_to_matrix(seq_list, T):
            if len(seq_list) == 0:
                return np.full((0, T), np.nan)
            out = np.full((len(seq_list), T), np.nan)
            for i, s in enumerate(seq_list):
                if s is None:
                    continue
                arr = np.asarray(s, dtype=float)
                out[i, :min(len(arr), T)] = arr[:min(len(arr), T)]
            return out

        def _mean_sem(mat):
            mean = np.nanmean(mat, axis=0)
            n = np.sum(~np.isnan(mat), axis=0)
            std = np.nanstd(mat, axis=0, ddof=1)
            sem = np.where(n > 1, std / np.sqrt(n), np.nan)  # require n>1
            return mean, sem

        def _sigma_mean_sem(df):
            if len(df) == 0 or "sigma_seq" not in df.columns:
                return np.full(t_max, np.nan), np.full(t_max, np.nan)
            mat = _pad_to_matrix(df["sigma_seq"].tolist(), t_max)
            return _mean_sem(mat)

        def _gap_mean_sem(df):
            if len(df) == 0:
                return np.full(t_max, np.nan), np.full(t_max, np.nan)
            vl = _pad_to_matrix(df.get("value_left_seq", []).tolist(), t_max)
            vr = _pad_to_matrix(df.get("value_right_seq", []).tolist(), t_max)
            vw = _pad_to_matrix(df.get("value_wait_seq", []).tolist(), t_max)
            if vl.shape[0] == 0 or vr.shape[0] == 0 or vw.shape[0] == 0:
                return np.full(t_max, np.nan), np.full(t_max, np.nan)
            min_go = np.nanmin(np.stack([vl, vr], axis=0), axis=0)
            gap = vw - min_go
            return _mean_sem(gap)

        # ---------- simulate per rate ----------
        series = []
        for r in rates:
            m = self.__class__(rate_stop_trial=float(r), **params)
            m.value_iteration_tensor()

            # A=success, B=error, M=miss (GM only)
            rec_a, rec_b, rec_m = [], [], []
            n_iter = int(n_batch)

            for _ in range(n_iter):
                true_go = np.random.choice(["left", "right"])

                if trial_type == "stop":
                    ssd_use = 10 if ssd is None else int(ssd)
                    res = m.simu_trial(true_go, "stop", ssd_use, verbose=True)
                    outcome = res["result"] if isinstance(res, dict) else (
                        res[0] if isinstance(res, (list, tuple)) else None)
                    if outcome == "SS":
                        rec_a.append(res)
                    elif outcome == "SE":
                        rec_b.append(res)
                    # no "SM" in stop trials; keep two buckets
                else:
                    res = m.simu_trial(true_go, "nonstop", None, verbose=True)
                    outcome = res["result"] if isinstance(res, dict) else (
                        res[0] if isinstance(res, (list, tuple)) else None)
                    if outcome == "GS":
                        rec_a.append(res)
                    elif outcome == "GE":
                        rec_b.append(res)
                    elif outcome == "GM":
                        rec_m.append(res)

            df_a, df_b, df_m = pd.DataFrame(
                rec_a), pd.DataFrame(rec_b), pd.DataFrame(rec_m)

            sigma_a_mean, sigma_a_sem = _sigma_mean_sem(df_a)
            sigma_b_mean, sigma_b_sem = _sigma_mean_sem(df_b)
            sigma_m_mean, sigma_m_sem = _sigma_mean_sem(df_m)

            gap_a_mean, gap_a_sem = _gap_mean_sem(df_a)
            gap_b_mean, gap_b_sem = _gap_mean_sem(df_b)
            gap_m_mean, gap_m_sem = _gap_mean_sem(df_m)

            series.append(dict(
                rate=float(r),
                sigma_a_mean=sigma_a_mean, sigma_a_sem=sigma_a_sem,
                sigma_b_mean=sigma_b_mean, sigma_b_sem=sigma_b_sem,
                sigma_m_mean=sigma_m_mean, sigma_m_sem=sigma_m_sem,
                gap_a_mean=gap_a_mean,   gap_a_sem=gap_a_sem,
                gap_b_mean=gap_b_mean,   gap_b_sem=gap_b_sem,
                gap_m_mean=gap_m_mean,   gap_m_sem=gap_m_sem
            ))

        # ---------- plotting ----------
        T = np.arange(1, t_max + 1)
        fig, axes = plt.subplots(1, 2, figsize=(
            12, 3.5), constrained_layout=True)
        lw = 1.5
        xticks = [1, max(1, t_max//4), max(1, t_max//2),
                  max(1, 3*t_max//4), t_max]

        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        n_rates = len(series)
        sorted_rates = sorted(rates)
        rate_ranks = {r: i for i, r in enumerate(sorted_rates)}
        reds = desat_ramp("#7A1F2C", n_rates, s_scale=0.55,
                          v_low=0.42, v_high=0.88)[::-1]
        grays = gray_ramp(n_rates, g_low=0.18, g_high=0.70)

        if trial_type == "stop":
            style_a_lbl, style_b_lbl, style_m_lbl = "SS", "SE", None
        else:
            style_a_lbl, style_b_lbl, style_m_lbl = "GS", "GE", "GM"

        # ----- Left panel: sigma_t -----
        ax = axes[0]
        rate_lines = []
        for s in series:
            r = s["rate"]
            rank = rate_ranks[r]
            col = reds[rank]
            # lw = 2.2 - 0.6*rank/(n_rates-1)
            # al = 0.95 - 0.35*rank/(n_rates-1)
            lw = 1.5
            al = 0.18

            ax.plot(T, s["sigma_a_mean"], color=col, lw=lw, linestyle='-')
            ax.fill_between(T, s["sigma_a_mean"] - s["sigma_a_sem"],
                            s["sigma_a_mean"] + s["sigma_a_sem"], color=col, alpha=al)

            ax.plot(T, s["sigma_b_mean"], color=col, lw=lw, linestyle='--')
            ax.fill_between(T, s["sigma_b_mean"] - s["sigma_b_sem"],
                            s["sigma_b_mean"] + s["sigma_b_sem"], color=col, alpha=al)

            if trial_type == "nonstop":
                ax.plot(T, s["sigma_m_mean"], color=col, lw=lw, linestyle='-.')
                ax.fill_between(T, s["sigma_m_mean"] - s["sigma_m_sem"],
                                s["sigma_m_mean"] + s["sigma_m_sem"], color=col, alpha=al)

            rate_lines.append(
                Line2D([0], [0], color=col, lw=lw, label=f"{r:.2f}"))

        style_lines = [
            Line2D([0], [0], color='k', lw=lw,
                   linestyle='-',  label=style_a_lbl),
            Line2D([0], [0], color='k', lw=lw,
                   linestyle='--', label=style_b_lbl),
        ]
        if trial_type == "nonstop":
            style_lines.append(
                Line2D([0], [0], color='k', lw=lw, linestyle='-.', label=style_m_lbl))

        if trial_type == "stop" and ssd is not None:
            ax.axvline(x=int(ssd) + 1,      color='k',
                       linestyle='--', linewidth=1)
            ax.axvline(x=int(ssd) + 12 + 1, color='k',
                       linestyle='--', linewidth=1)

        legend_title = (
            "Stop Prob. " + r"$\tilde{r}^{\nu}$" + " (color)\n" +
            ("Outcome (style)" if trial_type == "nonstop" else "Outcome (style)")
        )

        leg = ax.legend(
            handles=rate_lines + style_lines,
            frameon=False,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            title=legend_title,
            fontsize=12
        )
        leg.get_title().set_fontsize(12)

        ax.set_ylim(0, 1)
        ax.set_xlim(1, t_max)
        ax.set_xticks(xticks)
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel(r"Belief State $\sigma^\nu_t$", fontsize=12)

        # ----- Right panel: gap -----
        ax = axes[1]
        rate_lines = []
        for s in series:
            r = s["rate"]
            rank = rate_ranks[r]
            col = grays[rank]
            # lw = 2.2 - 0.6*rank/(n_rates-1)
            # al = 0.95 - 0.35*rank/(n_rates-1)
            lw = 1.5
            al = 0.18

            ax.plot(T, s["gap_a_mean"], color=col, lw=lw, linestyle='-')
            ax.fill_between(T, s["gap_a_mean"] - s["gap_a_sem"],
                            s["gap_a_mean"] + s["gap_a_sem"], color=col, alpha=al)

            ax.plot(T, s["gap_b_mean"], color=col, lw=lw, linestyle='--')
            ax.fill_between(T, s["gap_b_mean"] - s["gap_b_sem"],
                            s["gap_b_mean"] + s["gap_b_sem"], color=col, alpha=al)

            if trial_type == "nonstop":
                ax.plot(T, s["gap_m_mean"], color=col, lw=lw, linestyle='-.')
                ax.fill_between(T, s["gap_m_mean"] - s["gap_m_sem"],
                                s["gap_m_mean"] + s["gap_m_sem"], color=col, alpha=al)

            rate_lines.append(
                Line2D([0], [0], color=col, lw=lw, label=f"{r:.2f}"))

        style_lines = [
            Line2D([0], [0], color='k', lw=lw,
                   linestyle='-',  label=style_a_lbl),
            Line2D([0], [0], color='k', lw=lw,
                   linestyle='--', label=style_b_lbl),
        ]
        if trial_type == "nonstop":
            style_lines.append(
                Line2D([0], [0], color='k', lw=lw, linestyle='-.', label=style_m_lbl))

        if trial_type == "stop" and ssd is not None:
            ax.axvline(x=int(ssd) + 1,      color='k',
                       linestyle='--', linewidth=1)
            ax.axvline(x=int(ssd) + 12 + 1, color='k',
                       linestyle='--', linewidth=1)

        leg = ax.legend(
            handles=rate_lines + style_lines,
            frameon=False,
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            title=legend_title,
            fontsize=12
        )
        leg.get_title().set_fontsize(12)

        ax.set_xlim(1, t_max)
        ax.set_xticks(xticks)
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel(
            r"$Q^\nu_t(W) - \min(Q^\nu_t(L), Q^\nu_t(R))$", fontsize=12)

        plt.tight_layout()
        if saveto:
            fig.savefig(saveto, dpi=300, bbox_inches='tight', pad_inches=0.03)
            plt.show()
            plt.close(fig)
        else:
            plt.show()

    def plot_rate_behavior(
        self,
        params: dict,
        rates=(0.05, 0.10, 1/6, 0.30, 0.50),
        ssd=10,
        n_session_trials=1000,
        step_in_ms=25.0,
        saveto=None,
        rt_ylim_ms=(0, 1000),  # set None to auto-scale
    ):
        """
        Two-panel behavioral summary across stop rates:
        Left: Accuracy (Go & Stop) with SEM
        Right: RT (Go-success & Stop-error) [ms] with SEM

        Styling: all black, mean±SEM, no top/right spines.
        """

        # --- simulate sessions at each rate ---
        rows = []
        for r in rates:
            m = self.__class__(rate_stop_trial=float(r), **params)
            m.value_iteration_tensor()
            df = m._simulate_mixed_session(n_trials=n_session_trials, ssd=ssd)
            stats = m._summarize_behavior_with_sem(df, step_in_ms=step_in_ms)
            rows.append(dict(rate=float(r), **stats))
        beh = pd.DataFrame(rows).sort_values("rate")

        # --- figure ---
        fig, axes = plt.subplots(1, 2, figsize=(
            12, 3.5), constrained_layout=True)
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", which="both",
                           direction="out", length=3, width=0.9)

        # common kwargs
        k_go = dict(color="black", marker="o", linestyle="-",
                    linewidth=1.6, markersize=5, capsize=3)
        k_stop = dict(color="black", marker="o", linestyle="--",
                      linewidth=1.6, markersize=5, capsize=3)

        # ----- LEFT: Accuracy -----
        ax = axes[0]
        ax.errorbar(beh["rate"], beh["go_acc"],
                    yerr=beh["go_acc_sem"],   label="Go",   **k_go)
        ax.errorbar(beh["rate"], beh["stop_succ"],
                    yerr=beh["stop_succ_sem"], label="Stop", **k_stop)
        ax.set_xlabel(r"Stop Probability $\tilde{r}^{\nu}$", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_ylim(0, 1)
        # ax.legend(frameon=False, ncol=2)
        ax.legend(
            frameon=False,
            ncol=1,
            loc='upper left',
            bbox_to_anchor=(1.02, 1.0)
        )

        # ----- RIGHT: RTs -----
        ax = axes[1]
        ax.errorbar(beh["rate"], beh["rt_go_succ"],
                    yerr=beh["rt_go_succ_sem"],  label="Go (Success)",  **k_go)
        ax.errorbar(beh["rate"], beh["rt_stop_err"],
                    yerr=beh["rt_stop_err_sem"], label="Stop (Error)", **k_stop)
        ax.set_xlabel(r"Stop Probability $\tilde{r}^{\nu}$", fontsize=12)
        ax.set_ylabel("Reaction Time (ms)", fontsize=12)
        if rt_ylim_ms is None:
            ymax = np.nanmax([beh["rt_go_succ"], beh["rt_stop_err"]]) * 1.1
            ax.set_ylim(0, max(600, float(ymax)))
        else:
            ax.set_ylim(*rt_ylim_ms)
        # ax.legend(frameon=False, ncol=1)
        ax.legend(
            frameon=False,
            ncol=1,
            loc='upper left',
            bbox_to_anchor=(1.02, 1.0)
        )

        # tidy
        for ax in axes:
            ax.margins(x=0.05)

        # save or show
        if saveto is not None:
            fig.canvas.draw()
            fig.savefig(saveto, dpi=300, bbox_inches="tight", pad_inches=0.02)
            plt.show()
            plt.close(fig)
        else:
            plt.show()

        return saveto, beh
