# ABCD SST
import numpy as np

class POMDP:
    """
    Partially Observable Markov Decision Process (POMDP) for the Stop Signal Task (SST).

    This model simulates within-trial belief dynamics under uncertainty about sensory inputs
    and trial type. It supports dynamic hazard modeling and cost-sensitive decision policy.
    """

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
            # Determine optimal action and value
            # ----------------------
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



# =============================================================================
# Helper Class: Structural Independence
# =============================================================================

class POMDP_Independent(POMDP):
    """
    [Structural Helper]
    Inherits from the Base POMDP but removes the Dependency.
    In this class, Stop Signal does NOT mask Go Signal.
    """
    def _setup_observation_distributions(self):
        """Override: Enforce Independence (reverse=False even for z=1)."""
        # Go Signal: z=0 (Normal)
        self.p_x_dL_z0, self.p_x_dR_z0 = self._generate_observation_probs(
            self.q_d, self.q_d_n, reverse=False)
        
        # Go Signal: z=1 (Stop) -> REMAINS NORMAL (Independent)
        self.p_x_dL_z1, self.p_x_dR_z1 = self._generate_observation_probs(
            self.q_d, self.q_d_n, reverse=False) 

        # Stop Signal (Standard)
        self.p_y_z0_pre, self.p_y_z1_pre = self._generate_observation_probs(self.q_s, self.q_s_n)
        self.p_y_z0_on, self.p_y_z1_on   = self._generate_observation_probs(self.q_s, self.q_s_n)
        self.p_y_z0_post, self.p_y_z1_post = self._generate_observation_probs(self.q_s, self.q_s_n, reverse=True)


# =============================================================================
# The 5 Models for Ablation Study
# =============================================================================

# --- M4: The Full Model (POMDP) ---
# Structure: Dependent
# Perception: Free (q_n fitted)
# Policy: Softmax
class M4_Full(POMDP):
    """
    Alias for the original POMDP class. 
    This is the proposed model with all features enabled.
    """
    pass


# --- M3: Rigid Dependent (Structure Only) ---
# Structure: Dependent (Masking works)
# Perception: Fixed (q_n = 0)
# Policy: Deterministic (Argmax)
class M3_Rigid_Dependent(POMDP):
    def __init__(self, q_d, q_s, cost_go_error, cost_go_missing,
                 cost_stop_error, cost_time, **kwargs):
        """
        Forces q_d_n=0, q_s_n=0. Ignores inv_temp.
        """
        # Call parent with fixed null params and dummy inv_temp
        kwargs.pop('q_d_n', None)
        kwargs.pop('q_s_n', None)
        kwargs.pop('inv_temp', None)

        super().__init__(q_d_n=1e-5, q_d=q_d, 
                         q_s_n=1e-5, q_s=q_s,
                         cost_go_error=cost_go_error, 
                         cost_go_missing=cost_go_missing,
                         cost_stop_error=cost_stop_error, 
                         cost_time=cost_time, 
                         inv_temp=1.0, # Dummy, won't be used due to override below
                         **kwargs)

    def get_action_and_qvals(self, beta, sigma, t_step):
        """Override: Deterministic Policy (Argmax), ignoring inv_temp."""
        # Calculate Q-values same as parent
        b_idx = int(round(beta * self.bins))
        s_idx = int(round(sigma * self.bins))
        
        q_wait = self.value_wait[t_step, b_idx, s_idx]
        q_left = self.value_left[t_step, b_idx, s_idx]
        q_right = self.value_right[t_step, b_idx, s_idx]
        q_vals = np.array([q_left, q_right, q_wait])

        # Deterministic Choice
        action = np.argmax(q_vals)
        
        # Mock probabilities (1.0 for chosen, 0.0 for others) for API consistency
        probs = np.zeros(3)
        probs[action] = 1.0
        
        return action, q_vals, probs


# --- M2: Flexible Independent (The Strong Competitor) ---
# Structure: Independent
# Perception: Free (q_n fitted)
# Policy: Softmax
class M2_FullParam_Independent(POMDP_Independent):
    """
    Inherits Independent structure.
    Keeps Softmax and Free Parameters from base logic.
    """
    pass


# --- M1: Noisy Independent ---
# Structure: Independent
# Perception: Fixed (q_n = 0)
# Policy: Softmax
class M1_Softmax_Independent(POMDP_Independent):
    def __init__(self, q_d, q_s, cost_go_error, cost_go_missing,
                 cost_stop_error, cost_time, inv_temp, **kwargs):
        """
        Forces q_d_n=0, q_s_n=0. Keeps Softmax (inv_temp).
        """
        kwargs.pop('q_d_n', None)
        kwargs.pop('q_s_n', None)
        super().__init__(q_d_n=1e-5, q_d=q_d, 
                         q_s_n=1e-5, q_s=q_s,
                         cost_go_error=cost_go_error, 
                         cost_go_missing=cost_go_missing,
                         cost_stop_error=cost_stop_error, 
                         cost_time=cost_time, 
                         inv_temp=inv_temp, 
                         **kwargs)


# --- M0: The Baseline (Yu et al. 2011 style) ---
# Structure: Independent
# Perception: Fixed (q_n = 0)
# Policy: Deterministic
class M0_Baseline(POMDP_Independent):
    def __init__(self, q_d, q_s, cost_go_error, cost_go_missing,
                 cost_stop_error, cost_time, **kwargs):
        """
        Forces q_d_n=0, q_s_n=0. Ignores inv_temp.
        """
        kwargs.pop('q_d_n', None)
        kwargs.pop('q_s_n', None)
        kwargs.pop('inv_temp', None)
        kwargs.pop('rate_stop_trial', None)  # Use default from base class

        super().__init__(q_d_n=1e-5, q_d=q_d, 
                         q_s_n=1e-5, q_s=q_s,
                         cost_go_error=cost_go_error, 
                         cost_go_missing=cost_go_missing,
                         cost_stop_error=cost_stop_error, 
                         cost_time=cost_time, 
                         inv_temp=1.0, # Dummy
                         **kwargs)

    def get_action_and_qvals(self, beta, sigma, t_step):
        """Override: Deterministic Policy (Argmax)."""
        # (Same logic as M3, but applied to Independent structure)
        b_idx = int(round(beta * self.bins))
        s_idx = int(round(sigma * self.bins))
        q_vals = np.array([
            self.value_left[t_step, b_idx, s_idx],
            self.value_right[t_step, b_idx, s_idx],
            self.value_wait[t_step, b_idx, s_idx]
        ])
        action = np.argmax(q_vals)
        probs = np.zeros(3)
        probs[action] = 1.0
        return action, q_vals, probs