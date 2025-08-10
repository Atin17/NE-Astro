import numpy as np

# -------------------------------------
# Environment Code: Generate trial observations
# -------------------------------------
class MouseBehaviorEnvTemporalReward:
    """
    Multi-phase environment that does NOT end immediately after press/no-press.
    The timeline is:
      1) Fixation ON (1000 ms)
      2) Random delay (~650 ms)
      3) Sound (500 ms)
      4) Response window (800 ms) => fixation=0, track if agent pressed or not
      5) Post-outcome delay (250 ms) => no immediate reward
      6) Reward window (100 ms) if final action is correct => +1 each step
         or Punishment window (300 ms) if incorrect => -1 each step
      7) ITI (4000 ms)
    The entire trial then ends.

    Observations (shape: (T,4)):
      obs[t,0] = LED dimension (1 during fixation LED, else 0)
      obs[t,1] = freq dimension (+1 if go tone, -1 if no-go tone, else 0)
      obs[t,2] = intensity dimension
      obs[t,3] = fixation dimension (1=ON => don't press, 0=OFF => can press)

    Actions:
      0 => HOLD
      1 => PRESS

    Rewards:
      - Pressing while fixation=1 => negative partial reward (e.g., -0.25) each time
        but does NOT end the trial.
      - End of response window => final action = "press" if agent pressed at least once,
        else "hold."
        * If final action is correct => in the reward phase (100 ms),
          each time step gets +1.0
        * If final action is incorrect => in the punishment phase (300 ms),
          each time step gets -1.0
      - The trial ends after ITI.

    The environment has a step size (default 100 ms). Each step you call env.step(action).
    The environment returns (obs, reward, done, info). The trial fully ends once we pass ITI.
    """
    def __init__(
        self,
        go_tone=True,       # True => 'go' trial, False => 'no-go'
        intensities=[5,15,25,35],
        step_ms=100,
        noise_std=0.2,
        rng_seed=None
    ):
        if rng_seed is not None:
            np.random.seed(rng_seed)

        self.go_tone = go_tone
        self.intensity = np.random.choice(intensities)

        # Phase durations in ms
        self.led_cue_ms         = 1000
        self.delay_mean_ms      = 650
        self.delay_std_ms       = 150
        self.sound_ms           = 500
        self.response_ms        = 800
        self.post_outcome_ms    = 250
        self.reward_ms          = 100  # time steps with +1 if correct
        self.punish_ms          = 300  # time steps with -1 if incorrect
        self.iti_ms             = 0    # 4000
        self.step_ms            = step_ms

        # Random delay
        raw_delay = np.random.normal(self.delay_mean_ms, self.delay_std_ms)
        random_delay_ms = max(0, raw_delay)

        # Convert to step counts
        self.led_steps       = self.led_cue_ms // step_ms
        self.delay_steps     = int(random_delay_ms // step_ms)
        self.sound_steps     = self.sound_ms // step_ms
        self.response_steps  = self.response_ms // step_ms
        self.post_outcome_steps = self.post_outcome_ms // step_ms
        self.reward_steps    = self.reward_ms // step_ms
        self.punish_steps    = self.punish_ms // step_ms
        self.iti_steps       = self.iti_ms // step_ms

        # Build total timeline
        self.total_steps = (self.led_steps + self.delay_steps +
                            self.sound_steps + self.response_steps +
                            self.post_outcome_steps +
                            # The reward phase or punish phase might be used, but let's
                            # define the maximum so we don't do branching
                            max(self.reward_steps, self.punish_steps)) # + self.iti_steps)

        obs = np.zeros((self.total_steps, 4), dtype=float)

        idx = 0
        # (A) fixation on
        obs[idx:idx+self.led_steps, 0] = 1.0  # LED on
        obs[idx:idx+self.led_steps, 3] = 1.0  # fixation on
        idx += self.led_steps

        # (B) random delay => fixation still on
        obs[idx:idx+self.delay_steps, 3] = 1.0
        idx += self.delay_steps

        # (C) sound => freq, intensity, fixation=1
        freq_val = +1.0 if self.go_tone else -1.0

        for t in range(idx, idx + self.sound_steps):
            # Base freq_val = +1 for go, -1 for no-go
            obs[t,1] = freq_val + np.random.normal(0, noise_std)
        obs[idx:idx+self.sound_steps,2] = float(self.intensity)/35.0
        obs[idx:idx+self.sound_steps,3] = 1.0
        idx += self.sound_steps

        # (D) response window => fixation=0
        obs[idx:idx+self.response_steps,3] = 0.0
        idx += self.response_steps

        # (E) post-outcome delay => no reward
        #   we haven't assigned reward/punishment, just a 'dead' period
        idx += self.post_outcome_steps

        # (F) We'll allocate space for either reward or punishment window.
        # We'll fill obs with zeros, and handle the reward or punishment in step().
        # We'll keep track of the index for the start of reward/punish
        self.reward_punish_start = idx
        idx += max(self.reward_steps, self.punish_steps)

        # (G) ITI => last segment
        idx += self.iti_steps

        self.obs_timeline = obs
        self.current_step = 0
        self.done = False

        # Track final action
        self.has_pressed = False   # True if agent pressed in the response window
        self.final_decided = False # True after response window finishes

        # We'll store whether final action is correct or not
        self.final_correct = False

        # Precompute the step indices for each phase
        self.start_response = (self.led_steps +
                               self.delay_steps +
                               self.sound_steps)
        self.end_response   = self.start_response + self.response_steps

        self.in_response_window = False # True if we're in the response window

        self.start_postoutcome = self.end_response
        self.end_postoutcome   = self.start_postoutcome + self.post_outcome_steps

        self.start_rewardpunish = self.reward_punish_start
        self.end_rewardpunish   = self.reward_punish_start + max(self.reward_steps, self.punish_steps)

        self.start_iti = self.end_rewardpunish
        self.end_iti   = self.total_steps  # i.e., idx

    def _compute_final_action(self):
        """
        Called after response window finishes to determine final action
        and correctness. We define final action=Press if agent pressed
        at least once, else=Hold.
        """
        final_action_is_press = self.has_pressed
        if self.go_tone:
            # correct if agent pressed
            self.final_correct = final_action_is_press
        else:
            # correct if agent did not press
            self.final_correct = (not final_action_is_press)

    def step(self, action):
        """
        action in {0=hold, 1=press}.
        - If we're in the response window (self.start_response <= t < self.end_response):
            - Pressing while fix=1 => negative partial reward. (Though fix=1 won't happen in response window.)
            - Press => self.has_pressed = True, no immediate done.
        - If we're in the reward/punish window:
            - Provide +1 each step if final_correct==True (and we're within the reward window).
            - Provide -1 each step if final_correct==False (and we're within the punish window).
        - End the trial only after we pass the final ITI.
        """
        if self.done:
            return None, 0.0, True, {}

        obs = self.obs_timeline[self.current_step, :]
        reward = 0.0

        t = self.current_step

        
        # (1) If we are in the response window, track press
        if (t >= self.start_response) and (t < self.end_response):
            if action in [0, 1]:
                print(f"t={t}, action={action}, obs={int(self.go_tone)}")

            self.in_response_window = True
            fix_val = obs[3]  # should be 0.0 in the response window
            if action == 1:
                # agent pressed
                self.has_pressed = True
                # Possibly give partial penalty if you want pressing while fix=0 is free
                # But if you want a penalty for multiple presses, you can do so:
                # e.g. reward -= 0.0 or something
                # Force the environment to jump straight to post-outcome
                self.current_step = self.start_postoutcome  

        # (2) Once we pass the response window, if we haven't computed final action, do so
        if (t >= self.end_response) and (not self.final_decided):
            self._compute_final_action()
            self.final_decided = True
            self.in_response_window = False

        # (3) If we are in the post-outcome delay, no reward
        if (t >= self.start_postoutcome) and (t < self.end_postoutcome):
            reward = 0.0

        # (4) If we are in the reward/punish window
        if (t >= self.start_rewardpunish) and (t < self.end_rewardpunish):
            # how many steps into this phase are we?
            phase_step = t - self.start_rewardpunish
            if self.has_pressed and self.go_tone: # or (not self.has_pressed) and (not self.go_tone):
                # +1 for the first reward_steps
                if phase_step < self.reward_steps:
                    reward = +1.0
            elif self.has_pressed and (not self.go_tone): # or (not self.has_pressed) and self.go_tone:
                # -1 for the first punish_steps
                if phase_step < self.punish_steps:
                    reward = -1.0
                    

        # (5) If pressing while fixation=1 anywhere outside the response window => partial penalty
        # This might happen if agent presses in early phases
        fix_val = obs[3]
        if (fix_val > 0.5):
            if (action == 1):
                # partial penalty
                # e.g., -0.25 each time step
                reward = -0.5
                self.done = True
            # else:
            #     reward = 0.1

        # (6) Move to next step
        self.current_step += 1
        if self.current_step >= self.total_steps:
            self.done = True

        # next_obs or None
        next_obs = (self.obs_timeline[self.current_step, :].copy()
                    if not self.done else None)

        return next_obs, reward, self.done, {}

    def reset(self):
        # Re-initialize
        self.current_step = 0
        self.done = False
        self.has_pressed = False
        self.final_decided = False
        self.final_correct = False
        return self.obs_timeline[0, :].copy()
