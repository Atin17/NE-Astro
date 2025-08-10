import numpy as np

class DrummondGoNoGoEnv:
    def __init__(self,
                 intensities=[5, 15, 25, 35],
                 go_freq_id=1.0,         # code for go
                 nogo_freq_id=-1.0,      # code for no-go
                 noise_std=0.25,         # noise for the observation
                 reward_hit=+1.0,
                 reward_false_alarm=-1.0,
                 reward_miss=0.0,
                 reward_correct_reject=0.0,
                 max_trials=1000):
        
        """
        A simplified environment to mimic Drummond et al.:
        - intensities: dB levels for the tone.
        - go_freq_id, nogo_freq_id: how we encode frequency dimension numerically.
        - noise_std: standard deviation of noise added to the observation.
        - reward_xxx: reward structure for different outcomes.
        - max_trials: max number of trials in one 'session'.
        """
        
        self.intensities = intensities
        self.go_freq_id = go_freq_id
        self.nogo_freq_id = nogo_freq_id
        self.noise_std = noise_std
        
        self.reward_hit = reward_hit
        self.reward_false_alarm = reward_false_alarm
        self.reward_miss = reward_miss
        self.reward_correct_reject = reward_correct_reject
        
        self.max_trials = max_trials
        
        self.trial_count = 0
        self.done = False
        
        # We'll store the true label (go=1, no-go=0) and intensity index for reference
        self.current_stim = None  # (go_or_not, intensity)
    
    def reset(self):
        self.trial_count = 0
        self.done = False
        return self._generate_observation()
    
    def step(self, action):
        """
        action in {0,1}
          - 1 => press lever
          - 0 => do nothing
        returns (obs, reward, done, info)
        """
        # reward from last trial
        reward = self._calculate_reward(self.current_stim, action)
        
        self.trial_count += 1
        if self.trial_count >= self.max_trials:
            self.done = True
        
        obs = self._generate_observation()
        info = {"stim": self.current_stim}
        return obs, reward, self.done, info
    
    def _generate_observation(self):
        """
        - 50% GO, 50% NO-GO
        - pick random intensity
        - obs[0] = + intensityVal if GO, 0 if NO-GO
        - obs[1] = - intensityVal if NO-GO, 0 if GO
        Then add noise to each dimension.
        """
        is_go = (np.random.rand() < 0.5)
        intensity_idx = np.random.randint(len(self.intensities))
        intensity = self.intensities[intensity_idx]

        # normalized intensity
        intensity_val = intensity / 35.0

        # store ground truth
        self.current_stim = (1 if is_go else 0, intensity_idx)

        if is_go:
            # go => first dimension is +intensity, second is 0
            obs = np.array([ intensity_val, 0.0 ], dtype=float)
        else:
            # no-go => first dimension is 0, second is -intensity
            obs = np.array([ 0.0, -intensity_val ], dtype=float)

        # add noise
        obs += np.random.normal(0, self.noise_std, size=obs.shape)
        return obs
    
    def _calculate_reward(self, stim, action):
        """
        stim = (1 if go else 0, intensity_idx)
        action in {0,1}
        """
        is_go = (stim[0] == 1)
        if is_go:
            # correct => press
            return self.reward_hit if (action == 1) else self.reward_miss
        else:
            # correct => do nothing
            return self.reward_false_alarm if (action == 1) else self.reward_correct_reject
