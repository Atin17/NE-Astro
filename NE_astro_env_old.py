import numpy as np

class DrummondGoNoGoEnv:
    def __init__(self,
                 intensities=[5, 15, 25, 35],
                 go_freq_id=1.0,         # code for go
                 nogo_freq_id=-1.0,      # code for no-go
                 noise_std=0.5,         # noise for the observation
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
        """
        Reset environment for a new 'session'.
        """
        self.trial_count = 0
        self.done = False
        return self._generate_observation()
    
    def step(self, action):
        """
        action = 1 means 'press lever'
        action = 0 means 'do nothing'
        
        Return observation, reward, done, info
        """
        # 1. Determine reward from last trial’s stim + this trial’s action
        reward = self._calculate_reward(self.current_stim, action)
        
        # 2. Increment trial count and check if done
        self.trial_count += 1
        if self.trial_count >= self.max_trials:
            self.done = True
        
        # 3. Generate the next observation
        obs = self._generate_observation()
        
        # Info dictionary can store trial outcome details, if needed
        info = {"stim": self.current_stim}
        
        return obs, reward, self.done, info
    
    def _generate_observation(self):
        """
        Randomly choose go/no-go + intensity, produce a 2D input with noise.
        """
        # 50% chance go, 50% chance no-go (you can adjust to match the real experiment).
        is_go = np.random.rand() < 0.5  
        
        # pick intensity index
        intensity_idx = np.random.randint(len(self.intensities))
        intensity = self.intensities[intensity_idx]
        
        # encode frequency dimension
        freq_val = self.go_freq_id if is_go else self.nogo_freq_id
        
        # scale intensity to [0..1] or keep it raw. Let's do a simple scale
        # e.g., normalized by 35 (the max dB).
        intensity_val = intensity / 35.0
        
        # store the "ground truth" stim
        self.current_stim = (1 if is_go else 0, intensity_idx)
        
        # create a 2D observation
        obs = np.array([freq_val, intensity_val], dtype=float)
        
        # add noise
        obs += np.random.normal(0, self.noise_std, size=obs.shape)
        
        return obs
    
    def _calculate_reward(self, stim, action):
        """
        stim = (go_or_not, intensity_index)
        action = 0 or 1
        """
        is_go = (stim[0] == 1)
        
        if is_go:
            # correct action is to press
            if action == 1:
                return self.reward_hit
            else:
                return self.reward_miss
        else:
            # correct action is do nothing
            if action == 1:
                return self.reward_false_alarm
            else:
                return self.reward_correct_reject