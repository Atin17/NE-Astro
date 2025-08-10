import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

############################################
# 1) Build a single-trial timeline
############################################

def build_drummond_trial(
    go_tone=True,
    intensity_dB=25,
    step_ms=100,
    rng_seed=None,
    freq_noise_std=0.0
):
    """
    Return an observation timeline for a single trial, with these phases:
      - 1s LED
      - random delay ~ N(650,150) ms
      - 0.5 s sound window => SINE WAVE:
          12 Hz if go_tone, 4 Hz if no_go_tone
      - 0.8 s response
      - 4 s ITI

    The 'freq dimension' (index=1) is replaced by the sinusoidal wave.
    The 'intensity dimension' (index=2) can remain as a constant (or carry noise).
    freq_noise_std is optional noise added to the sine wave amplitude at each step.
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    # durations in ms
    led_cue_ms   = 1000
    delay_mean_ms= 650
    delay_std_ms = 150
    sound_ms     = 500
    response_ms  = 800
    iti_ms       = 4000

    # random delay
    raw_delay = np.random.normal(delay_mean_ms, delay_std_ms)
    random_delay_ms = max(0, raw_delay)  # clamp at 0

    # convert each phase to step counts
    led_steps  = led_cue_ms      // step_ms
    delay_steps= int(random_delay_ms // step_ms)
    sound_steps= sound_ms        // step_ms
    resp_steps = response_ms     // step_ms
    iti_steps  = iti_ms          // step_ms
    total_steps= led_steps + delay_steps + sound_steps + resp_steps + iti_steps

    # Build obs array => shape (total_steps, 3)
    #   obs[:,0]: LED dimension
    #   obs[:,1]: freq dimension => SINE wave
    #   obs[:,2]: amplitude dimension => maybe store 'intensity_dB'
    obs = np.zeros((total_steps, 3), dtype=np.float32)

    # LED period
    idx = 0
    obs[idx : idx+led_steps, 0] = 1.0
    idx += led_steps

    # random delay => remain 0
    idx += delay_steps

    # sound period => sinusoidal wave in freq dimension
    sound_start = idx
    sound_end   = idx + sound_steps
    freq_hz = 12.0 if go_tone else 4.0

    for s in range(sound_steps):
        time_sec = (s * step_ms) / 1000.0   # convert ms to seconds
        # simple sine wave => amplitude=1
        wave_val = np.sin(2.0 * np.pi * freq_hz * time_sec)

        # Optionally add Gaussian noise to the wave
        if freq_noise_std>0:
            wave_val += np.random.normal(0, freq_noise_std)

        obs[sound_start + s, 1] = wave_val

        # Keep amplitude dimension for intensity => or 0 if you prefer
        obs[sound_start + s, 2] = float(intensity_dB)

    idx += sound_steps

    # response window
    response_start = sound_end
    response_end   = response_start + resp_steps
    idx += resp_steps

    # ITI
    idx += iti_steps

    return obs, total_steps, response_start, response_end
############################################
# 2) Tempotrone RNN
############################################
class TempotroneRNN(nn.Module):
    """
    Minimal RNN with single readout potential each time step.
    We find earliest crossing in the decision window => 'press time'.
    Also define a logistic 'prob' of press or not for RL sampling.
    """
    def __init__(self, input_size=3, hidden_size=32, threshold=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        # readout
        self.readout_w = nn.Parameter(torch.randn(hidden_size)/np.sqrt(hidden_size))
        self.readout_b = nn.Parameter(torch.zeros(1))

    def forward(self, input_seq):
        """
        input_seq: shape (T, input_size)
        returns:
          potentials: shape (T,) => readout potential each step
          hidden states if needed
        """
        T = input_seq.shape[0]
        h = torch.zeros(self.hidden_size)
        pots = []
        for t in range(T):
            x_t = input_seq[t,:]
            h = self.rnn_cell(x_t, h)
            pot = torch.dot(self.readout_w, h) + self.readout_b
            pots.append(pot)
        pots = torch.stack(pots)  # (T,)
        return pots

    def single_press_time(self, potentials, response_start, response_end):
        """
        Find earliest crossing in [response_start, response_end). None if no crossing.
        """
        T = potentials.shape[0]
        end_ = min(response_end, T)
        for t in range(response_start, end_):
            if potentials[t] > self.threshold:
                return t
        return None

    def press_probability(self, potentials, resp_start, resp_end):
        """
        For RL: define a single Bernoulli(prob_press).
        E.g. we can take max potential in [resp_start..resp_end), shift by threshold,
        then apply a sigmoid to interpret it as p(press).
        """
        if resp_start>=potentials.shape[0]:
            # no time window => prob ~ 0
            return torch.tensor(0.0)
        sub_pots = potentials[resp_start:resp_end]
        max_sub  = torch.max(sub_pots)
        # shift by threshold, then logistic
        # e.g. p = sigma( scale*(max_sub - threshold) )
        scale=5.0  # hyperparam to tune
        p_press = torch.sigmoid( scale * (max_sub - self.threshold) )
        return p_press

############################################
# 3) Combine Tempotron Cost + RL
############################################
def train_tempotron_with_RL(
    num_episodes=500,
    go_prob=0.5,
    intensities=[5,15,25,35],
    step_ms=100,
    hidden_size=32,
    threshold=0.5,
    lr=1e-3,
    rng_seed=42,
    intensity_noise_std=0.0,
    freq_noise_std=0.0,
    # Extended reward windows
    press_delay_ms=250,
    reward_dur_ms=100,
    punish_dur_ms=300,
    # weighting
    weight_decay=1e-2,
    alpha_tempotron=1.0,
    alpha_rl=1.0
):
    """
    We'll do a 2-part objective:
      1) A 'tempotron' cost that ensures crossing or not crossing in relevant windows.
      2) An RL cost from a 'temporal' reward:
         - If go & press => after 0.25s => +1 for 100ms
         - If no-go & press => after 0.25s => -1 for 300ms
         - else => 0
    The final reward for the episode is how many 100ms increments of +1 or -1 we fit
    before the trial ended. 
    """

    # np.random.seed(rng_seed)
    # torch.manual_seed(rng_seed)

    model = TempotroneRNN(
        input_size=3,
        hidden_size=hidden_size,
        threshold=threshold
    )
    # L2 reg for example
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    success_history = []
    press_delay_steps = press_delay_ms // step_ms
    rew_steps = reward_dur_ms // step_ms
    pun_steps = punish_dur_ms // step_ms

    for ep in range(num_episodes):
        # 1) Build trial
        is_go = (np.random.rand() < go_prob)
        intensity = np.random.choice(intensities)
        # optional: noise to intensity
        # intensity += np.random.normal(0.0, intensity_noise_std * intensity)

        obs, T, resp_start, resp_end = build_drummond_trial(
            go_tone=is_go,
            intensity_dB=intensity,
            step_ms=step_ms,
            freq_noise_std=freq_noise_std
        )

        # 2) Forward pass => potentials
        obs_torch = torch.from_numpy(obs)   # shape(T,3)
        potentials = model(obs_torch)       # shape(T,)

        # 3) single press time from threshold crossing
        press_time = model.single_press_time(potentials, resp_start, resp_end) # random.choice([1, 10])

        # 4) Calculate "temporal reward"
        env_reward = 0.0
        if press_time is not None:
            outcome_start = press_time + press_delay_steps
            if is_go:
                # deliver +1 for 100 ms
                # that is rew_steps steps => each step => +1
                # outcome_end = min(outcome_start + rew_steps, T)
                # length = outcome_end - outcome_start
                # length is how many 100ms intervals fit
                env_reward = 1.0 * 100 // step_ms # length  
            else:
                # no-go => punishment => -1 for 300ms
                # outcome_end = min(outcome_start + pun_steps, T)
                # length = outcome_end - outcome_start
                env_reward = -1.0 * 300 // step_ms # length
        else:
            # no press => no reward
            env_reward = 0.0

        # success => if is_go => press, if no-go => no press
        # or we can define success if the agent net positive
        if is_go:
            success = (press_time is not None)
        else:
            success = (press_time is None)
        success_history.append(1.0 if success else 0.0)

        # 5) RL Probability => for REINFORCE
        # interpret "p_press" from max potential
        p_press = model.press_probability(potentials, resp_start, resp_end)
        action = 1 if press_time is not None else 0
        if action == 1:
            log_prob = torch.log(p_press + 1e-8)
        else:
            log_prob = torch.log((1.0 - p_press) + 1e-8)

        # => RL cost = - env_reward * log_prob
        cost_rl = - env_reward * log_prob

        # 6) Tempotron cost => same approach as before
        cost_temp = torch.tensor(0.0, requires_grad=True)
        if press_time is not None:
            # there's a crossing
            if not is_go:
                # no-go => punish window => want pot< threshold => cost= ReLU(max_sub - threshold)
                outcome_end = min(press_time + press_delay_steps + pun_steps, T)
                outcome_start = press_time + press_delay_steps
                if outcome_start < T:
                    sub_pots = potentials[outcome_start: outcome_end]
                    if sub_pots.shape[0] > 0:
                        max_sub = torch.max(sub_pots)
                        c_nogo = F.relu(max_sub - model.threshold)
                        cost_temp = cost_temp + c_nogo
        else:
            # no crossing
            if is_go:
                # missed => cost= ReLU(threshold - max_sub) in resp window
                sub_pots = potentials[resp_start:resp_end]
                if sub_pots.shape[0]>0:
                    max_sub = torch.max(sub_pots)
                    c_miss = F.relu(model.threshold - max_sub)
                    cost_temp = cost_temp + c_miss
            else:
                pass

        cost = alpha_tempotron * cost_temp + alpha_rl * cost_rl

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    return model, success_history

########################################
# 4) Evaluate at Different Noise Levels
########################################
def evaluate_model_freq(model, freq_noise_levels, baseline_noise, n_trials=200):
    """
    For each freq noise level in freq_noise_levels, run 'n_trials' go/no-go episodes,
    measure the success rate, then return an array of success rates.
    """
    success_rates = []
    for std in freq_noise_levels:
        successes = []
        for _ in range(n_trials):
            # random 50% go / no-go
            is_go = (np.random.rand()<0.5)
            intensity = np.random.choice([5,15,25,35])
            obs, T, resp_start, resp_end = build_drummond_trial(
                go_tone=is_go,
                intensity_dB=intensity,
                # intensity_noise_std=0.0,
                freq_noise_std=std + baseline_noise
            )
            with torch.no_grad():
                obs_torch = torch.from_numpy(obs)
                pots = model(obs_torch)
                press_time = model.single_press_time(pots, resp_start, resp_end)

                # environment reward
                if is_go:
                    rew = +1 if press_time is not None else 0
                else:
                    rew = -1 if press_time is not None else 0
                success = ((rew>0) or (rew==0 and not is_go))
                successes.append(1.0 if success else 0.0)
        sr = np.mean(successes)
        success_rates.append(sr)
    return np.array(success_rates)

def evaluate_model_inten(model, inten_noise_levels, baseline_noise, n_trials=200):
    """
    For each freq noise level in freq_noise_levels, run 'n_trials' go/no-go episodes,
    measure the success rate, then return an array of success rates.
    """
    success_rates = []
    for std in inten_noise_levels:
        successes = []
        for _ in range(n_trials):
            # random 50% go / no-go
            is_go = (np.random.rand()<0.5)
            intensity = np.random.choice([5,15,25,35])
            obs, T, resp_start, resp_end = build_drummond_trial(
                go_tone=is_go,
                intensity_dB=intensity,
                intensity_noise_std=std + baseline_noise,
                freq_noise_std=0.0
            )
            with torch.no_grad():
                obs_torch = torch.from_numpy(obs)
                pots = model(obs_torch)
                press_time = model.single_press_time(pots, resp_start, resp_end)

                # environment reward
                if is_go:
                    rew = +1 if press_time is not None else 0
                else:
                    rew = -1 if press_time is not None else 0
                success = ((rew>0) or (rew==0 and not is_go))
                successes.append(1.0 if success else 0.0)
        sr = np.mean(successes)
        success_rates.append(sr)
    return np.array(success_rates)


def run_experiment():

    successes = []
    hidden_size=[1, 4, 16] # , 64, 128]
    intensity_noise_std=0.0
    freq_noise_std=1.0

    for i in hidden_size:
        _, success = train_tempotron_with_RL(
            num_episodes=1000,
            go_prob=0.5,
            intensities=[5,15,25,35],
            step_ms=100,
            hidden_size=i,
            threshold=0.3,
            lr=1e-1,
            rng_seed=42,
            intensity_noise_std=intensity_noise_std,
            freq_noise_std=freq_noise_std,
            press_delay_ms=250,
            reward_dur_ms=100,
            punish_dur_ms=300,
            alpha_tempotron=1.0,
            alpha_rl=0.5
        )

        successes.append(success)  # store for plotting

    # plot a rolling average of success
    window=20
    smoothed_all = []
    for s_array in successes:
            smoothed = []
            for j in range(len(s_array)):
                start = max(0, j - window + 1)
                smoothed.append(np.mean(s_array[start:j+1]))
            smoothed_all.append(smoothed)

    plt.figure(figsize=(8,4))
    for i, hs in enumerate(hidden_size):
            plt.plot(smoothed_all[i], label=f"hidden size={hs}")
    plt.xlabel("Trials")
    plt.ylabel("Success Rate")
    plt.title(f"Learning Performance \nintensity noise std={intensity_noise_std} and freq noise std={freq_noise_std}")
    plt.legend()
    plt.grid(True)
    plt.show()

def run_psychometric_curve():
    # 1) Train the model at a baseline noise
    baseline_noise=0.5
    successes = []
    hidden_size=[64]
    intensity_noise_std=0.0
    freq_noise_std=baseline_noise
    success_rates_list = []

    for i in hidden_size:
        model, success = train_tempotron_with_RL(
            num_episodes=1000,
            go_prob=0.5,
            intensities=[5,15,25,35],
            step_ms=100,
            hidden_size=i,
            threshold=0.3,
            lr=1e-4,
            rng_seed=36,
            intensity_noise_std=intensity_noise_std,
            freq_noise_std=freq_noise_std,
            press_delay_ms=250,
            reward_dur_ms=100,
            punish_dur_ms=300,
            weight_decay=1e-3,
            alpha_tempotron=1.0,
            alpha_rl=0.5
        )

        # successes.append(success)  # store for plotting
        # 2) Evaluate at higher noise levels
        noise_levels = np.linspace(0.0, 5.0, 16)  # e.g. [0.0, 0.1, 0.2,...1.0]
        success_rates = evaluate_model_freq(model, noise_levels, baseline_noise, n_trials=300)
        success_rates_list.append(success_rates)

    # 3) Plot the psychometric curve
    plt.figure(figsize=(8,4))
    for i, hs in enumerate(hidden_size):
        plt.plot(noise_levels, success_rates_list[i], marker='o', label=f"hidden size={hs}")
    plt.axhline(y=0.5, color='black', linestyle='-')
    plt.xlabel("Frequency Noise Std")
    plt.ylabel("Accuracy (Success Rate)")
    plt.title(f"Psychometric Curve: RNN vs. Noise (Trained at baseline={baseline_noise})")
    plt.grid(True)
    plt.ylim([0,1])
    plt.legend()
    plt.show()



if __name__=="__main__":
    run_psychometric_curve()
