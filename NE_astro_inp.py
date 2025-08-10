import numpy as np
import matplotlib.pyplot as plt

def db_to_amplitude(dB):
    """
    Convert a decibel value to linear amplitude:
      amplitude = 10^(dB/20).
    E.g. 5 dB ~ 1.78, 15 dB ~ 5.62, 35 dB ~ 56.23, etc.
    """
    return 10.0 ** (dB / 20.0)

def build_drummond_trial(
    go_tone=True,
    intensity_dB=25,
    amplitude_noise_std = 0.25,
    led_cue_ms=1000,
    delay_mean_ms=650,
    delay_std_ms=150,
    sound_ms=500,
    response_ms=800,
    iti_ms=4000,
    step_ms=100,
    rng_seed=None
):
    """
    Build the exact temporal input for one trial of the Drummond go/no-go task, discretized in step_ms increments.

    Timeline:
      1) LED ON for led_cue_ms (mouse must hold still)
      2) Random delay ~ N(mean=delay_mean_ms, std=delay_std_ms), min 0 for safety
      3) Sound for sound_ms (0.5 s) => freq dimension + amplitude dimension
      4) 0.8s response window => no LED, no sound
      5) 4s ITI => no LED, no sound

    We'll produce an array `obs` of shape (T, 3), each row = [LED, freq, amplitude].
      - LED: 1 during the LED cue, else 0
      - freq: +1 if go tone, -1 if no-go, only during the sound; 0 otherwise
      - amplitude: 10^(dB/20) only during the sound; 0 otherwise

    The total length in steps is:
      T = (led_cue_ms + random_delay + sound_ms + response_ms + iti_ms) / step_ms
    (rounded down).

    :param go_tone: bool, True => go (12 kHz => freq=+1), False => no-go (4 kHz => freq=-1)
    :param intensity_dB: one of [5, 15, 25, 35], or any dB value
    :param led_cue_ms: 1000 ms (1 s) for LED
    :param delay_mean_ms: 650 ms
    :param delay_std_ms: 150 ms
    :param sound_ms: 500 ms (0.5 s)
    :param response_ms: 800 ms (0.8 s)
    :param iti_ms: 4000 ms (4 s)
    :param step_ms: each time step (default 100 ms)
    :param rng_seed: optional seed for reproducibility
    :return: obs, shape (T, 3), a float array
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    # 1) Sample random delay
    raw_delay = np.random.normal(delay_mean_ms, delay_std_ms)
    random_delay_ms = max(0, raw_delay)  # ensure it's not negative
    # convert each epoch to steps
    led_steps  = led_cue_ms // step_ms
    delay_steps= int(random_delay_ms // step_ms)
    sound_steps= sound_ms // step_ms
    response_steps = response_ms // step_ms
    iti_steps  = iti_ms // step_ms
    total_steps = led_steps + delay_steps + sound_steps + response_steps + iti_steps

    # 2) Prepare freq sign & amplitude
    freq_sign = 1.0 if go_tone else -1.0
    base_amp  = db_to_amplitude(intensity_dB)

    # 3) Initialize obs to zeros
    obs = np.zeros((total_steps, 3), dtype=float)

    # We define indices for each epoch
    idx_start = 0
    # (A) LED period
    idx_end = idx_start + led_steps
    if idx_end > idx_start:
        obs[idx_start:idx_end, 0] = 1.0  # LED=1
    idx_start = idx_end

    # (B) Random delay
    idx_end = idx_start + delay_steps
    # LED/freq/amplitude remain 0 in this delay
    idx_start = idx_end

    # (C) Sound (0.5 s)
    idx_end = idx_start + sound_steps
    if idx_end > idx_start:
        obs[idx_start:idx_end, 1] = freq_sign
        obs[idx_start:idx_end, 2] = intensity_dB # base_amp + np.random.normal(
    #     0.0, amplitude_noise_std, size=sound_steps
    # )
    idx_start = idx_end

    # (D) Response window (0.8 s)
    idx_end = idx_start + response_steps
    # all zero => no LED, freq=0, amplitude=0
    idx_start = idx_end

    # (E) ITI (4 s)
    idx_end = idx_start + iti_steps
    # all zero => obs remain 0
    idx_start = idx_end

    # done
    return obs

import numpy as np
import matplotlib.pyplot as plt

def build_outcome_time_series(
    total_steps,
    step_ms=100,
    press_step=None,
    correct_press=None,
    reward_delay_ms=250,
    punishment_duration_ms=300,
):
    """
    Build a time-series array (T,2) for reward/punishment signals.
    
    :param total_steps: total # of steps in this trial
    :param step_ms:     how many ms each time step represents
    :param press_step:  which step (0..total_steps-1) the lever press occurs, or None if no press
    :param correct_press: bool or None => if True => reward, if False => punishment, if None => no outcome
    :param reward_delay_ms:  250 ms after lever press for water
    :param punishment_duration_ms: 300 ms for air puff
    :return: outcome array shape (total_steps,2),
       outcome[:,0] => reward dimension
       outcome[:,1] => punishment dimension
    """

    outcome = np.zeros((total_steps * step_ms, 1), dtype=float)

    if press_step is None or correct_press is None:
        # No press or uncertain correctness => no outcome
        return outcome

    # Step index at which the outcome *starts*
    outcome_start_delay_steps = reward_delay_ms
    # The outcome starts after some delay from the press
    start_idx = outcome_start_delay_steps

    if correct_press:
        # A single-time-step water drop
        outcome[start_idx:start_idx+50, 0] = 1.0
    else:
        # Punishment: last 0.3s => how many steps is that?
        punish_steps = punishment_duration_ms   # e.g. 300 ms => 3 steps if step_ms=100
        end_idx = min(start_idx + punish_steps, total_steps * step_ms)
        outcome[start_idx:end_idx, 0] = -1.0

    return outcome

def plot_drummond_trial(obs, step_ms=100, title="Drummond-Style Trial Input"):
    """
    Plot the 3D time series vs. time in ms:
      obs[:,0] => LED dimension
      obs[:,1] => freq dimension
      obs[:,2] => amplitude dimension
    """
    T = obs.shape[0]
    times_ms = np.arange(T) * step_ms

    plt.figure(figsize=(7,4))
    plt.plot(times_ms, obs[:,0], label="LED (Cue) dimension")
    plt.plot(times_ms, obs[:,1], label="Freq dimension")
    plt.plot(times_ms, obs[:,2], label="Loudness (dB) dimension")
    plt.xlabel("Time (ms)")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_outcome_time_series(outcome, step_ms=100, title="Outcome Timeline"):
    """
    Plot the reward/punishment over time:
      outcome[:,0] = reward dimension
      outcome[:,1] = punishment dimension
    We'll see a single step for reward, or multiple steps for punishment.
    """
    T = outcome.shape[0]
    time_arr = np.arange(T)

    plt.figure(figsize=(7,3))
    plt.plot(time_arr, outcome[:,0], label="Reward/Punishment", drawstyle='steps-post')
    # plt.plot(time_arr, outcome[:,1], label="Punishment", drawstyle='steps-post')
    plt.xlabel("Time (ms)")
    plt.ylabel("Value (1 or -1)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Demo usage:
if __name__ == "__main__":
    # Example trial: Go=12 kHz, intensity=25 dB
    # random delay ~ N(650,150), step=100 ms
    trial_obs = build_drummond_trial(
        go_tone=True,
        intensity_dB=25,
        rng_seed=42
    )
    print("Shape of trial_obs:", trial_obs.shape)

    # Plot
    plot_drummond_trial(trial_obs, step_ms=100, title="Go Trial, 25 dB")


    trial_obs = build_drummond_trial(
        go_tone=False,
        intensity_dB=25,
        rng_seed=42
    )

    print("Shape of trial_obs:", trial_obs.shape)

    # Plot
    plot_drummond_trial(trial_obs, step_ms=100, title="No-Go Trial, 25 dB")


    T = 10
    step_ms = 100

    # Example 1: correct press at step=20
    outcome_correct = build_outcome_time_series(
        total_steps=T,
        step_ms=step_ms,
        press_step=20,       # pressed at time step 20
        correct_press=True,  # it's a correct press => reward
        reward_delay_ms=250, # 0.25s
        punishment_duration_ms=300
    )
    plot_outcome_time_series(outcome_correct, step_ms, title="Correct Press => Reward")

    # Example 2: incorrect press at step=25 => punishment
    outcome_incorrect = build_outcome_time_series(
        total_steps=T,
        step_ms=step_ms,
        press_step=20,
        correct_press=False,  # no-go trial => false alarm => punishment
        reward_delay_ms=250,
        punishment_duration_ms=300
    )
    plot_outcome_time_series(outcome_incorrect, step_ms, title="Incorrect Press => Punishment")
