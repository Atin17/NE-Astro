import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from drummond_env import MouseBehaviorEnvTemporalReward  # your environment class
from CustomGRU import CustomActorCriticGRU2Layer  # your custom GRU class defined earlier
from ppoCustomGRU import record_hidden_activations_during_task


# from custom_gru_2layer import CustomActorCriticGRU2Layer
# from drummond_env import MouseBehaviorEnvTemporalReward

def record_hidden_activations_over_episodes(model, env_class, num_episodes=10, max_steps=200):
    """
    Runs `num_episodes` episodes in the environment, collecting layer2 hidden states
    at each time step.

    Returns:
      all_layer2_acts: shape (total_steps, hidden_size)
        where total_steps is the sum of steps from all episodes.
    """
    device = model.device
    hidden_size = model.hidden_size

    all_layer2 = []
    rewards = []

    for e in range(num_episodes):
        env = MouseBehaviorEnvTemporalReward(go_tone = False)
        h = model.initial_hidden(batch_size=1)  # shape (2,1,hidden_size)
        obs_np = env.reset()
        obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).unsqueeze(0).to(device)
        done = False
        steps = 0

        while not done and steps < max_steps:
            with torch.no_grad():
                logits, value, h_new = model.forward(obs_t, h)
            
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            # Store the hidden state from layer2
            layer2_vec = h[1,0,:].cpu().numpy().copy()  # shape (hidden_size,)
            all_layer2.append(layer2_vec)

            # Step environment
            obs_next, reward, done, _info = env.step(action.item())
            h = h_new.detach()

            if not done:
                obs_t = torch.from_numpy(obs_next).float().unsqueeze(0).unsqueeze(0).to(device)

            steps += 1
            rewards.append(reward)

    all_layer2_acts = np.array(all_layer2)  # shape (total_steps, hidden_size)
    return all_layer2_acts, rewards

def classify_units_by_threshold(layer2_acts, fraction_of_max=0.5, min_frac_active=0.2):
    """
    For each unit in layer2_acts, compute:
      - avg_activation = mean
      - max_activation
      - threshold = fraction_of_max * max_activation
    Then count the fraction of time the unit is above that threshold.
    
    If fraction_of_time_above >= min_frac_active => classify as "slow"
    else => "fast"

    Returns:
      classification: list of "slow"/"fast"
      stats: dict of arrays:
        "avg": shape (hidden_size,)
        "max": shape (hidden_size,)
        "fraction_time_above": shape (hidden_size,)
    """
    total_steps, hidden_size = layer2_acts.shape
    avg_vals = np.mean(layer2_acts, axis=0)  # shape (hidden_size,)
    max_vals = np.max(layer2_acts, axis=0)   # shape (hidden_size,)

    classification = []
    fraction_time_above = np.zeros(hidden_size, dtype=np.float32)

    for i in range(hidden_size):
        thr = fraction_of_max * max_vals[i]
        above_mask = layer2_acts[:, i] >= thr
        frac_above = np.mean(above_mask)
        fraction_time_above[i] = frac_above

        if frac_above >= min_frac_active:
            classification.append("slow")
        else:
            classification.append("fast")

    stats = {
        "avg": avg_vals,
        "max": max_vals,
        "fraction_time_above": fraction_time_above
    }
    return classification, stats

def run_episode_collect_hidden(model, env):
    """
    Returns:
        acts   : (T , hidden_size)  numpy array of layer-2 hidden activity
        rewards: (T,)              optional – could be logged for behaviour
    """
    device = model.device
    h      = model.initial_hidden(batch_size=1)          # (2,1,H)
    obs    = torch.from_numpy(env.reset()).float()\
             .unsqueeze(0).unsqueeze(0).to(device)
    done   = False
    A = [] ; R = []

    while not done:
        logits, _, h_new = model.forward(obs, h)
        act = torch.distributions.Categorical(logits=logits).sample()
        obs_next, rew, done, _ = env.step(act.item())

        # store layer-2 activity (index 0 because we routed slow/fast to layer 2)
        A.append(h[0, 0].detach().cpu().numpy())   # (hidden_size,)
        R.append(rew)

        h = h_new.detach()
        if not done:
            obs = torch.from_numpy(obs_next).float()\
                  .unsqueeze(0).unsqueeze(0).to(device)

    return np.stack(A), np.asarray(R)


def classify_units_by_threshold_mod(
    layer2_acts,
    fraction_of_max=0.5,
    min_frac_active=0.2
):
    """
    Classify units as 'slow' or 'fast' based on how long they stay
    above a threshold fraction of their peak absolute activation.
    """
    T, H = layer2_acts.shape

    # compute per-unit peak of the absolute activation
    max_abs = np.max(np.abs(layer2_acts), axis=0)  # shape (H,)
    # (optionally) compute average absolute activation
    avg_abs = np.mean(np.abs(layer2_acts), axis=0)

    thr = fraction_of_max * max_abs

    fraction_time_above = np.mean(np.abs(layer2_acts) >= thr[None, :], axis=0)

    classification = [
        "slow" if fraction_time_above[i] >= min_frac_active else "fast"
        for i in range(H)
    ]

    stats = {
        "avg": avg_abs,
        "max": max_abs,
        "fraction_time_above": fraction_time_above
    }
    return classification, stats


def main():
    # 1) Load or instantiate your model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    slow_units_all = []
    

    # for min_frac_active in [0.5, 0.6, 0.7, 0.8, 0.9]:
    #     slow_units_active = []
    #     for i in range(10):
    #         model = CustomActorCriticGRU2Layer(input_size=4, hidden_size=64, action_size=2, device=device)
    #         # Optionally load pretrained

    #         model.load_state_dict(torch.load(f"ppo_custom_gru_noise0.2.{i}.pth"))

    #         # 2) Record hidden activations across multiple episodes
    #         layer2_acts = record_hidden_activations_over_episodes(
    #             model=model, 
    #             env_class=lambda: MouseBehaviorEnvTemporalReward(go_tone=True, step_ms=100, noise_std=0.2),
    #             num_episodes=5,  # run 10 episodes
    #             max_steps=200
    #         )
    #         # print("layer2_acts shape =", layer2_acts.shape)  # e.g. (T, hidden_size)

    #         # 3) Classify units as slow/fast
    #         classification, stats = classify_units_by_threshold_mod(
    #             layer2_acts,
    #             fraction_of_max=0.75,   # threshold = 0.5 * max
    #             min_frac_active=min_frac_active  # must be above threshold >= 20% of steps
    #         )
    #         slow_units_active.append(classification.count("slow"))
    #         print(f"ppo_custom_gru_noise0.2.{i}.pth has {classification.count('slow')} slow units")
    #     slow_units_all.append(slow_units_active)

    # means = [np.mean(slow_units_all[i]) for i in range(len(slow_units_all))]
    # stds = [np.std(slow_units_all[i]) for i in range(len(slow_units_all))]

    # # print(slow_units_all[-1])
    # # Print summary
    # hidden_size = model.hidden_size
    # for i in range(hidden_size):
    #     print(f"Unit {i}: {classification[i]}, avg={stats['avg'][i]:.3f}, max={stats['max'][i]:.3f}, frac_above={stats['fraction_time_above'][i]:.3f}")

    # # 4) Plot or do further analysis
    # # e.g. a histogram of fraction_time_above

    # # Define x values (e.g., unit indices)
    # x = [f"{i}" for i in  [0.5, 0.6, 0.7, 0.8, 0.9]]

    # # Create the plot with error bars
    # plt.figure(figsize=(8, 5))
    # plt.bar(x, means, yerr=stds, capsize=6, color='orange', edgecolor='black')

    # plt.xlabel('Minimum Fraction Activity >= ', fontsize=12)
    # plt.ylabel('Mean Count (± STD)', fontsize=12)
    # plt.title('Number of Slow Units', fontsize=14)
    # plt.tight_layout()
    # plt.show()

    # plt.figure()


    # plt.hist(stats['fraction_time_above'], bins=20, edgecolor='black')
    # plt.title("Histogram of fraction_time_above threshold")
    # plt.xlabel("Fraction of time above threshold")
    # plt.ylabel("Count of units")
    # plt.grid(True)
    # plt.show()

    # # If you want to see how many 'slow' vs 'fast' units
    # slow_count = classification.count("slow")
    # fast_count = classification.count("fast")
    # print(f"Identified {slow_count} slow units, {fast_count} fast units out of {hidden_size} total.")

    model_path  = "ppo_custom_gru_noise0.2.9.pth"
    model       = CustomActorCriticGRU2Layer(input_size=4, hidden_size=64,
                                            action_size=2, device=device)
    model.load_state_dict(torch.load(model_path))

    # -----------------------------------------------------------------
    # classify units once more so we know which indices are slow / fast
    # -----------------------------------------------------------------
    acts_train, rewards = record_hidden_activations_over_episodes(
                    model, MouseBehaviorEnvTemporalReward(),
                    num_episodes=3,  max_steps=200)

    labels_slow, _   = classify_units_by_threshold_mod(acts_train,
                                            fraction_of_max=0.75,
                                            min_frac_active=0.7)
    labels_fast, _   = classify_units_by_threshold_mod(acts_train,
                                            fraction_of_max=0.75,
                                            min_frac_active=0.3)


    slow_idx = np.where(np.array(labels_slow) == "slow")[0]
    fast_idx = np.where(np.array(labels_fast) == "fast")[0]
    print(f"Slow units: {slow_idx}\nFast units: {fast_idx}")

# now record actual task activity:
    layer1, layer2, rewards = record_hidden_activations_during_task(
        model,
        MouseBehaviorEnvTemporalReward(go_tone=False, step_ms=100, noise_std=0.2),
        num_trials=10,
        max_steps=200
    )

    # e.g. plot the first 100 timesteps of a couple of units:
    T, H = layer2.shape
    ts = np.arange(T) * 0.1  # in seconds

    plt.figure(figsize=(8,4))
    # plot a few slow units (you already found their indices):
    for idx in slow_idx[:5]:
        plt.plot(ts, layer2[:, idx], label=f"slow {idx}", alpha=0.8)
    for idx in fast_idx[:5]:
        plt.plot(ts, layer2[:, idx], '--', label=f"fast {idx}", alpha=0.8)

    plt.xlabel("Time (s)")
    plt.ylabel("Layer-2 activation")
    plt.title("Example slow vs fast unit traces during task")
    plt.legend(ncol=2, fontsize="small")
    plt.show()

    # plot rewards over time
    plt.figure(figsize=(8,4))
    plt.plot(ts, rewards[:T], label="reward", color='orange', alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Reward")
    plt.title("Reward over time")
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()