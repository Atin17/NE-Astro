import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from drummond_env import MouseBehaviorEnvTemporalReward

# ------------------------------------------------
# 1. Device
# ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------
# 2. Actor-Critic GRU
# ------------------------------------------------
class ActorCriticGRU(nn.Module):
    """
    A stacked GRU that outputs both:
      - Policy logits (for action selection)
      - A single value prediction (critic) for the state.
    """
    def __init__(self, input_size=4, hidden_size=32, num_layers=2, action_size=2):
        super(ActorCriticGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Recurrent backbone
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # Policy head -> outputs logits for discrete actions
        self.policy_head = nn.Linear(hidden_size, action_size)
        # Value head -> outputs a single scalar value
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x, h):
        """
        x shape: (batch=1, seq=1, input_size=4)
        h shape: (num_layers, batch=1, hidden_size)
        Returns:
          - logits (1, action_size)
          - value  (1,) scalar
          - h_new  (num_layers, 1, hidden_size)
        """
        out, h_new = self.gru(x, h)   # out shape: (1, 1, hidden_size)
        out = out[:, -1, :]          # shape: (1, hidden_size)
        logits = self.policy_head(out)   # (1, action_size)
        value = self.value_head(out).squeeze(-1)  # shape (1,)
        return logits, value, h_new

    def initial_hidden(self, batch_size=1):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

# ------------------------------------------------
# 3. GAE helper
# ------------------------------------------------
def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    """
    GAE-lambda advantage calculation.
    rewards, values: 1D numpy arrays
    We assume a terminal value=0 (like after done).
    """
    values = np.append(values, 0.0)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns

# ------------------------------------------------
# 4. PPO Training with Hidden-State Detach
# ------------------------------------------------
def train_ppo(
    env_class=MouseBehaviorEnvTemporalReward,
    freq_noise=0.2,
    go_prob=0.5,
    num_episodes=1000,
    hidden_size=32,
    num_layers=2,
    lr=1e-3,
    gamma=0.99,
    lam=0.95,
    clip_ratio=0.2,
    train_iters=4,
    batch_size=999999,
    value_loss_coeff=0.5,
    entropy_coeff=0.01,
    eps=0.2,
    step_ms=100
):
    model = ActorCriticGRU(
        input_size=4, hidden_size=hidden_size, num_layers=num_layers
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    success_history = []

    for episode in range(num_episodes):
        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes}")

        print("----------------------------")

        # ---- (1) Collect a single trajectory ----
        go_tone=np.random.rand() < go_prob
        env = env_class(go_tone, step_ms=step_ms, noise_std=freq_noise)

        obs_list = []
        act_list = []
        rew_list = []
        val_list = []
        logp_list = []
        hid_list = []

        h = model.initial_hidden(batch_size=1)

        obs_np = env.reset()
        obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).unsqueeze(0).to(device)
        done = False

        while not done:
            # Forward pass
            logits, value, h_new = model(obs_t, h)

            dist = torch.distributions.Categorical(logits=logits)

                        # ---- Forced exploration logic ----
            # If we are in response window, go tone, etc. => force action=1 with probability eps
            # if np.random.rand() < eps and getattr(env, 'in_response_window', True) and getattr(env, 'go_tone', True):
            #     action = torch.tensor([1], device=device)  # forced "press"
            # else:
            action = dist.sample()  # normal sample from policy
            # ----------------------------------

            logp = dist.log_prob(action)

            # step environment
            obs_next, reward, done, _ = env.step(action.item())

            # store
            obs_list.append(obs_t)
            act_list.append(action)
            rew_list.append(reward)
            val_list.append(value.item())
            logp_list.append(logp.item())
            hid_list.append(h)

            # Detach hidden state so we don't build a giant graph over the entire episode
            h = h_new.detach()

            if not done:
                obs_np = obs_next
                obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).unsqueeze(0).to(device)
            else:
                obs_t = None

        # Evaluate success
        if env.go_tone and env.has_pressed and env.final_decided:
            success = 1
        elif (not env.go_tone) and (not env.has_pressed) and env.final_decided:
            success = 1
        else:
            success = 0
        success_history.append(success)

        # ---- (2) Convert to Tensors / Numpy for training ----
        acts = torch.stack(act_list).squeeze(-1)  # shape (T,)
        rews = np.array(rew_list, dtype=np.float32)
        vals = np.array(val_list, dtype=np.float32)

        # ---- (3) Compute advantages & returns (GAE) ----
        advantages, returns = compute_advantages(rews, vals, gamma=gamma, lam=lam)
        advantages_t = torch.from_numpy(advantages).float().to(device)
        returns_t    = torch.from_numpy(returns).float().to(device)

        # "Old" log probs from data collection
        old_logp_t = torch.tensor(logp_list, dtype=torch.float32, device=device)

        # ---- (4) PPO update: multiple epochs ----
        T = len(acts)
        idxs = np.arange(T)

        for _ in range(train_iters):
            np.random.shuffle(idxs)
            start = 0
            while start < T:
                end = min(start + batch_size, T)
                mb_idx = idxs[start:end]

                mb_obs = [obs_list[i]  for i in mb_idx]
                mb_hid = [hid_list[i] for i in mb_idx]
                mb_acts = acts[mb_idx]
                mb_old_logp = old_logp_t[mb_idx]
                mb_adv = advantages_t[mb_idx]
                mb_ret = returns_t[mb_idx]

                # Recompute "new" log probs & values with current model params
                new_logps = []
                new_vals  = []
                for i, obs_i in enumerate(mb_obs):
                    h_i = mb_hid[i]
                    # Build new graph each time
                    logits_i, val_i, _ = model(obs_i, h_i)
                    dist_i = torch.distributions.Categorical(logits=logits_i)
                    a_i = mb_acts[i].unsqueeze(0)  # shape (1,)
                    logp_i = dist_i.log_prob(a_i)
                    new_logps.append(logp_i)
                    new_vals.append(val_i)

                new_logps = torch.stack(new_logps).squeeze(-1)  # shape (batch,)
                new_vals  = torch.stack(new_vals).squeeze(-1)   # shape (batch,)

                ratio = torch.exp(new_logps - mb_old_logp)
                adv_det = mb_adv.detach()  # advantage is just a buffer, safe to detach
                surr1 = ratio * adv_det
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_det
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (mb_ret - new_vals).pow(2).mean()

                # (Optional) approximate distribution entropy
                with torch.no_grad():
                    # We'll do a new distribution for entropy
                    dist_ent = torch.distributions.Categorical(logits=new_logps.unsqueeze(1).detach())
                    entropy_loss = dist_ent.entropy().mean()

                loss = policy_loss + value_loss_coeff*value_loss - entropy_coeff*entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                start = end

    return model, success_history

# ------------------------------------------------
# 5. Helper
# ------------------------------------------------
def moving_average(data, window_size=20):
    if len(data) < window_size:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

def test_noise_levels(model, noise_levels, n_trials=100, go_prob=0.5, freq_baseline_noise=0.1):
    """
    Evaluate 'model' at each noise level by running 'n_trials' episodes
    of MouseBehaviorEnvTemporalRewardWithNoise. 
    Return a list of success rates [0..1].
    """
    model.eval()
    success_rates = []

    for noise in noise_levels:
        successes = 0
        print(f"Testing noise level: {noise:.2f}")
        for _ in range(n_trials):
            env = MouseBehaviorEnvTemporalReward(
                go_tone=(np.random.rand() < go_prob),
                noise_std=noise + freq_baseline_noise
            )
            h = model.initial_hidden(batch_size=1)
            obs_np = env.reset()
            obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).unsqueeze(0).to(device)
            done = False

            while not done:
                with torch.no_grad():
                    logits, value, h_new = model(obs_t, h)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    obs_next, reward, done, _ = env.step(action.item())
                    h = h_new
                    if not done:
                        obs_t = torch.from_numpy(obs_next).float().unsqueeze(0).unsqueeze(0).to(device)
                    else:
                        obs_t = None

            # Check if this trial was a success
            if env.go_tone and env.has_pressed and env.final_decided:
                successes += 1
            elif (not env.go_tone) and (not env.has_pressed) and env.final_decided:
                successes += 1

        success_rate = successes / n_trials
        success_rates.append(success_rate)

    return success_rates


def main_test_noise_curve(freq_baseline_noise=0.1):
    # Load the model that was trained at noise=0.1
    model = ActorCriticGRU(input_size=4, hidden_size=64, num_layers=2, action_size=2).to(device)
    model.load_state_dict(torch.load("./ppo_gru_noise0.2.pth"))
    model.eval()

    # Evaluate at various noise levels from 0 up to 0.3
    noise_levels = np.linspace(0.0, 2.0, 16)  # e.g. [0.0, 0.1, 0.2,...2.0]
    success_rates = test_noise_levels(model, noise_levels, n_trials=250, go_prob=0.5, freq_baseline_noise=freq_baseline_noise)

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(noise_levels, success_rates, marker='o')
    plt.xlabel("Frequency Noise Level (std)")
    plt.ylabel("Success Rate")
    plt.title("Psychometric Curve: Baseline noise=0.2")
    plt.ylim([0,1])
    plt.grid(True)
    plt.show()

# ------------------------------------------------
# 6. Main
# ------------------------------------------------
def main():
    num_episodes = 1000
    model, success_history = train_ppo(
        env_class=MouseBehaviorEnvTemporalReward,
        freq_noise=0.0,
        go_prob=0.5,
        num_episodes=num_episodes,
        hidden_size=64,
        num_layers=2,
        lr=1e-4,
        gamma=0.99,
        lam=0.9,
        clip_ratio=0.2,
        train_iters=8,
        batch_size=256,
        value_loss_coeff=0.5,
        entropy_coeff=0.05,
        eps=0.0,
        step_ms=100
    )

    # Save the trained model
    torch.save(model.state_dict(), "ppo_gru_noise0.1.pth")

    # Plot
    smoothed = moving_average(success_history, window_size=50)
    plt.figure(figsize=(7,4))
    plt.plot(success_history, alpha=0.3, label='Raw Success')
    plt.plot(range(50, len(smoothed)+50), smoothed, color='red', label='Smoothed')
    plt.xlabel("Episode")
    plt.ylabel("Success")
    plt.title("PPO with GRU on Temporal Reward Env")
    plt.legend()
    plt.grid(True)
    plt.show()

# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    # 1) Train with noise=0.1
    main()
    # freq_baseline_noise = 0.2
    # # 2) Then test with a range of noise levels
    # main_test_noise_curve(freq_baseline_noise=freq_baseline_noise)
