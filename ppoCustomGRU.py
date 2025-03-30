import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from drummond_env import MouseBehaviorEnvTemporalReward  # Assuming this is the environment class
from CustomGRU import CustomActorCriticGRU2Layer  

def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    values = np.append(values, 0.0)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns


def train_ppo(
    env_class,  # e.g. MouseBehaviorEnvTemporalReward
    freq_noise=0.2,
    go_prob=0.5,
    num_episodes=1000,
    hidden_size=32,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use our custom 2-layer GRU from scratch:
    model = CustomActorCriticGRU2Layer(
        input_size=4,
        hidden_size=hidden_size,
        action_size=2,  # discrete actions: [0=hold, 1=press]
        device=device
    )

    # We can now create an Adam optimizer from model.parameters
    optimizer = optim.Adam(model.parameters, lr=lr)

    success_history = []

    for episode in range(num_episodes):
        if (episode+1) % 100 == 0:
            print(f"Trial {episode+1}/{num_episodes}")
            print(f"eps: {eps:.3f}")

        go_tone=np.random.rand() < go_prob
        env = env_class(go_tone=go_tone, step_ms=step_ms, noise_std=freq_noise)

        # We'll store trajectory data
        obs_list  = []
        act_list  = []
        rew_list  = []
        val_list  = []
        logp_list = []
        hid_list  = []

        # Initialize hidden states (2 layers)
        h = model.initial_hidden(batch_size=1)  # shape(2,1,hidden_size)

        obs_np = env.reset()
        obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).unsqueeze(0).to(device)
        done = False

        while not done:
            # Single-step forward
            logits, value, h_new = model.forward(obs_t, h)

            dist = torch.distributions.Categorical(logits=logits)

            # forced exploration
            if (np.random.rand() < eps and 
                getattr(env, 'in_response_window', True) and
                getattr(env, 'go_tone', True)):
                action = torch.tensor([1], device=device)
            else:
                action = dist.sample()

            logp = dist.log_prob(action)

            obs_next, reward, done, _ = env.step(action.item())

            obs_list.append(obs_t)
            act_list.append(action)
            rew_list.append(reward)
            val_list.append(value.item())
            logp_list.append(logp.item())
            hid_list.append(h)

            # Detach hidden to avoid building large graph
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

        # Convert trajectory data -> Tensors
        acts = torch.stack(act_list).squeeze(-1)  # shape (T,)
        rews = np.array(rew_list, dtype=np.float32)
        vals = np.array(val_list, dtype=np.float32)

        # GAE advantages
        advantages, returns = compute_advantages(rews, vals, gamma=gamma, lam=lam)
        advantages_t = torch.from_numpy(advantages).float().to(device)
        returns_t    = torch.from_numpy(returns).float().to(device)
        old_logp_t   = torch.tensor(logp_list, dtype=torch.float32, device=device)

        T = len(acts)
        idxs = np.arange(T)

        # PPO multiple epochs
        for _ in range(train_iters):
            np.random.shuffle(idxs)
            start = 0
            while start < T:
                end = min(start + batch_size, T)
                mb_idx = idxs[start:end]

                mb_obs = [obs_list[i] for i in mb_idx]
                mb_hid = [hid_list[i] for i in mb_idx]
                mb_acts = acts[mb_idx]
                mb_old_logp = old_logp_t[mb_idx]
                mb_adv = advantages_t[mb_idx]
                mb_ret = returns_t[mb_idx]

                new_logps = []
                new_vals = []
                for i, obs_i in enumerate(mb_obs):
                    h_i = mb_hid[i]
                    logits_i, val_i, _ = model.forward(obs_i, h_i)
                    dist_i = torch.distributions.Categorical(logits=logits_i)
                    a_i = mb_acts[i].unsqueeze(0)
                    logp_i = dist_i.log_prob(a_i)

                    new_logps.append(logp_i)
                    new_vals.append(val_i)

                new_logps = torch.stack(new_logps).squeeze(-1)
                new_vals  = torch.stack(new_vals).squeeze(-1)

                ratio = torch.exp(new_logps - mb_old_logp)
                adv_det = mb_adv.detach()
                surr1 = ratio * adv_det
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_det
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (mb_ret - new_vals).pow(2).mean()

                with torch.no_grad():
                    dist_ent = torch.distributions.Categorical(logits=new_logps.unsqueeze(1).detach())
                    entropy_loss = dist_ent.entropy().mean()

                loss = policy_loss + value_loss_coeff*value_loss - entropy_coeff*entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters, 5.0)
                optimizer.step()

                start = end

        eps = eps * 0.99  # decay exploration

    return model, success_history

# optional plotting helper
def moving_average(data, window_size=20):
    if len(data) < window_size:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

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
        # num_layers=2,
        lr=1e-3,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        train_iters=4,
        batch_size=256,
        value_loss_coeff=0.5,
        entropy_coeff=0.01,
        eps=0.1,
        step_ms=100
    )

    # Save the trained model
    # torch.save(model.state_dict(), "ppo_gru_noise0.1.pth")


    # Plot
    smoothed = moving_average(success_history, window_size=50)
    plt.figure(figsize=(7,4))
    plt.plot(success_history, alpha=0.3, label='Raw Success')
    plt.plot(range(50, len(smoothed)+50), smoothed, color='red', label='Smoothed')
    plt.xlabel("Trials")
    plt.ylabel("Success")
    plt.title("PPO with GRU on Temporal Reward Env")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()