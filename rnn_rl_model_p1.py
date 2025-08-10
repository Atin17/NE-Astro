import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from NE_astro_env import DrummondGoNoGoEnv


class RNNQNetwork(nn.Module):
    """
    A recurrent Q-network:
      - RNNCell: input_size -> hidden_size
      - Linear: hidden_size -> Q-values (2 actions)
    """
    def __init__(self, input_size=2, hidden_size=16, output_size=2):
        super().__init__()
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.fc_out   = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        """
        x: (batch_size, input_size)
        h: (batch_size, hidden_size)
        returns: q_values, new_hidden
        """
        new_h = self.rnn_cell(x, h)        # shape: (batch_size, hidden_size)
        q_values = self.fc_out(new_h)      # shape: (batch_size, output_size)
        return q_values, new_h

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.rnn_cell.hidden_size)


def integrate_astrocyte(a_init, rpe, dt=0.1, T=5.0):
    """
    Numerically integrate da/dt = -a + rpe from t=0..T
    (Euler's method), treating rpe as constant over [0..T].
    """
    a = a_init
    steps = int(T / dt)
    for _ in range(steps):
        da = -a + rpe
        a += da * dt
    return a

def hebbian_update(W, post, pre, astro_a, alpha, rpe, T, dt, astro_vals):
    """
    W: shape (post_dim, pre_dim)
    post: shape (post_dim,)
    pre: shape (pre_dim,)
    astro_a: scalar float (astrocyte activity)
    alpha: learning rate
    dt: time step for integration
    """

    steps = int(T / dt)
    # Outer product: post x pre^T
    # We interpret 'post' and 'pre' as 1D vectors => Delta W is a matrix
    post_np = post.detach().cpu().numpy() if isinstance(post, torch.Tensor) else post
    pre_np  = pre.detach().cpu().numpy()  if isinstance(pre, torch.Tensor)  else pre

    astro_a += rpe
    
    for _ in range(steps):
        astro_a += dt * (-astro_a)

        dw = alpha * astro_a * np.outer(post_np, pre_np) * dt
    
        # Update W in place
        with torch.no_grad():
            W += torch.from_numpy(dw).float()

        astro_vals.append(astro_a)
        
    return W, astro_a


def train_td_astro_hebb(
    num_episodes=10,      # how many episodes
    max_steps=5,          # steps per episode
    gamma=0.9,
    lr=1e-3,
    epsilon=0.1,
    alpha_hebb=0.01,
    dt=0.1,
    T_iti=2.0,            # "inter-trial interval" in seconds
    seed=0
):
    """
    Combine:
      - TD learning (Q-learning) with backprop
      - Astrocyte integration ( a(t) ) with RPE = TD error
      - Hebbian update on the RNN hidden->hidden weights, scaled by astro state
      - Track & plot astro, weight norms, and performance
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = DrummondGoNoGoEnv(
        intensities=[5,15,25,35],
        go_freq_id=+1.0,
        nogo_freq_id=-1.0,
        noise_std=0.25,
        reward_hit=+1.0,
        reward_false_alarm=-1.0,
        reward_miss=-1.0,
        reward_correct_reject=1.0,
        max_trials=1000
    )
    model = RNNQNetwork(input_size=2, hidden_size=16, output_size=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # We'll store time-series data:
    astro_vals = []         # astro state after each step
    weight_norms = []       # norm of rnn_cell.weight_hh after each step
    performance_list = []   # sum of rewards in each episode
    trial_counter = 0       # global step index
    astro_val = 0.0  # carry across episodes if you prefer

    # Start training
    for episode in range(num_episodes):
        obs = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float).unsqueeze(0)  # shape (1,2)
        h = model.init_hidden(batch_size=1)                        # shape (1,16)

        episode_reward = 0
        done = False
        

        while not done:
            trial_counter += 1

            # Forward pass => Q-values
            q_values, h_next = model(obs_t, h)
            # Epsilon-greedy action
            if np.random.rand() < epsilon:
                action = np.random.randint(2)
            else:
                action = q_values.argmax(dim=1).item()

            # Step environment
            next_obs, reward, done, _ = env.step(action)
            episode_reward += reward

            next_obs_t = torch.tensor(next_obs, dtype=torch.float).unsqueeze(0)

            # TD target
            with torch.no_grad():
                q_next, _ = model(next_obs_t, h_next)
                max_q_next = q_next.max(dim=1)[0].item()
            td_target = reward + (0.0 if done else gamma * max_q_next)

            predicted_q = q_values[0, action]
            td_error = td_target - predicted_q  # scalar

            # ---------------------------
            # 1) Standard backprop for Q-learning
            # ---------------------------
            loss_td = td_error**2
            optimizer.zero_grad()
            loss_td.backward()
            optimizer.step()

            # ---------------------------
            # 2) Astrocyte update:
            #    integrate da/dt = -a + td_error
            # ---------------------------
            rpe_val = td_error.detach().item()
            # astro_val = integrate_astrocyte(astro_val, rpe_val, dt=dt, T=T_iti)

            # ---------------------------
            # 3) Hebbian update on hidden->hidden
            #    post = h_next, pre = h
            # ---------------------------
            post_vec = h_next[0]  # shape(16,)
            pre_vec  = h[0]       # shape(16,)

            with torch.no_grad():
                # shape of weight_hh: (16,16)
                model.rnn_cell.weight_hh, astro_val = hebbian_update(
                    model.rnn_cell.weight_hh,
                    post_vec,
                    pre_vec,
                    astro_val,
                    alpha_hebb,
                    rpe_val * 10,
                    T_iti,
                    dt,
                    astro_vals
                )

            # ---------------------------
            # Track data for plotting
            # ---------------------------
            # Norm of hidden->hidden weight
            w_hh_norm = torch.norm(model.rnn_cell.weight_hh).item()
            weight_norms.append(w_hh_norm)

            astro_vals.append(astro_val)

            # Next step
            obs_t = next_obs_t
            h = h_next.detach()

            performance_list.append(reward)

    # Final plotting
    plot_results(astro_vals, weight_norms, performance_list, max_steps)

    return performance_list


def plot_results(astro_vals, weight_norms, performance_list, max_steps):
    """
    We'll make 3 subplots:
      1) Astrocyte value over environment steps
      2) Weight norm vs. environment steps
      3) Performance (sum of rewards) vs. episode
    """
    import matplotlib.pyplot as plt

    num_steps_total = len(astro_vals)
    # We know we did 'num_episodes' * 'max_steps' environment steps in total,
    # though in practice some episodes might terminate earlier.

    # Episode indices
    num_episodes = len(performance_list)
    episodes = np.arange(1, num_episodes+1)

    fig, axes = plt.subplots(1, 3, figsize=(15,4))

    # 1) Astro over steps
    axes[0].plot(astro_vals, label='Astro')
    axes[0].set_title("Astrocyte Value vs. Step")
    axes[0].set_xlabel("Environment Step")
    axes[0].set_ylabel("Astro Value")

    # 2) Weight Norm
    axes[1].plot(weight_norms, label='||weight_hh||')
    axes[1].set_title("Weight Norm vs. Step")
    axes[1].set_xlabel("Environment Step")
    axes[1].set_ylabel("Norm of rnn_cell.weight_hh")

    # 3) Performance (sum of rewards) vs. Episode
    axes[2].plot(episodes, performance_list, marker='o', label='Episode Reward')
    axes[2].set_title("Performance vs. Episode")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Sum of Rewards")

    for ax in axes:
        ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    rewards = train_td_astro_hebb(
        num_episodes=1,   # how many episodes
        max_steps=1000,       # steps per episode
        gamma=0.9,
        lr=1e-3,
        epsilon=0.1,
        alpha_hebb=0.1,
        dt=0.1,
        T_iti=5.0,
        seed=42
    )
    print("Training complete. Final 5 episodes' rewards:", rewards[-5:])