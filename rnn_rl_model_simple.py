import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from NE_astro_env import DrummondGoNoGoEnv

class RNNQNetwork(nn.Module):
    """
    A minimal recurrent Q-network:
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
        new_h = self.rnn_cell(x, h)       # shape: (batch_size, hidden_size)
        q_values = self.fc_out(new_h)     # shape: (batch_size, output_size)
        return q_values, new_h

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.rnn_cell.hidden_size)
    


def train_simple_rnn_td(
    env,
    model,
    num_episodes=100,
    gamma=0.9,
    lr=1e-3,
    epsilon=0.1,
    seed=42
):
    """
    Minimal Q-learning with an RNN.
    - No astrocyte updates
    - No Hebbian updates
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    weight_norms = []
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    all_episode_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float).unsqueeze(0)  # shape: (1,2)
        h = model.init_hidden(batch_size=1)                        # shape: (1, hidden_size)

        done = False
        episode_reward = 0.0

        while not done:
            # 1) forward => Q-values
            q_values, h_next = model(obs_t, h)

            # 2) epsilon-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(2)
            else:
                action = q_values.argmax(dim=1).item()

            # 3) step environment
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward

            next_obs_t = torch.tensor(next_obs, dtype=torch.float).unsqueeze(0)

            # 4) compute TD target
            with torch.no_grad():
                q_next, _ = model(next_obs_t, h_next)
                max_q_next = q_next.max(dim=1)[0].item()
            td_target = reward + (0.0 if done else gamma * max_q_next)

            predicted_q = q_values[0, action]
            td_error = td_target - predicted_q

            # 5) standard TD loss
            loss = td_error**2

            # 6) backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 7) move to next step
            obs_t = next_obs_t
            h = h_next.detach()

            w_hh_norm = torch.norm(model.rnn_cell.weight_hh).item()
            weight_norms.append(w_hh_norm)

            all_episode_rewards.append(reward)

    return all_episode_rewards, weight_norms

if __name__ == "__main__":
    # Create environment
    env = DrummondGoNoGoEnv(
        intensities=[5,15,25,35],
        go_freq_id=+1.0,
        nogo_freq_id=-1.0,
        noise_std=0.25,
        reward_hit=+1.0,
        reward_false_alarm=-1.0,
        reward_miss=0.0,
        reward_correct_reject=0.0,
        max_trials=1000
    )

    # Create model
    model = RNNQNetwork(input_size=2, hidden_size=1, output_size=2)

    # Train
    rewards, weight_norms = train_simple_rnn_td(env, model, num_episodes=1, gamma=0.9, lr=1e-2, epsilon=0.0)
    print("Final 10 episode rewards:", rewards[-10:])
    
    # If desired, plot rewards over episodes
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    # 2) Weight Norm
    axes[0].plot(weight_norms, label='||weight_hh||')
    axes[0].set_title("Weight Norm vs. Step")
    axes[0].set_xlabel("Environment Step")
    axes[0].set_ylabel("Norm of rnn_cell.weight_hh")

    # 3) Performance (sum of rewards) vs. Episode
    axes[1].plot(rewards, marker='o', label='Trial Reward')
    axes[1].set_title("Performance vs. Trials")
    axes[1].set_xlabel("Trials")
    axes[1].set_ylabel("Rewards")

    for ax in axes:
        ax.legend()
    plt.tight_layout()
    plt.show()
