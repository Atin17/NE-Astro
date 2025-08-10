import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from drummond_env import MouseBehaviorEnvTemporalReward

# ------------------------------------------------
# 1. Choose a Device (GPU if available)
# ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------------------
# 2. StackedPolicyRNN
# ------------------------------------------------
class StackedPolicyRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=2, output_size=2):
        super(StackedPolicyRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Use an RNN with multiple layers, batch_first=True
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        """
        x shape: (batch_size=1, seq_len=1, input_size=4)
        h shape: (num_layers, batch_size=1, hidden_size)
        Output:
          - logits shape (1, output_size)
          - updated hidden state shape (num_layers, 1, hidden_size)
        """
        out, h_new = self.rnn(x, h)      # out: (1, 1, hidden_size)
        out = out.squeeze(1)            # now shape (1, hidden_size)
        logits = self.fc(out)           # shape (1, output_size)
        return logits, h_new

# ------------------------------------------------
# 3. Helper function to discount rewards
# ------------------------------------------------
def discount_rewards(rewards, gamma=0.99):
    """
    Compute discounted returns for a list of rewards:
      G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    """
    discounted = []
    running_sum = 0.0
    for r in reversed(rewards):
        running_sum = r + gamma * running_sum
        discounted.append(running_sum)
    discounted.reverse()
    return discounted

# ------------------------------------------------
# 4. Policy Gradient Training (with entropy bonus)
# ------------------------------------------------
def train_policy_gradient(
    go_prob=0.5,
    num_layers=2, 
    num_episodes=1000, 
    gamma=0.99, 
    lr=1e-3, 
    hidden_size=32,
    alpha=0.01,  # entropy coefficient
    step_ms=100
    ):
    # Instantiate policy, move it to GPU if available
    policy = StackedPolicyRNN(
        input_size=4,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=2
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    success_history = []

    for episode in range(num_episodes):

        if episode % 100 == 0:
            print(f"Episode {episode}")

        # Randomly pick go vs. no-go
        go_tone = np.random.rand() < go_prob
        env = MouseBehaviorEnvTemporalReward(go_tone=go_tone, step_ms=step_ms)

        obs_list  = []
        act_list  = []
        logp_list = []
        rew_list  = []
        ent_list  = []  # store entropies

        # Initialize hidden state on GPU
        # shape: (num_layers, batch_size=1, hidden_size)
        h = torch.zeros((num_layers, 1, hidden_size), device=device)

        # Reset environment
        obs = env.reset()        # shape (4,)
        # Move obs to GPU
        obs = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(1).to(device)  
        # shape => (1,1,4)

        done = False
        while not done:
            # Forward pass through stacked RNN
            logits, h = policy(obs, h)  # logits: (1,2), h: (num_layers,1,hidden_size)

            # Sample an action from the categorical distribution (on GPU)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()  # shape (1,)

            # Step the environment (CPU). We'll get numpy arrays back.
            next_obs, reward, done, _ = env.step(action.item())

            # Convert reward to a Python float, store in rew_list
            rew_list.append(float(reward))

            # Compute log prob & entropy on GPU
            logp = dist.log_prob(action)  # shape (1,)
            ent = dist.entropy()          # shape (1,)
            logp_list.append(logp)
            ent_list.append(ent)

            if not done:
                next_obs = torch.from_numpy(next_obs).float().unsqueeze(0).unsqueeze(1).to(device)
                obs = next_obs
            else:
                obs = None

        # Evaluate success:
        if go_tone and env.has_pressed:
            success = 1
        elif (not go_tone) and (not env.has_pressed):
            success = 1
        else:
            success = 0
        success_history.append(success)

        # Compute discounted returns (CPU or GPU is fine; let's do GPU)
        returns = discount_rewards(rew_list, gamma=gamma)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # Normalize returns (optional but common)
        if len(returns) > 1:
            mean = returns.mean()
            std = returns.std() + 1e-6
            returns = (returns - mean) / std

        # Build logp / ent tensor (move them to same device)
        logp_tensor = torch.stack(logp_list).to(device)  # shape (#timesteps,)
        ent_tensor = torch.stack(ent_list).to(device)

        # Policy gradient loss
        pg_loss = - torch.sum(logp_tensor * returns)

        # Entropy bonus
        ent_loss = torch.sum(ent_tensor)
        loss = pg_loss - alpha * ent_loss

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        # Optional gradient clipping
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        optimizer.step()

    return success_history

# ------------------------------------------------
# 5. Simple Moving Average
# ------------------------------------------------
def moving_average(data, window_size=20):
    if len(data) < window_size:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

# ------------------------------------------------
# 6. Main
# ------------------------------------------------
def main():
    num_episodes = 2000
    hidden_size = [128]  # or [64, 128, 256] if you want multiple runs
    num_layers = 2

    for hs in hidden_size:
        success_history = train_policy_gradient(
            go_prob=0.5,
            num_layers=num_layers,
            num_episodes=num_episodes,
            gamma=0.99,
            lr=4e-3,
            hidden_size=hs,
            alpha=0.01,
            step_ms=100
        )

        # Compute running average success with window=50
        if len(success_history) >= 50:
            smoothed = moving_average(success_history, window_size=50)
        else:
            smoothed = success_history  # not enough data to average

        plt.plot(smoothed, label=f"Hidden={hs}")

    plt.axhline(y=0.5, color='black', linestyle='-')
    plt.xlabel("Trials")
    plt.ylabel("Success (moving avg)")
    plt.title("Policy Gradient Training")
    plt.grid(True)
    plt.ylim([0,1])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
