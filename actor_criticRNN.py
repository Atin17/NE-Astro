import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from drummond_env import MouseBehaviorEnvTemporalReward

# Suppose you have your environment that provides a multi-step trial:
# from drummond_env import MouseBehaviorEnvTemporalReward
# For this example, let's assume you have the same or similar environment.

# ------------------------------------------------
# 1. Choose Device (GPU if available)
# ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ------------------------------------------------
# 2. Actor-Critic RNN
# ------------------------------------------------
class ActorCriticRNN(nn.Module):
    """
    A stacked RNN that outputs both:
      - Policy logits (for action selection)
      - A single value prediction (critic) for the state.
    """
    def __init__(self, input_size=4, hidden_size=32, num_layers=2, action_size=2):
        super(ActorCriticRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Recurrent backbone
        self.rnn = nn.RNN(
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
        x shape: (batch=1, seq_len=1, input_size=4)
        h shape: (num_layers, batch_size=1, hidden_size)
        Returns:
          - logits (1, action_size)
          - value  (1,) scalar
          - h_new  (num_layers, 1, hidden_size)
        """
        out, h_new = self.rnn(x, h)  # out shape: (1, 1, hidden_size)
        out = out.squeeze(1)        # shape: (1, hidden_size)

        # Policy head
        logits = self.policy_head(out)  # shape (1, action_size)
        # Value head
        value = self.value_head(out)    # shape (1,1), we squeeze to (1,)
        value = value.squeeze(-1)

        return logits, value, h_new


# ------------------------------------------------
# 3. Helper: Discounted Returns
# ------------------------------------------------
def discount_rewards(rewards, gamma=0.99):
    """
    Convert a list of rewards [r_0, r_1, ... r_{T-1}]
    into discounted returns [G_0, G_1, ... G_{T-1}]
    where G_t = r_t + gamma * r_{t+1} + ...
    """
    discounted = []
    running_sum = 0.0
    for r in reversed(rewards):
        running_sum = r + gamma * running_sum
        discounted.append(running_sum)
    discounted.reverse()
    return discounted


# ------------------------------------------------
# 4. Actor-Critic Training
# ------------------------------------------------
def train_actor_critic(
    env_class,            # constructor for environment
    go_prob=0.5,          # probability
    num_episodes=1000,
    gamma=0.99,
    lr=1e-3,
    hidden_size=32,
    num_layers=2,
    entropy_coeff=0.01,   # alpha
    value_coeff=0.5,      # c in the formula
    step_ms=100
):
    """
    Trains an actor-critic model on the specified environment for a certain number of episodes.
    Each episode is one trial from the environment or the environment runs multiple steps.
    """
    # Instantiate model
    model = ActorCriticRNN(
        input_size=4,
        hidden_size=hidden_size,
        num_layers=num_layers,
        action_size=2
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    success_history = []

    for episode in range(num_episodes):
        if (episode+1) % 100 == 0:
            print(f"Trial {episode+1}/{num_episodes}")

        # Create env
        # e.g. if you want 50% go, do go_tone=np.random.rand()<0.5
        env = env_class(go_tone=(np.random.rand()<go_prob), step_ms=step_ms)

        # Buffers to store episode data
        log_probs = []
        values = []
        rewards = []
        entropies = []

        # Initialize RNN hidden state
        h = torch.zeros((num_layers, 1, hidden_size), device=device)

        obs = env.reset()
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(1).to(device)  # shape (1,1,4)

        done = False

        while not done:
            # Forward
            logits, value, h = model(obs_t, h)  # logits:(1,2), value:(1,), h:(num_layers,1,hidden_size)

            # Sample an action
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()  # shape ()
            # print(action)

            log_prob = dist.log_prob(action)  # shape ()
            entropy = dist.entropy()          # shape ()

            # Step environment
            obs_next, reward, done, _ = env.step(action.item())

            # Store
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(float(reward))
            entropies.append(entropy)

            # Move to next obs
            if not done:
                obs_t = torch.from_numpy(obs_next).float().unsqueeze(0).unsqueeze(1).to(device)
            else:
                obs_t = None

        print("----------")
        # Evaluate success:
        # For example, if environment has .has_pressed or .final_correct, define success:
        # Here, let's guess you have 'env.go_tone' and 'env.has_pressed' as typical:
        #   success = 1 if (go_tone & pressed) or (not go_tone & not pressed)
        #   else 0
        if env.go_tone and env.has_pressed:
            success = 1
        elif (not env.go_tone) and (not env.has_pressed):
            success = 1
        else:
            success = 0
        success_history.append(success)

        # Compute discounted returns
        returns = discount_rewards(rewards, gamma=gamma)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)  # shape (T,)

        # If desired, can normalize the returns. But often with advantage, it's less critical:
        # returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        # Convert everything to Tensors
        log_probs = torch.stack(log_probs)    # shape (T,)
        values    = torch.stack(values).squeeze(-1)  # shape (T,) because each value was shape(1,)
        entropies = torch.stack(entropies)    # shape (T,)

        # Compute advantages: A_t = G_t - V_t
        advantages = returns - values

        # Actor (policy) loss:
        # sum over t of [ -log_prob(a_t) * advantage_t ]
        policy_loss = -(log_probs * advantages.detach()).sum()

        # Critic (value) loss:
        # sum of (advantage^2). We can do MSE or .pow(2).
        value_loss = (advantages ** 2).sum()

        # Entropy bonus:
        entropy_loss = entropies.sum()

        # Combine them
        loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy_loss

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
# 6. Example Main
# ------------------------------------------------
def main():
    num_episodes = 2000
    success_history = train_actor_critic(
        env_class=MouseBehaviorEnvTemporalReward,  # pass your environment
        go_prob=0.5,
        num_episodes=num_episodes,
        gamma=0.99,
        lr=1e-4,
        hidden_size=256,
        num_layers=2,
        entropy_coeff=0.1,
        value_coeff=0.5,
        step_ms=100
    )

    # Smooth success
    smoothed = moving_average(success_history, window_size=50)

    plt.figure(figsize=(6,4))
    plt.plot(success_history, label='success', alpha=0.3)
    plt.plot(range(50,len(smoothed)+50), smoothed, color='red', label='Smoothed')
    plt.xlabel("Trials")
    plt.ylabel("Success")
    plt.title("Actor-Critic with RNN on Temporal Reward Env")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
