import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from NE_astro_env import DrummondGoNoGoEnv

class HebbianRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=8):
        """
        We store:
         - W_ih: weights from input -> hidden
         - W_hh: weights from hidden -> hidden
         - b_h: bias
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Use nn.Parameter so that PyTorch tracks them as 'weights'
        # But we won't do backprop in this example; we manually update them.
        self.W_ih = nn.Parameter(torch.randn(hidden_size, input_size)*0.1)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size)*0.1)
        self.b_h  = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, x, h_prev):
        """
        x: shape (input_size,)
        h_prev: shape (hidden_size,)
        Returns new hidden state h
        """
        # Convert to PyTorch tensors if they're numpy
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        if not isinstance(h_prev, torch.Tensor):
            h_prev = torch.tensor(h_prev, dtype=torch.float)
        
        # RNN update: h = tanh(W_ih x + W_hh h_prev + b_h)
        h = (self.W_ih @ x) + (self.W_hh @ h_prev) + self.b_h
        h = torch.tanh(h)
        return h
    
    def get_weights(self):
        """ Return raw references to the weight matrices (as numpy if desired). """
        return self.W_ih.data, self.W_hh.data, self.b_h.data


def integrate_astrocyte_state(a_init, rpe, dt=0.1, T=5.0):
    """
    Integrate da/dt = -a + rpe, for t in [0..T].
    rpe is treated as constant during the ITI.
    """
    a = a_init
    steps = int(T / dt)
    for _ in range(steps):
        da = -a + rpe
        a += da * dt
    return a

def hebbian_update(W, post, pre, astro_a, alpha, dt):
    """
    W: shape (post_dim, pre_dim)
    post: shape (post_dim,)
    pre: shape (pre_dim,)
    astro_a: scalar float (astrocyte activity)
    alpha: learning rate
    dt: time step for integration
    """
    # Outer product: post x pre^T
    # We interpret 'post' and 'pre' as 1D vectors => Delta W is a matrix
    post_np = post.detach().cpu().numpy() if isinstance(post, torch.Tensor) else post
    pre_np  = pre.detach().cpu().numpy()  if isinstance(pre, torch.Tensor)  else pre
    
    dw = alpha * astro_a * np.outer(post_np, pre_np) * dt
    
    # Update W in place
    with torch.no_grad():
        W += torch.from_numpy(dw).float()
    return W


def run_hebbian_astro_model(num_episodes=20, max_trials_per_episode=50,
                            alpha_hebb=0.01, dt=0.1, T_iti=5.0, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment + model
    env = DrummondGoNoGoEnv(max_trials=max_trials_per_episode)
    model = HebbianRNN(input_size=2, hidden_size=8)
    
    # Initialize astrocyte value (persistent across trials)
    a_astro = 0.0
    
    # Lists to store time series
    astro_vals = []       # track astro over time
    weight_norms_ih = []  # track norm of W_ih
    weight_norms_hh = []  # track norm of W_hh
    performance_list = [] # track total reward per episode
    
    trial_counter = 0     # global trial index (across episodes)
    
    for ep in range(num_episodes):
        obs = env.reset()
        h   = np.zeros(model.hidden_size, dtype=np.float32)
        
        episode_reward = 0
        
        while not env.done:
            trial_counter += 1
            
            # Forward pass
            x_t = obs
            h_prev = h
            h_t = model.forward(x_t, h_prev)  # shape(8,)
            
            # For demonstration, pick random action
            action = np.random.randint(2)
            
            # Step environment
            next_obs, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # "RPE" = reward (toy example)
            rpe = reward
            
            # Integrate astrocyte across 5-second ITI (carrying over a_astro)
            a_astro = integrate_astrocyte_state(a_astro, rpe, dt=dt, T=T_iti)
            
            # We do Hebbian update for (W_ih, W_hh)
            # post = h_t, pre_ih = x_t, pre_hh = h_prev
            with torch.no_grad():
                model.W_ih = hebbian_update(
                    model.W_ih,
                    post=h_t,
                    pre=x_t,
                    astro_a=a_astro,
                    alpha=alpha_hebb,
                    dt=T_iti
                )
                model.W_hh = hebbian_update(
                    model.W_hh,
                    post=h_t,
                    pre=h_prev,
                    astro_a=a_astro,
                    alpha=alpha_hebb,
                    dt=T_iti
                )
            
            # Store variables for plotting
            W_ih_data, W_hh_data, _ = model.get_weights()
            weight_norms_ih.append(torch.norm(W_ih_data).item())
            weight_norms_hh.append(torch.norm(W_hh_data).item())
            astro_vals.append(a_astro)
            
            # Next step
            obs = next_obs
            h   = h_t.detach().numpy()
        
        performance_list.append(episode_reward)
        # Next episode
    
    # Done training; now let's do some plots
    plot_results(astro_vals, weight_norms_ih, weight_norms_hh, performance_list, max_trials_per_episode)


def plot_results(astro_vals, wih_vals, whh_vals, performance_list, max_trials_per_episode):
    """
    Make 3 subplots:
      1) Astrocyte value over trials
      2) Weight norm (W_ih, W_hh) over trials
      3) Performance (sum of rewards) vs. episode
    """
    # 1) Setup figure
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    
    # 2) Astro
    axes[0].plot(astro_vals, label='Astrocyte value')
    axes[0].set_title("Astrocyte value vs. trial")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Astrocyte value")
    axes[0].legend()
    
    # 3) Weight norms
    axes[1].plot(wih_vals, label='||W_ih||')
    axes[1].plot(whh_vals, label='||W_hh||')
    axes[1].set_title("Weight norms vs. trial")
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("Frobenius norm")
    axes[1].legend()
    
    # 4) Performance vs. episode
    # Each episode is "max_trials_per_episode" environment steps
    # performance_list: total reward in each episode
    episodes = range(1, len(performance_list)+1)
    axes[2].plot(episodes, performance_list, marker='o')
    axes[2].set_title("Performance vs. episode")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Sum of rewards")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_hebbian_astro_model(
        num_episodes=30,
        max_trials_per_episode=5,
        alpha_hebb=0.01,
        dt=0.1,
        T_iti=5.0,
        seed=42
    )
