import numpy as np
import matplotlib.pyplot as plt
from NE_astro_env import DrummondGoNoGoEnv

class TDActorCriticAstro:
    def __init__(self,
                 obs_dim=2,
                 hidden_dim=16,
                 astro_dim=4,
                 actor_lr=0.01,
                 critic_lr=0.01,
                 astro_lr=0.005,
                 gamma=0.99,
                 astro_decay=0.9):
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.astro_dim = astro_dim
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.astro_lr = astro_lr
        self.astro_decay = astro_decay

        # Actor weights
        self.W_actor1 = 0.1 * np.random.randn(hidden_dim, obs_dim + astro_dim)
        self.b_actor1 = np.zeros(hidden_dim)
        self.W_actor2 = 0.1 * np.random.randn(2, hidden_dim)
        self.b_actor2 = np.zeros(2)

        # Critic weights
        self.W_critic1 = 0.1 * np.random.randn(hidden_dim, obs_dim)
        self.b_critic1 = np.zeros(hidden_dim)
        self.W_critic2 = 0.1 * np.random.randn(1, hidden_dim)
        self.b_critic2 = np.zeros(1)

        # Astro-related
        self.W_astro_td = 0.1 * np.random.randn(astro_dim, 1)
        self.b_astro = np.zeros(astro_dim)

        self.W_actor_astro = 0.1 * np.random.randn(hidden_dim, astro_dim)
        self.b_actor_astro = np.zeros(hidden_dim)

        self.astro_state = np.zeros(astro_dim)
    
    def reset_astro_state(self):
        self.astro_state = np.zeros(self.astro_dim)

    def forward_actor(self, obs):
        combined_inp = np.concatenate([obs, self.astro_state], axis=0)
        h_raw = self.W_actor1 @ combined_inp + self.b_actor1
        h_activated = np.maximum(h_raw, 0)
        logits = self.W_actor2 @ h_activated + self.b_actor2
        return h_activated, logits

    def forward_critic(self, obs):
        h_raw = self.W_critic1 @ obs + self.b_critic1
        h_activated = np.maximum(h_raw, 0)
        val = self.W_critic2 @ h_activated + self.b_critic2
        return h_activated, val[0]

    def sample_action(self, logits):
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        return np.random.choice([0, 1], p=probs), probs

    def update_astro_state(self, td_error):
        z = (self.W_astro_td * td_error).sum() + self.b_astro.sum()
        increment = z if z > 0 else 0.0
        self.astro_state = self.astro_decay * self.astro_state
        self.astro_state += increment

    def update(self, obs, action, td_error, actor_hidden, actor_logits, critic_hidden):
        # Critic update
        # (Same as earlier code, skipping detailed comments for brevity)
        grad_critic_W2, grad_critic_b2, grad_critic_W1, grad_critic_b1 = self._critic_grad(obs, critic_hidden, td_error)

        # Actor update
        grad_actor_W2, grad_actor_b2, grad_actor_W1, grad_actor_b1, grad_actor_Wastro, grad_actor_bastro = \
            self._actor_grad(obs, action, td_error, actor_hidden, actor_logits)

        # Astro param update
        grad_astro_Wtd, grad_astro_b = self._astro_grad(td_error)

        # apply
        self.W_critic2 += self.critic_lr * grad_critic_W2
        self.b_critic2 += self.critic_lr * grad_critic_b2
        self.W_critic1 += self.critic_lr * grad_critic_W1
        self.b_critic1 += self.critic_lr * grad_critic_b1

        self.W_actor2 += self.actor_lr * grad_actor_W2
        self.b_actor2 += self.actor_lr * grad_actor_b2
        self.W_actor1 += self.actor_lr * grad_actor_W1
        self.b_actor1 += self.actor_lr * grad_actor_b1
        self.W_actor_astro += self.actor_lr * grad_actor_Wastro
        self.b_actor_astro += self.actor_lr * grad_actor_bastro

        self.W_astro_td += self.astro_lr * grad_astro_Wtd
        self.b_astro += self.astro_lr * grad_astro_b

    def _critic_grad(self, obs, critic_hidden, td_error):
        # output layer
        d_val_d_W2 = critic_hidden.reshape(-1,1)
        grad_critic_W2 = td_error * d_val_d_W2.T
        grad_critic_b2 = td_error

        # hidden
        mask_hidden = (self.W_critic1 @ obs + self.b_critic1) > 0
        d_val_d_hidden = self.W_critic2[0]  # shape = (hidden_dim,)

        grad_critic_W1 = np.zeros_like(self.W_critic1)
        grad_critic_b1 = np.zeros_like(self.b_critic1)

        for i in range(self.hidden_dim):
            if mask_hidden[i]:
                for j in range(self.obs_dim):
                    grad_critic_W1[i, j] = td_error * d_val_d_hidden[i] * obs[j]
                grad_critic_b1[i] = td_error * d_val_d_hidden[i]

        return grad_critic_W2, grad_critic_b2, grad_critic_W1, grad_critic_b1

    def _actor_grad(self, obs, action, td_error, actor_hidden, actor_logits):
        # softmax
        probs = np.exp(actor_logits - np.max(actor_logits))
        probs /= np.sum(probs)
        dlogpi = -probs
        dlogpi[action] += 1.0

        grad_actor_W2 = np.zeros_like(self.W_actor2)
        grad_actor_b2 = np.zeros_like(self.b_actor2)

        for k in range(2):
            grad_actor_b2[k] = td_error * dlogpi[k]
            for i in range(self.hidden_dim):
                grad_actor_W2[k, i] = td_error * dlogpi[k] * actor_hidden[i]

        d_logpi_d_h = np.zeros(self.hidden_dim)
        for i in range(self.hidden_dim):
            d_logpi_d_h[i] = td_error * (dlogpi @ self.W_actor2[:, i])

        mask_a_hidden = (self.W_actor1 @ np.concatenate([obs, self.astro_state]) + self.b_actor1) > 0

        grad_actor_W1 = np.zeros_like(self.W_actor1)
        grad_actor_b1 = np.zeros_like(self.b_actor1)
        grad_actor_Wastro = np.zeros_like(self.W_actor_astro)
        grad_actor_bastro = np.zeros_like(self.b_actor_astro)

        combined_inp = np.concatenate([obs, self.astro_state])
        for i in range(self.hidden_dim):
            if mask_a_hidden[i]:
                for j in range(self.obs_dim):
                    grad_actor_W1[i, j] = d_logpi_d_h[i] * combined_inp[j]
                for j in range(self.astro_dim):
                    grad_actor_Wastro[i, j] = d_logpi_d_h[i] * combined_inp[self.obs_dim + j]
                grad_actor_b1[i] = d_logpi_d_h[i]

        return grad_actor_W2, grad_actor_b2, grad_actor_W1, grad_actor_b1, grad_actor_Wastro, grad_actor_bastro

    def _astro_grad(self, td_error):
        z = (self.W_astro_td * td_error).sum() + self.b_astro.sum()
        if z > 0:
            grad_astro_Wtd = td_error * np.ones_like(self.W_astro_td)
            grad_astro_b = td_error * np.ones_like(self.b_astro)
        else:
            grad_astro_Wtd = 0.0 * self.W_astro_td
            grad_astro_b = 0.0 * self.b_astro
        return grad_astro_Wtd, grad_astro_b


# ---------------------------------------------------------
# 3) Running single episode with TWO PHASES
# ---------------------------------------------------------
def run_two_phase_experiment(agent,
                             env,
                             phase1_steps=100,
                             phase2_steps=200,
                             noise_std_phase1=0.25,
                             noise_std_phase2=0.25):
    """
    A single 'episode' composed of:
      Phase 1: only go trials (like pre-training)
      Phase 2: normal go/no-go trials.
    We store astro states, selected weights, and performance.
    """
    # Initialize data containers
    all_astro = []
    all_weights = []  # let's store a subset, e.g. actor's W_actor1[0, :]
    rewards_per_trial = []

    # --------------- Phase 1 ---------------
    # we forcibly only present go trials
    # by creating a function that always returns is_go=True
    env.reset()
    env.noise_std = noise_std_phase1

    for step_i in range(phase1_steps):
        # create a forced "go" observation:
        # (freq_val=+1, intensity val random, + noise)
        intensity = np.random.choice(env.intensities)
        intensity_val = intensity / 35.0
        obs = np.array([env.go_freq_id, intensity_val]) + \
              np.random.normal(0, env.noise_std, size=2)
        
        # forward actor & critic
        actor_h, actor_logits = agent.forward_actor(obs)
        critic_h, value_s = agent.forward_critic(obs)
        
        # sample action
        action, _ = agent.sample_action(actor_logits)
        
        # in Phase 1, correct action is always 1 (press).
        # Reward: 1.0 if action=1, else 0.0
        reward = 1.0 if action == 1 else 0.0
        
        # compute next value (since we simulate "end of trial")
        # we won't generate a new observation from env here, but let's do a random one:
        next_obs = np.array([env.go_freq_id, intensity_val]) + \
                   np.random.normal(0, env.noise_std, size=2)
        _, value_s_next = agent.forward_critic(next_obs)

        td_error = reward + agent.gamma * value_s_next - value_s
        
        # update astro
        agent.update_astro_state(td_error)
        # update network
        agent.update(obs, action, td_error, actor_h, actor_logits, critic_h)

        # log
        all_astro.append(agent.astro_state.copy())
        all_weights.append(agent.W_actor1[0,:].copy())  # first row
        rewards_per_trial.append(reward)

    # --------------- Phase 2 ---------------
    # now we do real environment stepping with go/no-go
    env.reset()
    env.noise_std = noise_std_phase2

    for step_i in range(phase2_steps):
        # normal environment step
        obs = env._generate_observation()  # or env.reset if we like, but let's just do step logic
        # forward pass
        actor_h, actor_logits = agent.forward_actor(obs)
        critic_h, value_s = agent.forward_critic(obs)
        action, _ = agent.sample_action(actor_logits)

        # we mimic env.step(action) logic to get reward
        reward = env._calculate_reward(env.current_stim, action)

        # generate next obs to compute TD
        next_obs = env._generate_observation()
        _, value_s_next = agent.forward_critic(next_obs)
        
        td_error = reward + agent.gamma * value_s_next - value_s

        # update astro
        agent.update_astro_state(td_error)
        # update net
        agent.update(obs, action, td_error, actor_h, actor_logits, critic_h)

        # log
        all_astro.append(agent.astro_state.copy())
        all_weights.append(agent.W_actor1[0,:].copy())
        rewards_per_trial.append(reward)

    return np.array(all_astro), np.array(all_weights), np.array(rewards_per_trial)


# ---------------------------------------------------------
# 4) Main Script: Train and Plot
# ---------------------------------------------------------
if __name__ == "__main__":
    # instantiate environment
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

    # instantiate agent
    agent = TDActorCriticAstro(obs_dim=2,
                               hidden_dim=16,
                               astro_dim=4,
                               actor_lr=0.01,
                               critic_lr=0.01,
                               astro_lr=0.005,
                               gamma=0.99,
                               astro_decay=0.9)
    
    agent.reset_astro_state()

    # We'll run one big "episode" with a Phase 1 of 100 steps (only go),
    # then Phase 2 of 300 steps (go/no-go).
    astro_vals, weight_vals, reward_vals = run_two_phase_experiment(
        agent, env,
        phase1_steps=100,
        phase2_steps=500,
        noise_std_phase1=0.2,
        noise_std_phase2=0.25
    )

    # Calculate rolling average reward (performance)
    window_size = 20
    rolling_perf = []
    for i in range(len(reward_vals)):
        start = max(0, i - window_size + 1)
        rolling_perf.append( np.mean(reward_vals[start:i+1]) )
    rolling_perf = np.array(rolling_perf)

    # ---------------------------------------------------------
    # 5) Plot
    # ---------------------------------------------------------
    fig, axs = plt.subplots(3,1, figsize=(8,10), tight_layout=True)

    # (A) Astrocyte states
    axs[0].plot(astro_vals, alpha=0.8)
    axs[0].set_title("Astrocyte State Over Trials (Each Dim)")
    axs[0].set_xlabel("Trial")
    axs[0].set_ylabel("Astro State Value")

    # Add a vertical line indicating end of Phase 1
    phase1_end = 100
    axs[0].axvline(phase1_end, color='k', linestyle='--', label='End of Phase 1')
    axs[0].legend()

    # (B) A subset of the actor's W_actor1 first row
    axs[1].plot(weight_vals[:, :8])  # just 8 columns to visualize
    axs[1].set_title("First Row of Actor W_actor1 (subset of dims)")
    axs[1].set_xlabel("Trial")
    axs[1].set_ylabel("Weight Value")
    axs[1].axvline(phase1_end, color='k', linestyle='--')

    # (C) Performance
    axs[2].plot(reward_vals, '.', alpha=0.3, label="Reward each trial")
    axs[2].plot(rolling_perf, 'r-', label=f"Rolling average reward (win={window_size})")
    axs[2].axvline(phase1_end, color='k', linestyle='--')
    axs[2].set_title("Performance Over Trials")
    axs[2].set_xlabel("Trial")
    axs[2].set_ylabel("Reward")
    axs[2].legend()

    plt.show()