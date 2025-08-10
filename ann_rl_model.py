import numpy as np
import matplotlib.pyplot as plt
import copy

def moving_average(data, window_size):
    if len(data) < window_size:
        return np.mean(data) if len(data) > 0 else 0
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

# Simulation parameters
dt = 0.01
tau_Ca = 1.0
gamma = 0.09
eta = 0.01
num_steps = 1000
window_size = 20

# Astrocyte parameters
adenosine_gain = 0.5
delta_threshold = 0

# Training phases
phase_1_criterion = 50        # 50 consecutive correct go-trials needed to switch to Phase 2
phase_1_hit_requirement = 0.8 # 80% hit rate for last 50 trials
in_phase_1 = True

# Initialize weights
w = np.random.uniform(1, 3, 2) # [0.2, 0.1] 

# Value function weights
w_v = np.zeros(1)
alpha_v = 0.01
beta = 5.0

cumu_reward = 0.0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Start from a neutral baseline for V
V = 0.0
V_new = 0.0

Ca_astro = 0.1
delta = 0.6

# Data tracking
w_history = []
delta_history = []
reward_history = []
action_history = []
cumu_reward_history = []
Ca_astro_history = []
Adenosine_history = []
phase_history = []

# Performance tracking for Phase 1
recent_go_actions = []
phase_1_switch = False

for t in range(num_steps):
    # Decide if we are in Phase 1 or Phase 2
    # Phase 1: Only go trials until performance criterion met
    if in_phase_1:
        # Only go trials
        tone_type = 'go' 
        # w[1] = np.copy(w[0])
    else:
        # Phase 2: Mixed go and no-go
        # 50% chance go or no-go
        tone_type = 'go' if np.random.rand() < 0.2 else 'no-go'

    # Present input
    if tone_type == 'go':
        # go tone
        x = np.array([1.0, 0.0])   # go input pattern
        correct_action = 1.0        # should press lever
    else:
        # no-go tone
        x = np.array([0.0, 1.0])   # no-go input pattern
        correct_action = 0.0      # should not press lever

    # Decay astrocyte Ca
    Ca_astro += dt * (-beta * Ca_astro / tau_Ca)
    Ca_astro = max(Ca_astro, 0.0)

    Adenosine = Ca_astro * adenosine_gain

    # Apply dampening only for no-go trials on the no-go synapse
    # For the go trial, we do not dampen.
    # if tone_type == 'no-go':
        # Only dampen the part of the input from the no-go synapse (w[1]*x[1])
        # go component: w[0]*x[0]
        # no-go component: (w[1]*x[1]) / (1.0 + Adenosine)
        # go trial: no dampening applied
    y_input = np.dot(w, x)

    y = sigmoid(y_input)

    # Decide action
    # If in Phase 1, a large window is allowed initially (but let's assume
    # we are just simulating discrete steps. You might later incorporate timing.)
    # For simplicity, action decision is immediate:
    action = 1.0 if y >= 0.5 else 0.0

    # Compute reward
    # go trial: press = +1 (hit), no press = 0 (miss)
    # no-go trial: press = -1 (false alarm), no press = 0 (correct rejection)
    if tone_type == 'go':
        if action == 1.0:
            r = 1.0
        else:
            r = 0.0
    else:
        if action == 1.0:
            r = -1.0
        else:
            r = 0.0

    # Compute value
    V_new += alpha_v * delta

    # TD error
    delta = r + gamma * V_new - V
    delta = np.round(delta, 2)

    # Update value function weights
    
    V = V_new

    # Surprising outcomes increase Ca_astro
    if abs(delta) > delta_threshold and t % 50 == 0:
        Ca_astro += abs(delta) * 10

    # Update synaptic weights
    w = np.add(w, eta * x * y * delta * Ca_astro) #, float(eta * x[1] * y)])
  

    cumu_reward += r

    if t % 100 == 0:
        print(f"Step {t}, Phase: {'1' if in_phase_1 else '2'}, Action: {action:.4f}, Weight: {w}, reward: {r}, y: {y}, delta: {delta}, mean_reward_last50: {np.mean(reward_history[-50:]):.2f}")

    # Store data
    w_history.append(w.copy())
    delta_history.append(float(delta))
    reward_history.append(r)
    action_history.append(action)
    cumu_reward_history.append(cumu_reward)
    Ca_astro_history.append(Ca_astro)
    Adenosine_history.append(Adenosine)
    phase_history.append(1 if in_phase_1 else 2)

    # Compute moving averages
    reward_moving_avg = moving_average(reward_history, window_size)

    # Check phase 1 performance
    if in_phase_1 and tone_type == 'go':
        recent_go_actions.append((action == 1.0))
        if len(recent_go_actions) > 50:
            recent_go_actions.pop(0)
        # Check if we have 50 consecutive trials with at least 80% hit
        if len(recent_go_actions) == 50 and np.mean(recent_go_actions) >= phase_1_hit_requirement:
            # Move to phase 2
            in_phase_1 = False
            print("Transitioning to Phase 2 at step:", t)


    # Print progress
    if t % 500 == 0:
        print(f"Step {t}, Phase: {'1' if in_phase_1 else '2'}, last_reward: {r}, delta: {delta}, mean_reward_last50: {np.mean(reward_history[-50:]):.2f}") # avg_delta: {avg_delta:.4f}

# Convert history lists to arrays for easier indexing
w_history = np.array(w_history)           # Shape: (num_steps, 2)
# Ca_astro_history = np.array(Ca_astro_history)           # Shape: (num_steps, 2)
#Adenosine_history = np.array(Adenosine_history)         # Shape: (num_steps,)
delta_history = np.array(delta_history)   # Shape: (num_steps,)
reward_moving_avg = np.array(reward_moving_avg) # Shape: (num_steps,)
action_history = np.array(action_history) # Shape: (num_steps,)
cumu_reward_history = np.array(cumu_reward_history) # Shape: (num_steps,)


# Plot astrocytic influence over time
plt.figure(figsize=(12, 6))
plt.plot(Ca_astro_history, label='Astrocyte Modulation Synapse 1')
# plt.plot(a_history[:, 1], label='Astrocyte Modulation Synapse 2')
plt.title('Astrocytic Influence over Time')
plt.xlabel('Time Steps')
plt.ylabel('Astrocyte Modulation (a)')
plt.legend()
plt.show()

# Plot synaptic weights over time
plt.figure(figsize=(12, 6))
plt.plot(w_history[:, 0], label='Weight w1')
plt.plot(w_history[:, 1], label='Weight w2')
plt.title('Synaptic Weights over Time')
plt.xlabel('Time Steps')
plt.ylabel('Weight Value')
plt.legend()
plt.show()


# Plot reward prediction error over time
plt.figure(figsize=(12, 6))
plt.plot(delta_history, label='Reward Prediction Error (delta)')
plt.title('Reward Prediction Error over Time')
plt.xlabel('Time Steps')
plt.ylabel('Delta')
plt.legend()
plt.show()

# Plot moving averages for rewards
plt.figure(figsize=(12, 6))
plt.plot(reward_moving_avg, label='Reward Moving Average')
plt.title('Reward Moving Average')
plt.xlabel('Time Steps')
plt.ylabel('Reward')
plt.legend()
plt.grid()
plt.show()

# Plot moving averages for actions and correct actions
plt.figure(figsize=(12, 6))
plt.plot(cumu_reward_history, label='Cumulative Reward', alpha=0.7)
plt.title('Cumulative Reward')
plt.xlabel('Time Steps')
plt.ylabel('Action (0 or 1)')
plt.legend()
plt.grid()
plt.show()
