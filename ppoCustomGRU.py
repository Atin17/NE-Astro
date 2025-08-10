import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from drummond_env import MouseBehaviorEnvTemporalReward  # Assuming this is the environment class
from CustomGRU import CustomActorCriticGRU2Layer  
from matplotlib.colors import TwoSlopeNorm

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

            # # forced exploration
            # if (np.random.rand() < eps and 
            #     getattr(env, 'in_response_window', True) and
            #     getattr(env, 'go_tone', True)):
            #     action = torch.tensor([1], device=device)
            # else:
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

        print("-" * 20)
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

        # eps = eps * 0.99  # decay exploration

    return model, success_history

# optional plotting helper
def moving_average(data, window_size=20):
    if len(data) < window_size:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

# -----------------------------------------------
# System Identification: "Pinging" Hidden Units
# -----------------------------------------------
def ping_hidden_units(model, num_steps=20, impulse_value=1.0, threshold_ratio=0.37):
    """
    For each hidden unit in the second layer of 'model', inject an impulse
    in its initial hidden state and then simulate the network forward with
    zero input for 'num_steps' steps.
    
    Returns:
      timescales: A numpy array of shape (hidden_size,) where each element
                  is the first time step at which the absolute value of that
                  unit in layer2 decays below threshold_ratio * impulse_value.
                  If it never decays, we return num_steps.
      responses: A numpy array of shape (hidden_size, num_steps) containing
                 the impulse responses.
    """
    hidden_size = model.hidden_size
    timescales = np.zeros(hidden_size, dtype=np.float32)
    responses = np.zeros((hidden_size, num_steps), dtype=np.float32)

    # Create a zero input sequence for simulation: shape (1, num_steps, input_size)
    x_seq = torch.zeros(1, num_steps, model.input_size, device=model.device)

    # For each unit in layer2, create an initial hidden state with an impulse.
    # For simplicity, we leave layer1 at zero.
    for i in range(hidden_size):
        # Initialize h: shape (2, 1, hidden_size)
        h0 = torch.zeros(2, 1, hidden_size, device=model.device)
        # Inject impulse in layer2 at unit i
        h0[0, 0, i] = impulse_value

        # Run the model forward over the entire sequence
        # We'll use forward_sequence to get outputs at each time step.
        _, values_seq, h_final = model.forward_sequence(x_seq, h0)
        # For system identification, we focus on the hidden response in layer2.
        # Unfortunately, forward_sequence doesn't return intermediate hidden states.
        # We'll instead simulate step-by-step manually to record h2.
        h_sim = h0.clone()
        h2_responses = []
        for t in range(num_steps):
            # Get zero input at time step t: shape (1,1,input_size)
            x_t = x_seq[:, t, :].unsqueeze(0)
            _, _, h_sim = model.forward(x_t, h_sim)
            # Record layer2's hidden state (as scalar per unit i)
            # We'll extract unit i's activation from layer2:
            h2_val = h_sim[0, 0, i].item()
            h2_responses.append(h2_val)
        responses[i, :] = np.array(h2_responses)

        # Determine timescale: the first time step where |response| falls below threshold_ratio * impulse_value
        threshold = threshold_ratio * abs(impulse_value)
        ts = num_steps  # default if never decays
        for t, val in enumerate(h2_responses):
            if abs(val) < threshold:
                ts = t
                break
        timescales[i] = ts

    return timescales, responses

# -----------------------------------------------
# Example usage of system identification (pinging)
# -----------------------------------------------
def plot_impulse_responses():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate our custom GRU actor-critic (e.g., with hidden_size=32)
    model = CustomActorCriticGRU2Layer(input_size=4, hidden_size=64, action_size=2, device=device)
    # Optionally, load trained weights here.
    model.load_state_dict(torch.load("ppo_custom_gru_noise0.2.pth"))
    
    num_steps = 20
    impulse_value = 1.0
    threshold_ratio = 0.37  # roughly 1/e

    timescales, responses = ping_hidden_units(model, num_steps=num_steps,
                                               impulse_value=impulse_value,
                                               threshold_ratio=threshold_ratio)
    # Plot histogram of timescales
    plt.figure(figsize=(6,4))
    plt.hist(timescales, bins=range(0, num_steps+1), edgecolor='black')
    plt.xlabel("Time constant (number of steps to decay below threshold)")
    plt.ylabel("Number of units")
    plt.title("Distribution of Impulse Response Timescales in Layer 2")
    plt.grid(True)
    plt.show()

    # Plot impulse response curves for a few example units (e.g., 5 units with highest timescale)
    idx_sorted = np.argsort(timescales)[::-1]  # descending order
    top_units = idx_sorted[:10]
    plt.figure(figsize=(8,5))
    for i in top_units:
        plt.plot(responses[i, :], label=f"Unit {i} (ts={timescales[i]:.1f})")
    plt.xlabel("Time step")
    plt.ylabel("Hidden activation (layer 2)")
    plt.title("Impulse Response Curves of Top 10 Slow Units")
    plt.legend()
    plt.grid(True)
    plt.show()

def collect_responses(model, env_class, n_trials=200, step_ms=100):
    """
    Runs model on env_class for n_trials.  Returns a dict:
       responses['layer1'][trial_type] = list of arrays (T, hidden_size)
       responses['layer2'][trial_type] = same
    trial_type in {'FA','CR','miss','hit'}.
    """
    device = model.device
    responses = {
        'layer1': {t: [] for t in ('FA','CR','miss','hit')},
        'layer2': {t: [] for t in ('FA','CR','miss','hit')}
    }

    for _ in range(n_trials):
        # randomly choose go vs no‐go
        go_tone = (np.random.rand() < 0.5)
        env = env_class(go_tone=go_tone, step_ms=step_ms, noise_std=1.0)
        h = model.initial_hidden(batch_size=1)      # shape (2,1,hidden_size)
        obs = env.reset()

        # collect this trial’s hidden trajectories
        traj_h1 = []
        traj_h2 = []
        done = False
        obs_t = torch.from_numpy(obs).float().view(1,1,-1).to(device)
        while not done:
            with torch.no_grad():
                logits, value, h_new = model.forward(obs_t, h)
            # record old h
            traj_h1.append(h[0,0,:].cpu().numpy())
            traj_h2.append(h[1,0,:].cpu().numpy())

            # step
            a = torch.distributions.Categorical(logits=logits).sample().item()
            obs_next, _, done, _ = env.step(a)

            h = h_new.detach()
            if not done:
                obs_t = torch.from_numpy(obs_next).float().view(1,1,-1).to(device)

        # determine trial type
        if   ( env.go_tone and  env.has_pressed):  trial_type = 'hit'
        elif ( env.go_tone and not env.has_pressed): trial_type = 'miss'
        elif (not env.go_tone and env.has_pressed):  trial_type = 'FA'
        else:                                        trial_type = 'CR'

        # stack and store
        responses['layer1'][trial_type].append( np.stack(traj_h1,axis=0) )
        responses['layer2'][trial_type].append( np.stack(traj_h2,axis=0) )

    return responses

def filter_responses_by_units(resp_dict, keep_idx):
    """
    Returns a deep-copy of `resp_dict` where the layer2 time-series
    have been trimmed to the columns listed in `keep_idx`.

    resp_dict : the structure returned by `collect_responses`
                resp_dict['layer2'][trial_type]  ->  list[ np.ndarray(T, H) ]
    keep_idx  : 1-D numpy array of *column indices* to keep
    ----------------------------------------------------------------
    """
    out = {'layer1': resp_dict['layer1'],       # keep layer-1 untouched
           'layer2': {t: [] for t in resp_dict['layer2']}}

    for t, trials in resp_dict['layer2'].items():
        for arr in trials:                      # arr shape (T, H)
            out['layer2'][t].append(arr[:, keep_idx])
    return out


# ================================================================
# 1.  Classify units ---------------------------------------------
# ================================================================
def split_units_into_fast_slow(resps_all_types,
                               fraction_of_max=0.5,
                               min_frac_active=0.20):
    """
    Concatenate *every* layer-2 time-series over all trial-types,
    then decide slow vs fast with `classify_units_by_threshold_mod`.
    """
    # -- 1a  concatenate over trials & trial-types --
    cat = []        # list of (T_i, H) arrays
    for trials in resps_all_types['layer2'].values():
        cat.extend(trials)
    layer2_big = np.concatenate(cat, axis=0)         # (T_total, H)

    # -- 1b  run the threshold logic you supplied --
    clf, _stats = classify_units_by_threshold_mod(layer2_big,
                                                  fraction_of_max,
                                                  min_frac_active)

    slow_idx = np.where(np.array(clf) == 'slow')[0]
    fast_idx = np.where(np.array(clf) == 'fast')[0]
    return slow_idx, fast_idx

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



# ================================================================
# 2.  Main driver -------------------------------------------------
# ================================================================
def run_and_plot(model, env_class,
                 n_trials   = 200,
                 step_ms    = 100,
                 frac_max   = 0.5,
                 min_frac   = 0.20):
    """Collect activity, split into slow/fast, make both I- and J-plots."""
    # -------------------------------------------------------------
    # 2a  run the network and collect responses
    # -------------------------------------------------------------
    resps = collect_responses(model, env_class,
                              n_trials=n_trials,
                              step_ms =step_ms)

    # -------------------------------------------------------------
    # 2b  identify indices of slow vs fast layer-2 units
    # -------------------------------------------------------------
    slow_idx, fast_idx = split_units_into_fast_slow(resps,
                                                    fraction_of_max=frac_max,
                                                    min_frac_active=min_frac)

    print(f'Identified  {len(slow_idx)}  slow units '
          f'and  {len(fast_idx)}  fast units '
          f'out of {model.hidden_size} total.')

    # -------------------------------------------------------------
    # 2c  build two *filtered* response dictionaries
    # -------------------------------------------------------------
    resps_slow = filter_responses_by_units(resps, slow_idx)
    resps_fast = filter_responses_by_units(resps, fast_idx)

    # -------------------------------------------------------------
    # 2d  make two sets of I/J panels
    # -------------------------------------------------------------
    step_sec = step_ms / 1000.0
    print('---  SLOW units  ---')
    make_IJ_plots(resps_slow, step_sec)

    print('---  FAST units  ---')
    make_IJ_plots(resps_fast, step_sec)

    # make_figure1_IJM_plots(
    #     resps_astrocyte_like=resps_slow['layer2'], # astrocyte-like
    #     resps_neuron_like=resps_fast['layer2'],   # neuron-like
    #     time_step_sec=step_sec)

# ----------------------------------  add this helper  --------------------------
def stack_with_padding(list_of_mats, pad_value=np.nan):
    """
    list_of_mats : list of 2-D arrays with shape (T_i, hidden)
    returns      : 3-D array (n_trials, T_max, hidden) padded with pad_value
    """
    T_max = max(m.shape[0] for m in list_of_mats)
    hidden = list_of_mats[0].shape[1]
    batch  = len(list_of_mats)

    out = np.full((batch, T_max, hidden), pad_value, dtype=list_of_mats[0].dtype)
    for k, m in enumerate(list_of_mats):
        T = m.shape[0]
        out[k, :T, :] = m
    return out
# ------------------------------------------------------------------------------

def right_pad(mat, target_T, fill_val=np.nan):
    """
    Pad `mat` on the right with `fill_val` so that its *time* dimension equals
    `target_T`. Works for either  

        - 2-D array  → shape (n_units , T)
        - 1-D array  → shape (T,)

    Returns a *copy* with the new length; if the length already matches,
    just returns `mat` unchanged.
    """
    if mat.ndim == 1:                           # ---- 1-D vector ----
        T = mat.shape[0]
        if T == target_T:
            return mat
        out = np.full(target_T, fill_val, dtype=mat.dtype)
        out[:T] = mat
        return out

    elif mat.ndim == 2:                         # ---- 2-D raster ----
        n_units, T = mat.shape
        if T == target_T:
            return mat
        out = np.full((n_units, target_T), fill_val, dtype=mat.dtype)
        out[:, :T] = mat
        return out

    else:
        raise ValueError("Only 1-D or 2-D inputs are supported.")
    
    # --- Helper function from your provided code, slightly adapted for plotting ---
def make_figure1_IJM_plots(
    resps_astrocyte_like,
    resps_neuron_like,
    time_step_sec,
    plot_title_suffix_astrocyte=" (Astrocyte-like)",
    plot_title_suffix_neuron=" (Neuron-like)"
    ):
    types  = ['FA','CR','miss','hit'] # Order for panel I: FA, CR, Miss, Hit (top to bottom)
    type_labels_panel_i = {'FA':'false\nalarm', 'CR':'correct\nrejection', 'miss':'miss', 'hit':'hit'} # For y-axis labels
    colors = {'FA':'red','CR':'purple','miss':'gray','hit':'deepskyblue'} # For average traces
    v_min_heatmap, v_max_heatmap = -40, 160 # dF/F range from original Fig 1I colorbar

    # --- Panel I: Astrocyte Responses ---
    if not resps_astrocyte_like or not any(resps_astrocyte_like[t] for t in types):
        print("No Astrocyte-like responses to plot for Panel I.")
    else:
        # Determine the number of astrocyte-like units for consistent y-axis
        num_astrocyte_units = 0
        for t in types:
            if resps_astrocyte_like[t]:
                num_astrocyte_units = resps_astrocyte_like[t][0].shape[1]
                break
        if num_astrocyte_units == 0:
            print("Astrocyte-like responses have 0 units. Skipping Panel I.")
        else:
            gap = 5 # Visual gap between trial type rasters
            astrocyte_raster_rows = []
            astrocyte_per_type_means = {}
            max_len_astrocyte = 0

            # Collect mean rasters and find max length
            for t in types:
                if resps_astrocyte_like[t]:
                    big_astro = stack_with_padding(resps_astrocyte_like[t])
                    if big_astro.size == 0 or big_astro.shape[2] == 0: continue
                    mean_astro = np.nanmean(big_astro, axis=0).T # (units, time_t)
                    astrocyte_per_type_means[t] = mean_astro
                    max_len_astrocyte = max(max_len_astrocyte, mean_astro.shape[1])
                else:
                    # If no trials of this type, create a NaN placeholder raster
                    astrocyte_per_type_means[t] = np.full((num_astrocyte_units, 1), np.nan)


            if max_len_astrocyte == 0:
                print("Max length for astrocyte responses is 0. Skipping Panel I.")
            else:
                time_common_astrocyte = np.arange(max_len_astrocyte) * time_step_sec

                # Build the composite raster for Panel I (top)
                current_y_pos = 0
                y_tick_positions = []
                y_tick_labels = []

                for t_idx, t in enumerate(types): # FA, CR, Miss, Hit order
                    if t in astrocyte_per_type_means and astrocyte_per_type_means[t].size > 0:
                        mat = right_pad(astrocyte_per_type_means[t], max_len_astrocyte, fill_val=0) # Pad with 0 for heatmap
                        astrocyte_raster_rows.append(mat)
                        y_tick_positions.append(current_y_pos + mat.shape[0] / 2)
                        y_tick_labels.append(type_labels_panel_i[t])
                        current_y_pos += mat.shape[0]
                    else: # Add empty spacer if no data, maintain unit count
                        empty_block = np.full((num_astrocyte_units, max_len_astrocyte), 0.0) # Use 0 for heatmap
                        astrocyte_raster_rows.append(empty_block)
                        y_tick_positions.append(current_y_pos + num_astrocyte_units / 2)
                        y_tick_labels.append(type_labels_panel_i[t] + " (no data)")
                        current_y_pos += num_astrocyte_units

                    if t_idx < len(types) - 1: # Add gap if not the last type
                         astrocyte_raster_rows.append(np.full((gap, max_len_astrocyte), np.nan)) # Gap as NaN
                         current_y_pos += gap


                I_top_astrocyte = np.vstack(astrocyte_raster_rows)

                fig_i, ax_i = plt.subplots(2, 1, figsize=(6, 7), sharex=True,
                                           gridspec_kw={'height_ratios': [3, 2]}) # Adjust ratio for raster vs avg
                fig_i.suptitle(f"Astrocyte-like Responses{plot_title_suffix_astrocyte}", fontsize=14)

                # --- Panel I (Top): Raster Plot ---
                norm = TwoSlopeNorm(vmin=v_min_heatmap, vcenter=0, vmax=v_max_heatmap)
                im_i_top = ax_i[0].imshow(I_top_astrocyte, aspect='auto', cmap='coolwarm', norm=norm,
                                          extent=[time_common_astrocyte[0], time_common_astrocyte[-1] if max_len_astrocyte > 0 else 0,
                                                  I_top_astrocyte.shape[0], 0])
                # Colorbar mimicking Fig 1I
                cbar_i = fig_i.colorbar(im_i_top, ax=ax_i[0], label="dF/F (%) equivalent", fraction=0.046, pad=0.04)
                cbar_i.set_ticks(np.linspace(v_min_heatmap, v_max_heatmap, 6))


                ax_i[0].set_yticks(y_tick_positions)
                ax_i[0].set_yticklabels(y_tick_labels, rotation=0, ha="right", va="center")
                ax_i[0].set_ylabel("Cells (by trial type)")
                ax_i[0].set_title("Average Astrocyte-like Unit Activity (Raster)")
                ax_i[0].axvline(0, color='dimgray', linestyle='--', linewidth=1.5, alpha=0.7) # Tone onset line

                # --- Panel I (Bottom): Average Traces ---
                for t in types: # Plot in specified color order
                    if resps_astrocyte_like[t]:
                        big_astro = stack_with_padding(resps_astrocyte_like[t])
                        if big_astro.size == 0 or big_astro.shape[2] == 0: continue
                        # Average over units first, then over trials for SEM equivalent
                        trial_traces_avg_units = np.nanmean(big_astro, axis=2) # (trials, time_t)
                        pop_mean_astro = np.nanmean(trial_traces_avg_units, axis=0) # (time_t,)
                        # SEM: std(trial_traces_avg_units) / sqrt(num_trials)
                        pop_sem_astro = np.nanstd(trial_traces_avg_units, axis=0) / np.sqrt(trial_traces_avg_units.shape[0]) if trial_traces_avg_units.shape[0] > 0 else 0

                        time_current_type_astro = np.arange(pop_mean_astro.shape[0]) * time_step_sec
                        ax_i[1].plot(time_current_type_astro, pop_mean_astro, color=colors[t], label=type_labels_panel_i[t], linewidth=2)
                        ax_i[1].fill_between(time_current_type_astro, pop_mean_astro - pop_sem_astro, pop_mean_astro + pop_sem_astro,
                                             color=colors[t], alpha=0.2)

                ax_i[1].set_ylabel("dF/F (%) equivalent")
                ax_i[1].set_xlabel("Time from tone onset (s)")
                ax_i[1].axvline(0, color='dimgray', linestyle='--', linewidth=1.5, alpha=0.7)
                ax_i[1].axhline(0, color='gray', linestyle='-', linewidth=0.5)
                ax_i[1].set_title("Population Average Astrocyte-like Activity")
                ax_i[1].legend(loc='upper right', fontsize='small')
                ax_i[1].set_ylim(min(v_min_heatmap, np.nanmin(pop_mean_astro - pop_sem_astro) if 'pop_mean_astro' in locals() and pop_mean_astro.size > 0 else v_min_heatmap),
                                 max(40, np.nanmax(pop_mean_astro + pop_sem_astro) if 'pop_mean_astro' in locals() and pop_mean_astro.size > 0 else 40)) # Match y-axis of Fig1I bottom


                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()


    # --- Panel J: Neuronal Responses & Comparison with Astrocytes on FA ---
    if not resps_neuron_like or not resps_neuron_like.get('FA') or \
       not resps_astrocyte_like or not resps_astrocyte_like.get('FA'):
        print("Missing Neuron-like or Astrocyte-like FA responses for Panel J.")
    elif resps_neuron_like['FA'][0].shape[1] == 0 or resps_astrocyte_like['FA'][0].shape[1] == 0:
        print("Neuron-like or Astrocyte-like FA responses have 0 units. Skipping Panel J.")
    else:
        fig_j, ax_j = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 2]}) # Adjust ratio
        fig_j.suptitle(f"Neuronal-like vs Astrocyte-like (FA Trials){plot_title_suffix_neuron}", fontsize=14)

        # --- Panel J (Top): Neuronal FA Raster ---
        if resps_neuron_like['FA']:
            big_neuron_fa = stack_with_padding(resps_neuron_like['FA'])
            if big_neuron_fa.size > 0 and big_neuron_fa.shape[2] > 0:
                mean_neuron_fa_raster = np.nanmean(big_neuron_fa, axis=0).T # (units, time_fa)
                max_len_neuron_fa = mean_neuron_fa_raster.shape[1]
                time_neuron_fa = np.arange(max_len_neuron_fa) * time_step_sec

                norm_j_top = TwoSlopeNorm(vmin=v_min_heatmap, vcenter=0, vmax=v_max_heatmap)
                im_j_top = ax_j[0].imshow(mean_neuron_fa_raster, aspect='auto', cmap='coolwarm', norm=norm_j_top,
                                          extent=[time_neuron_fa[0], time_neuron_fa[-1] if max_len_neuron_fa > 0 else 0,
                                                  mean_neuron_fa_raster.shape[0], 0])
                cbar_j = fig_j.colorbar(im_j_top, ax=ax_j[0], label="dF/F (%)", fraction=0.046, pad=0.04)
                cbar_j.set_ticks(np.linspace(v_min_heatmap, v_max_heatmap, 6))

                ax_j[0].set_ylabel("Cells (Neurons)")
                ax_j[0].set_title("Average Neuronal-like Unit Activity (FA Raster)")
                ax_j[0].axvline(0, color='dimgray', linestyle='--', linewidth=1.5, alpha=0.7)
                ax_j[0].set_ylim(0, min(40, mean_neuron_fa_raster.shape[0])) # Match original approx y-axis for neurons

        # --- Panel J (Bottom): Population Average Comparison (FA only) ---
        astro_pop_mean_fa, astro_pop_sem_fa, time_astro_fa = None, None, None
        if resps_astrocyte_like['FA']:
            big_astro_fa = stack_with_padding(resps_astrocyte_like['FA'])
            if big_astro_fa.size > 0 and big_astro_fa.shape[2] > 0:
                trial_traces_avg_units_astro_fa = np.nanmean(big_astro_fa, axis=2)
                astro_pop_mean_fa = np.nanmean(trial_traces_avg_units_astro_fa, axis=0)
                astro_pop_sem_fa = np.nanstd(trial_traces_avg_units_astro_fa, axis=0) / np.sqrt(trial_traces_avg_units_astro_fa.shape[0]) if trial_traces_avg_units_astro_fa.shape[0] > 0 else 0
                time_astro_fa = np.arange(astro_pop_mean_fa.shape[0]) * time_step_sec
                ax_j[1].plot(time_astro_fa, astro_pop_mean_fa, color='darkviolet', label='Astrocytes (FA)', linewidth=2.5)
                ax_j[1].fill_between(time_astro_fa, astro_pop_mean_fa - astro_pop_sem_fa, astro_pop_mean_fa + astro_pop_sem_fa,
                                     color='darkviolet', alpha=0.25)
                # Baseline for astrocytes from Fig 1J
                baseline_astro_mean = np.nanmean(astro_pop_mean_fa[time_astro_fa < 0]) if np.any(time_astro_fa < 0) else 0
                baseline_astro_std = np.nanstd(astro_pop_mean_fa[time_astro_fa < 0]) if np.any(time_astro_fa < 0) else 0
                ax_j[1].axhline(baseline_astro_mean, color='darkviolet', linestyle='-', linewidth=0.7, alpha=0.5)
                ax_j[1].fill_between(time_astro_fa, baseline_astro_mean - baseline_astro_std, baseline_astro_mean + baseline_astro_std,
                                     color='darkgray', alpha=0.3, step='mid', linewidth=0)


        neuron_pop_mean_fa, neuron_pop_sem_fa, time_neuron_fa_avg = None, None, None
        if resps_neuron_like['FA']:
            big_neuron_fa_pop = stack_with_padding(resps_neuron_like['FA'])
            if big_neuron_fa_pop.size > 0 and big_neuron_fa_pop.shape[2] > 0:
                trial_traces_avg_units_neuron_fa = np.nanmean(big_neuron_fa_pop, axis=2)
                neuron_pop_mean_fa = np.nanmean(trial_traces_avg_units_neuron_fa, axis=0)
                neuron_pop_sem_fa = np.nanstd(trial_traces_avg_units_neuron_fa, axis=0) / np.sqrt(trial_traces_avg_units_neuron_fa.shape[0]) if trial_traces_avg_units_neuron_fa.shape[0] > 0 else 0
                time_neuron_fa_avg = np.arange(neuron_pop_mean_fa.shape[0]) * time_step_sec
                ax_j[1].plot(time_neuron_fa_avg, neuron_pop_mean_fa, color='orangered', label='Neurons (FA)', linewidth=2.5)
                ax_j[1].fill_between(time_neuron_fa_avg, neuron_pop_mean_fa - neuron_pop_sem_fa, neuron_pop_mean_fa + neuron_pop_sem_fa,
                                     color='orangered', alpha=0.25)
                # Baseline for neurons from Fig 1J
                baseline_neuron_mean = np.nanmean(neuron_pop_mean_fa[time_neuron_fa_avg < 0]) if np.any(time_neuron_fa_avg < 0) else 0
                baseline_neuron_std = np.nanstd(neuron_pop_mean_fa[time_neuron_fa_avg < 0]) if np.any(time_neuron_fa_avg < 0) else 0
                ax_j[1].axhline(baseline_neuron_mean, color='orangered', linestyle='-', linewidth=0.7, alpha=0.5)
                ax_j[1].fill_between(time_neuron_fa_avg, baseline_neuron_mean - baseline_neuron_std, baseline_neuron_mean + baseline_neuron_std,
                                      color='darkgray', alpha=0.3, step='mid', linewidth=0) # Shared gray zone


        ax_j[1].set_ylabel("dF/F (%) equivalent")
        ax_j[1].set_xlabel("Time from tone onset (s)")
        ax_j[1].axvline(0, color='dimgray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax_j[1].axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax_j[1].set_title("Population Average (FA Trials)")
        ax_j[1].legend(loc='upper right', fontsize='small')
        ax_j[1].set_ylim(min(v_min_heatmap, -10), max(40, 1.05 * max(np.nanmax(astro_pop_mean_fa) if astro_pop_mean_fa is not None and astro_pop_mean_fa.size>0 else -np.inf,
                                                                       np.nanmax(neuron_pop_mean_fa) if neuron_pop_mean_fa is not None and neuron_pop_mean_fa.size>0 else -np.inf) ) ) # Match Fig1J bottom y-axis


        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


    # --- Panel M: Fraction of responsive cells ---
    # This requires a definition of "responsive" which is not explicitly detailed
    # in the PPO code (e.g., based on peak activity exceeding a threshold
    # relative to baseline). For now, we'll create a placeholder plot.
    # A proper implementation would involve:
    # 1. Defining baseline activity for each unit (e.g., pre-tone onset).
    # 2. Defining a responsiveness criterion (e.g., peak dF/F > mean_baseline + X * std_baseline).
    # 3. Calculating the fraction of units meeting this criterion for each trial type.

    fig_m, ax_m = plt.subplots(1, 1, figsize=(5, 4))
    fig_m.suptitle("Fraction of Responsive Cells (Placeholder)", fontsize=14)
    bar_width = 0.35
    index = np.arange(len(types))

    # Placeholder data (replace with actual calculations)
    responsive_astro_placeholder = {t: np.random.rand() * 0.8 + 0.1 for t in types} # e.g. FA high, others low
    responsive_neuron_placeholder = {t: np.random.rand() * 0.8 + 0.1 for t in types} # e.g. all somewhat responsive

    astro_fractions = [responsive_astro_placeholder[t] for t in types]
    neuron_fractions = [responsive_neuron_placeholder[t] for t in types]

    rects1 = ax_m.bar(index - bar_width/2, astro_fractions, bar_width, label='Astrocytes', color=[colors[t] for t in types])
    rects2 = ax_m.bar(index + bar_width/2, neuron_fractions, bar_width, label='Neurons', color=[colors[t] for t in types], alpha=0.7) # Neuron bars distinct

    ax_m.set_ylabel('% Responsive')
    ax_m.set_title('Fraction of Responsive Cells by Trial Type')
    ax_m.set_xticks(index)
    ax_m.set_xticklabels([type_labels_panel_i[t] for t in types])
    ax_m.legend(title="Cell Type", loc="upper left", fontsize='small')
    ax_m.set_ylim(0, 1.05) # Percentages
    ax_m.grid(axis='y', linestyle='--', alpha=0.7)

    # Create a shared legend for colors of trial types for clarity if needed, or use distinct bars.
    # For now, the bar colors themselves indicate trial type. The main legend distinguishes Astrocyte/Neuron.

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def make_IJ_plots(resps, time_step_sec):
    types  = ['FA','CR','miss','hit']
    colors = {'FA':'r','CR':'k','miss':'m','hit':'c'}

    # --------------------- PANEL-I --------------------------------------------
    gap = 5
    raster_rows = []
    widths = []

    # pass-1: collect the per-type means, remember their widths
    per_type_means = {}
    for t in types:
        big  = stack_with_padding(resps['layer2'][t])     # (batch,T_t,hidden)
        mean = np.nanmean(big, axis=0).T                 # (hidden,T_t)
        per_type_means[t] = mean
        widths.append(mean.shape[1])

    global_T = max(widths)                # ← one width for everybody
    time     = np.arange(global_T) * time_step_sec

    # pass-2: build raster_rows all padded to global_T
    for t in types:
        mat = right_pad(per_type_means[t], global_T)     # (hidden,global_T)
        raster_rows.append(mat)
        raster_rows.append(np.zeros((gap, global_T)))    # visual spacer

    I_top = np.vstack(raster_rows)                       # now shapes match!
    # -------------------------------------------------------------------------

    # ---------- plotting (unchanged except time uses global_T) ---------------
    fig, ax = plt.subplots(2,1, figsize=(6,8), sharex=True)

    ax[0].imshow(I_top, aspect='auto', cmap='RdBu_r',
                 vmin=-2, vmax=2,
                 extent=[time[0], time[-1], I_top.shape[0], 0])
    ax[0].axvline(0, ls='--', c='k');  ax[0].set_ylabel('Layer-2 units')

    # population averages for each trial-type
    for t in types:
        big = stack_with_padding(resps['layer2'][t])           # (batch,T_t,hidden)
        # big_p = right_pad(np.nanmean(big, axis=(0, 1)), global_T)  # (global_T,)
        big_p = np.nanmean(big, axis=(0, 1)) 
        sd    = np.nanstd(big, axis=(0, 1))
        
        T_cur = big_p.shape[0]                 # this is 64 in your trace
        time  = np.arange(T_cur) * 0.1    # step_sec = 0.1 → same 64-element vector

        ax[1].plot(time, big_p, c=colors[t], label=t)
        ax[1].fill_between(time,
                        big_p - sd,
                        big_p + sd,
                        color=colors[t], alpha=0.20)

    ax[1].axvline(0, ls='--', c='k')
    ax[1].set_ylabel('Mean act.'); ax[1].legend()

    # -------- PANEL-J layer-1 FA raster & overlay --------------------------------
    big_FA_1 = stack_with_padding(resps['layer1']['FA'])
    mean_FA_1 = np.nanmean(big_FA_1, axis=0).T
    fig2, (ax0,ax1) = plt.subplots(2,1, figsize=(6,6), sharex=True)
    ax0.imshow(mean_FA_1, aspect='auto', cmap='RdBu_r',
               vmin=-2,vmax=2, extent=[time[0],time[-1], mean_FA_1.shape[0],0])
    ax0.axvline(0, ls='--', c='k'); ax0.set_ylabel('Layer-1 units')

    for L,color,label in [(1,'r','Layer-1'), (2,'b','Layer-2')]:
        big = stack_with_padding(resps[f'layer{L}']['FA'])       # (batch,Tmax,hidden)
        pop = np.nanmean(big, axis=(0,2))                       # (T_cur,)
        sd  = np.nanstd(big.reshape(-1, big.shape[1]), axis=0)  # (T_cur,)

        # --- make a time vector that matches THIS trace ---
        t_local = np.arange(pop.shape[0]) * time_step_sec       # <= length-match!

        ax1.plot(t_local, pop,  c=color, label=label)
        ax1.fill_between(t_local, pop-sd, pop+sd, color=color, alpha=0.25)
    ax1.axvline(0, ls='--', c='k'); ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Mean act.'); ax1.legend()
    plt.tight_layout(); plt.show()

# ------------------------------------------------
# 6. Main
# ------------------------------------------------
def main_0():

    for i in range(10):
        print("Model #", i)
        num_episodes = 1100
        model, success_history = train_ppo(
            env_class=MouseBehaviorEnvTemporalReward,
            freq_noise=0.2,
            go_prob=0.5,
            num_episodes=num_episodes,
            hidden_size=64,
            # num_layers=2,
            lr=1e-4,
            gamma=0.995,
            lam=0.95,
            clip_ratio=0.2,
            train_iters=8,
            batch_size=256,
            value_loss_coeff=0.5,
            entropy_coeff=0.1,
            eps=0.0,
            step_ms=100
        )

        # Save the trained model
        torch.save(model.state_dict(), f"ppo_custom_gru_noise0.2.{i}.pth")  


        # Plot
        smoothed = moving_average(success_history, window_size=50)
        plt.figure(figsize=(7,4))
        plt.plot(success_history, alpha=0.3, label='Raw Success')
        plt.plot(range(50, len(smoothed)+50), smoothed, color='red', label='Smoothed')
        plt.xlabel("Trials")
        plt.ylabel("Success")
        plt.title("PPO with Custom GRU on Temporal Reward Env")
        plt.legend()
        plt.grid(True)
        plt.show()

############################################################
#  2) Record Hidden Activations in a Real Task Episode
############################################################
def record_hidden_activations_during_task(model, env, num_trials, max_steps=200):
    """
    Runs one episode in 'env' using the model's policy,
    logs the hidden states at each time step for layer1 & layer2.
    
    Returns:
      layer1_acts: shape (T, hidden_size)
      layer2_acts: shape (T, hidden_size)
      rewards:     shape (T,)
    """
    device = model.device
    h = model.initial_hidden(batch_size=1)  # shape(2,1,hidden_size)
    obs_np = env.reset()

    layer1_acts = []
    layer2_acts = []
    rewards = []

    for i in range(num_trials):
        obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).unsqueeze(0).to(device)

        # layer1_acts = []
        # layer2_acts = []
        # rewards = []

        for step in range(max_steps):
            with torch.no_grad():
                logits, value, h_new = model.forward(obs_t, h)
            # sample action
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            # record hidden states
            # shape(2,1,hidden_size): h[0]->layer1, h[1]->layer2
            l1_vec = h[0,0,:].cpu().numpy().copy()
            l2_vec = h[1,0,:].cpu().numpy().copy()
            layer1_acts.append(l1_vec)
            layer2_acts.append(l2_vec)

            # step env
            obs_next, reward, done, _info = env.step(action.item())
            rewards.append(reward)

            h = h_new.detach()
            if not done:
                obs_t = torch.from_numpy(obs_next).float().unsqueeze(0).unsqueeze(0).to(device)
            else:
                break
        
        obs_np = env.reset()  # reset the environment for the next trial
        
    layer1_acts = np.array(layer1_acts)  # shape (T, hidden_size)
    layer2_acts = np.array(layer2_acts)
    rewards = np.array(rewards, dtype=np.float32)
    return layer1_acts, layer2_acts, rewards


############################################################
#  3) Example: Compare "Slow" vs. "Fast" Units in a Real Episode
############################################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Suppose we have a trained model or we train it
    # model, success_history = train_ppo(...)
    # or we load:
    model = CustomActorCriticGRU2Layer(input_size=4, hidden_size=64, action_size=2, device=device)
    model.load_state_dict(torch.load("ppo_custom_gru_noise0.2.1.pth"))

    # ---- run everything -----------------------------------------
    run_and_plot(model, MouseBehaviorEnvTemporalReward,
                 n_trials = 1000,      # more trials → smoother averages
                 step_ms  = 100,
                 frac_max = 0.75,     # what fraction of each unit’s peak
                 min_frac = 0.70)     # ≥ this ⇒ “slow”

    # # 2) collect responses
    # resps = collect_responses(model, MouseBehaviorEnvTemporalReward, n_trials=200, step_ms=100)

    # # 3) build a time‐axis in seconds; here we assume fixed step_ms and constant total steps
    # total_steps = resps['layer2']['FA'][0].shape[0]
    # time = np.arange(total_steps) * (100/1000.)  # 100 ms steps → seconds

    # # 4) make the I & J plots
    # step_sec = 0.1
    # make_IJ_plots(resps, step_sec)

    # # 1) Identify slow vs. fast units from the ping test
    # timescales, responses = ping_hidden_units(model, num_steps=100, impulse_value=1.0, threshold_ratio=0.37)
    # # Let's define "slow" as top 5 timescales, "fast" as bottom 5
    # idx_sorted = np.argsort(timescales)  # ascending
    # fast_units = idx_sorted[:5]
    # slow_units = idx_sorted[-5:]

    # # 2) Run 1 test episode, record hidden states
    # env = MouseBehaviorEnvTemporalReward(go_tone=True, step_ms=100, noise_std=0.2)
    # layer1_acts, layer2_acts, rewards = record_hidden_activations_during_task(model, env, num_trials=2)

    # T = layer2_acts.shape[0]  # actual steps in the episode

    # # 3) Plot slow vs. fast units from layer2_acts
    # plt.figure(figsize=(10,6))
    # times = np.arange(T)
    # for i in slow_units:
    #     plt.plot(times, layer2_acts[:, i], label=f"Slow unit {i} (ts={timescales[i]:.1f})")
    # plt.xlabel("Time step (GRU steps)")
    # plt.ylabel("Activation (Layer 2)")
    # plt.title("Slow Units' Activations During Task")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # 4) Plot fast units
    # plt.figure(figsize=(10,6))
    # for i in fast_units:
    #     plt.plot(times, layer2_acts[:, i], label=f"Fast unit {i} (ts={timescales[i]:.1f})")
    # plt.xlabel("Time step (GRU steps)")
    # plt.ylabel("Activation (Layer 2)")
    # plt.title("Fast Units' Activations During Task")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # We can also show the reward timeline
    # plt.figure(figsize=(8,3))
    # plt.plot(times, rewards, marker='o', label='Reward')
    # plt.xlabel("Time step")
    # plt.ylabel("Reward")
    # plt.title("Reward Timeline During Trials")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    main()
