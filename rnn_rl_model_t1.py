import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

def db_to_amplitude(dB):
    """
    Convert a decibel value to linear amplitude:
      amplitude = 10^(dB/20).
    E.g. 5 dB ~ 1.78, 15 dB ~ 5.62, 35 dB ~ 56.23, etc.
    """
    return 10.0 ** (dB / 20.0)

def build_drummond_trial(
    go_tone=True,
    intensity_dB=25,
    amplitude_noise_std = 0.25,
    led_cue_ms=1000,
    delay_mean_ms=650,
    delay_std_ms=150,
    sound_ms=500,
    response_ms=800,
    iti_ms=4000,
    step_ms=100,
    rng_seed=None
):
    """
    Builds the exact temporal input for one trial of the Drummond task.
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    raw_delay = np.random.normal(delay_mean_ms, delay_std_ms)
    random_delay_ms = max(0, raw_delay)
    led_steps  = led_cue_ms // step_ms
    delay_steps= int(random_delay_ms // step_ms)
    sound_steps= sound_ms // step_ms
    response_steps = response_ms // step_ms
    iti_steps  = iti_ms // step_ms
    total_steps = led_steps + delay_steps + sound_steps + response_steps + iti_steps

    freq_sign = 1.0 if go_tone else -1.0
    base_amp  = db_to_amplitude(intensity_dB)

    obs = np.zeros((total_steps, 3), dtype=float)

    idx_start = 0
    # LED period
    idx_end = idx_start + led_steps
    if idx_end > idx_start:
        obs[idx_start:idx_end, 0] = 1.0
    idx_start = idx_end

    # Random delay
    idx_end = idx_start + delay_steps
    idx_start = idx_end

    # Sound
    idx_end = idx_start + sound_steps
    if idx_end > idx_start:
        obs[idx_start:idx_end, 1] = freq_sign
        obs[idx_start:idx_end, 2] = intensity_dB
    idx_start = idx_end

    # Response window
    idx_end = idx_start + response_steps
    idx_start = idx_end

    # ITI
    idx_end = idx_start + iti_steps
    idx_start = idx_end

    return obs


def build_outcome_time_series(
    total_steps,
    step_ms=100,
    press_step=None,
    correct_press=None,
    reward_delay_ms=250,
    punishment_duration_ms=300,
):
    """
    Build a time-series array (T,1) for reward/punishment signals.
    """

    outcome = np.zeros((total_steps * step_ms, 1), dtype=float)

    if press_step is None or correct_press is None:
        return outcome

    outcome_start_delay_steps = reward_delay_ms
    start_idx = outcome_start_delay_steps

    if correct_press:
        outcome[start_idx:start_idx+50, 0] = 1.0  # Reward
    else:
        punish_steps = punishment_duration_ms
        end_idx = min(start_idx + punish_steps, total_steps * step_ms)
        outcome[start_idx:end_idx, 0] = -1.0  # Punishment

    return outcome


class vRNNLayer(pl.LightningModule):
    """Vanilla RNN layer in continuous time."""

    def __init__(self, input_size, hidden_size, output_size, alpha, g, nonlinearity):
        super(vRNNLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha
        self.inv_sqrt_alpha = 1 / np.sqrt(alpha)
        self.cont_stab = False
        self.disc_stab = True
        self.g = g
        self.process_noise = 0.05

        # set nonlinearity of the vRNN
        self.nonlinearity = nonlinearity
        if nonlinearity == "tanh":
            self.phi = torch.tanh
        if nonlinearity == "relu":
            self.phi = F.relu
        if nonlinearity == "none":
            print("Nl = none")
            self.phi = torch.nn.Identity()

        # initialize the input-to-hidden weights
        self.weight_ih = nn.Parameter(
            torch.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, input_size))
        )

        # initialize the hidden-to-output weights
        self.weight_ho = nn.Parameter(
            torch.normal(0, 1 / np.sqrt(hidden_size), (output_size, hidden_size))
        )

        # initialize the hidden-to-hidden weights
        self.W = nn.Parameter(
            torch.normal(0, self.g / np.sqrt(hidden_size), (hidden_size, hidden_size))
        )

        # initialize the output bias weights
        self.bias_oh = nn.Parameter(
            torch.normal(0, 1 / np.sqrt(hidden_size), (1, output_size))
        )

        # initialize the hidden bias weights
        self.bias_hh = nn.Parameter(
            torch.normal(0, 1 / np.sqrt(hidden_size), (1, hidden_size))
        )

        # define mask for weight matrix do to structural perturbation experiments
        self.struc_p_0 = 0
        self.register_buffer(
            "struc_perturb_mask",
            torch.FloatTensor(self.hidden_size, hidden_size).uniform_()
            > self.struc_p_0,
        )

    def forward(self, input):

        # initialize state at the origin. randn is there just in case we want to play with this later.
        state = 0 * torch.randn(input.shape[0], self.hidden_size, device=self.device)

        # defines process noise using Euler-discretization of stochastic differential equation defining the RNN
        noise = (
            1.41
            * self.process_noise
            * torch.randn(
                input.shape[0], input.shape[1], self.hidden_size, device=self.device
            )
        )

        # for storing RNN outputs and hidden states
        outputs = []
        states = []

        # loop over input
        for i in range(input.shape[1]):

            # compute output
            hy = state @ self.weight_ho.T + self.bias_oh

            # save output and hidden states
            outputs += [hy]
            states += [state]

            # compute the RNN update
            fx = -state + self.phi(
                state @ (self.W * self.struc_perturb_mask)
                + input[:, i, :] @ self.weight_ih.T
                + self.bias_hh
                + self.inv_sqrt_alpha * noise[:, i, :]
            )

            # step hidden state foward using Euler discretization
            state = state + self.alpha * (fx)

        # organize states and outputs and return
        return (
            torch.stack(outputs).permute(1, 0, 2),
            torch.stack(states).permute(1, 0, 2),
            noise,
            None,
        )



class DrummondDataset(Dataset):
    def __init__(self, num_trials, step_ms, rng_seed=None):
        self.num_trials = num_trials
        self.step_ms = step_ms
        self.rng_seed = rng_seed
        if self.rng_seed is not None:
            np.random.seed(self.rng_seed)
        self.data = self._generate_trials()


    def _generate_trials(self):
        trials = []
        for _ in range(self.num_trials):
            go_tone = np.random.choice([True, False])
            intensity_dB = np.random.choice([5, 15, 25, 35])
            trial_input = build_drummond_trial(go_tone=go_tone, intensity_dB=intensity_dB, step_ms=self.step_ms, rng_seed=self.rng_seed)

            # Determine correct action and generate outcome
            total_steps = trial_input.shape[0]
            press_step = None
            correct_press = None

            # Simulate a press with some probability (adjust as needed)
            if np.random.rand() < 0.5:  # 50% chance of pressing
                press_step = np.random.randint(0, total_steps)  # Random press time

                # Determine if the press is correct
                if go_tone:
                    correct_press = True  # Correct press for go tone
                else:
                    correct_press = False # Incorrect press for no-go tone


            trial_outcome = build_outcome_time_series(total_steps, step_ms=self.step_ms, press_step=press_step, correct_press=correct_press)
            # Convert to float32 explicitly
            trials.append((trial_input.astype(np.float32), trial_outcome.astype(np.float32)))

        return trials

    def __len__(self):
        return self.num_trials

    def __getitem__(self, idx):
        return self.data[idx]

class DrummondModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, alpha, g, nonlinearity, lr):
        super().__init__()
        self.rnn = vRNNLayer(input_size, hidden_size, output_size, alpha, g, nonlinearity)
        self.lr = lr
        self.loss_fn = nn.MSELoss()  # Use MSE for continuous output
        self.save_hyperparameters()

    def forward(self, x):
        outputs, _, _, _ = self.rnn(x)
        return outputs

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        # targets = targets.squeeze(-1)
        loss = self.loss_fn(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        # targets = targets.squeeze(-1)
        loss = self.loss_fn(outputs, targets)
        self.log('test_loss', loss)
        return loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# --- Hyperparameters ---
input_size = 3   # LED, Frequency, Amplitude
hidden_size = 50
output_size = 1  # Reward/Punishment
alpha = 0.1
g = 1.5
nonlinearity = 'tanh'
lr = 0.001
batch_size = 32
num_epochs = 10
num_trials_train = 2000
num_trials_val = 500
num_trials_test = 500
step_ms = 100
rng_seed = 42  # For reproducibility


# --- Data Loading ---
train_dataset = DrummondDataset(num_trials_train, step_ms, rng_seed=rng_seed)
val_dataset = DrummondDataset(num_trials_val, step_ms, rng_seed=rng_seed+1)  # Different seed for validation
test_dataset = DrummondDataset(num_trials_test, step_ms, rng_seed=rng_seed+2)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# --- Model Initialization ---
model = DrummondModel(input_size, hidden_size, output_size, alpha, g, nonlinearity, lr)

# --- Training ---
trainer = pl.Trainer(max_epochs=num_epochs, accelerator="cpu")
trainer.fit(model, train_loader, val_loader)

# --- Testing ---
trainer.test(model, test_loader)


# --- Prediction Example (after training) ---
def predict_trial(model, go_tone, intensity_dB, step_ms, rng_seed=None):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        trial_input = build_drummond_trial(go_tone=go_tone, intensity_dB=intensity_dB, step_ms=step_ms, rng_seed=rng_seed)

        # Convert to tensor and add batch dimension
        trial_input_tensor = torch.tensor(trial_input, dtype=torch.float32).unsqueeze(0)
        predicted_outcome = model(trial_input_tensor)

        # Remove batch dimension and convert to numpy
        predicted_outcome = predicted_outcome.squeeze(0).numpy()
        return predicted_outcome
    

# Example usage of prediction (make sure this is *after* training):
predicted_outcome = predict_trial(model, go_tone=True, intensity_dB=25, step_ms=step_ms)


def plot_combined(trial_obs, outcome, predicted_outcome, step_ms=100, title="Trial and Outcomes"):
    """Plots input, true outcome, and predicted outcome."""
    T_input = trial_obs.shape[0]
    T_outcome = outcome.shape[0]
    T_predicted = predicted_outcome.shape[0]
    
    # print(T_input, T_outcome, T_predicted)
    # Find the minimum length
    min_T = min(T_input * step_ms, T_outcome, T_predicted * step_ms)
    print(min_T)
    
    # Create time arrays based on step_ms and minimum length
    times_ms_input = np.arange(0, min_T, step_ms)
    times_ms_outcome = np.arange(0, min_T)
    times_ms_predicted = np.arange(0, min_T, step_ms)

    # Truncate arrays to the minimum length
    trial_obs = trial_obs[:len(times_ms_input)]
    outcome = outcome[:min_T]
    predicted_outcome = predicted_outcome[:len(times_ms_predicted)]

    plt.figure(figsize=(10, 6))

    # Input
    plt.subplot(3, 1, 1)
    plt.plot(times_ms_input, trial_obs[:, 0], label="LED")
    plt.plot(times_ms_input, trial_obs[:, 1], label="Freq")
    plt.plot(times_ms_input, trial_obs[:, 2], label="Loudness")
    plt.title("Input")
    plt.xlabel("Time (ms)")
    plt.legend()

    # True Outcome
    plt.subplot(3, 1, 2)
    plt.plot(times_ms_outcome, outcome[:, 0], label="True Outcome", drawstyle='steps-post')
    plt.title("True Outcome")
    plt.xlabel("Time (ms)")
    plt.legend()

    # Predicted Outcome
    plt.subplot(3, 1, 3)
    plt.plot(times_ms_predicted, predicted_outcome[:, 0], label="Predicted Outcome", drawstyle='steps-post')
    plt.title("Predicted Outcome")
    plt.xlabel("Time (ms)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Example trials for visualization
go_trial_input = build_drummond_trial(go_tone=True, intensity_dB=25, step_ms=step_ms, rng_seed=123)
go_trial_outcome = build_outcome_time_series(go_trial_input.shape[0], step_ms, press_step=20, correct_press=True)  # Simulate correct press
go_predicted_outcome = predict_trial(model, go_tone=True, intensity_dB=25, step_ms=step_ms, rng_seed=123)

plot_combined(go_trial_input, go_trial_outcome, go_predicted_outcome, step_ms, "Go Trial, Correct Press, and Prediction")

nogo_trial_input = build_drummond_trial(go_tone=False, intensity_dB=15, step_ms=step_ms, rng_seed=456)
nogo_trial_outcome = build_outcome_time_series(nogo_trial_input.shape[0], step_ms, press_step=None, correct_press=None)  # No press
nogo_predicted_outcome = predict_trial(model, go_tone=False, intensity_dB=15, step_ms=step_ms, rng_seed=456)
plot_combined(nogo_trial_input, nogo_trial_outcome, nogo_predicted_outcome, step_ms, "No-Go Trial, No Press, and Prediction")

nogo_trial_input = build_drummond_trial(go_tone=False, intensity_dB=5, step_ms=step_ms, rng_seed=456)
nogo_trial_outcome = build_outcome_time_series(nogo_trial_input.shape[0], step_ms, press_step=25, correct_press=False)  # incorrect press
nogo_predicted_outcome = predict_trial(model, go_tone=False, intensity_dB=5, step_ms=step_ms, rng_seed=456)
plot_combined(nogo_trial_input, nogo_trial_outcome, nogo_predicted_outcome, step_ms, "No-Go Trial, Incorrect Press, and Prediction")