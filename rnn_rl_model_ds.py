import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from drummond_env import vRNNLayer

# Assuming the provided vRNNLayer and helper functions are already defined above

class DrummondRNN(pl.LightningModule):
    def __init__(self, input_size=3, hidden_size=128, output_size=2, alpha=0.1, g=1.0, lr=1e-3, gamma=0.99, entropy_coef=0.01):
        super().__init__()
        self.save_hyperparameters()
        
        self.rnn = vRNNLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            alpha=alpha,
            g=g,
            nonlinearity="tanh"
        )
        self.gamma = gamma
        self.entropy_coef = entropy_coef

    def forward(self, x):
        outputs, _, _, _ = self.rnn(x)
        action_logits = outputs[..., 0]
        value_estimates = outputs[..., 1]
        return action_logits, value_estimates

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def _process_trial(self, go_trial, intensity_dB):
        # Generate trial data
        trial_data = build_drummond_trial(
            go_tone=go_trial,
            intensity_dB=intensity_dB,
            step_ms=100,
            rng_seed=None
        )
        obs = trial_data
        phases = self._get_phases(obs)
        
        # Convert to tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Forward pass
        action_logits, value_estimates = self(obs_tensor)
        action_probs = torch.sigmoid(action_logits.squeeze())
        actions = torch.bernoulli(action_probs).cpu().numpy()
        
        # Determine reward
        reward, press_step = self._compute_reward(go_trial, actions, phases)
        return action_logits.squeeze(), value_estimates.squeeze(), reward, press_step

    def _get_phases(self, obs):
        # Determine phase indices from observation
        led_on = obs[:, 0] == 1
        cue_indices = np.where(led_on)[0]
        cue_start, cue_end = (0, 0) if len(cue_indices) == 0 else (cue_indices[0], cue_indices[-1]+1)
        
        sound_indices = np.where(obs[:, 1] != 0)[0]
        sound_start, sound_end = (cue_end, cue_end) if len(sound_indices) == 0 else (sound_indices[0], sound_indices[-1]+1)
        
        delay_start, delay_end = cue_end, sound_start
        response_start, response_end = sound_end, sound_end + 8  # 800ms / 100ms step
        iti_start, iti_end = response_end, len(obs)
        
        return {
            'cue': (cue_start, cue_end),
            'delay': (delay_start, delay_end),
            'sound': (sound_start, sound_end),
            'response': (response_start, response_end),
            'iti': (iti_start, iti_end)
        }

    def _compute_reward(self, go_trial, actions, phases):
        press_steps = np.where(actions == 1)[0]
        if len(press_steps) == 0:
            return 0.0, None
        
        first_press = press_steps[0]
        ds, de = phases['delay']
        rs, re = phases['response']
        
        if ds <= first_press < de:
            return -1.0, first_press  # Premature press
        elif rs <= first_press < re:
            if go_trial:
                return 1.0, first_press  # Correct press (hit)
            else:
                return -1.0, first_press  # False alarm
        else:
            return 0.0, first_press  # Press during ITI or other phases

    def _calculate_losses(self, action_logits, value_estimates, reward, press_step):
        T = action_logits.shape[0]
        rewards = torch.zeros(T, device=self.device)
        
        # Discounted reward calculation
        if press_step is not None and reward != 0:
            discount = self.gamma ** (torch.arange(T - press_step, device=self.device).float())
            rewards[press_step:] = reward * discount[:T - press_step]
        cumulative_reward = rewards.sum()
        
        # Policy loss
        log_probs = F.binary_cross_entropy_with_logits(
            action_logits,
            torch.tensor((rewards != 0).float(), device=self.device),
            reduction='none'
        )
        advantages = cumulative_reward - value_estimates.detach()
        policy_loss = (-log_probs * advantages).mean()
        
        # Entropy regularization
        probs = torch.sigmoid(action_logits)
        entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
        entropy_loss = -entropy.mean() * self.entropy_coef
        
        # Value loss
        value_loss = F.mse_loss(value_estimates, cumulative_reward.expand_as(value_estimates))
        
        total_loss = policy_loss + value_loss + entropy_loss
        return total_loss, policy_loss, value_loss, entropy_loss

    def training_step(self, batch, batch_idx):
        # Generate a batch of trials
        batch_size = 32  # Adjust based on computational resources
        total_loss = 0
        for _ in range(batch_size):
            go_trial = np.random.rand() > 0.5
            intensity = np.random.choice([5, 15, 25, 35])
            action_logits, value_estimates, reward, press_step = self._process_trial(go_trial, intensity)
            loss, ploss, vloss, eloss = self._calculate_losses(action_logits, value_estimates, reward, press_step)
            total_loss += loss
        
        total_loss /= batch_size
        self.log('train_loss', total_loss)
        self.log('policy_loss', ploss)
        self.log('value_loss', vloss)
        self.log('entropy_loss', eloss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        # Similar to training but without gradient updates
        go_trial = np.random.rand() > 0.5
        intensity = np.random.choice([5, 15, 25, 35])
        action_logits, value_estimates, reward, press_step = self._process_trial(go_trial, intensity)
        loss, _, _, _ = self._calculate_losses(action_logits, value_estimates, reward, press_step)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        go_trial = np.random.rand() > 0.5
        intensity = np.random.choice([5, 15, 25, 35])
        action_logits, value_estimates, reward, press_step = self._process_trial(go_trial, intensity)
        loss, _, _, _ = self._calculate_losses(action_logits, value_estimates, reward, press_step)
        self.log('test_loss', loss)

class TrialDataset(Dataset):
    def __init__(self, num_trials=1000):
        self.num_trials = num_trials
        
    def __len__(self):
        return self.num_trials
    
    def __getitem__(self, idx):
        go_trial = np.random.rand() > 0.5
        intensity = np.random.choice([5, 15, 25, 35])
        trial_data = build_drummond_trial(go_trial, intensity)
        return trial_data

if __name__ == "__main__":
    model = DrummondRNN(hidden_size=128, alpha=0.1, g=1.0, lr=1e-3)
    trainer = pl.Trainer(max_epochs=100, gpus=1 if torch.cuda.is_available() else 0)
    trainer.fit(model)