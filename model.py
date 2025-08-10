import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class vRNNLayer(pl.LightningModule):
    """
    Vanilla RNN layer in continuous time.
    Provided by user.
    """

    def __init__(self, input_size, hidden_size, output_size, alpha, g, nonlinearity):
        super(vRNNLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha
        self.inv_sqrt_alpha = 1 / np.sqrt(alpha)
        self.g = g
        self.process_noise = 0.05

        # set nonlinearity
        if nonlinearity == "tanh":
            self.phi = torch.tanh
        elif nonlinearity == "relu":
            self.phi = F.relu
        else:
            self.phi = torch.nn.Identity()

        # input-to-hidden
        self.weight_ih = nn.Parameter(
            torch.normal(0, 1/np.sqrt(hidden_size), (hidden_size, input_size))
        )
        # hidden-to-output
        self.weight_ho = nn.Parameter(
            torch.normal(0, 1/np.sqrt(hidden_size), (output_size, hidden_size))
        )
        # hidden-to-hidden
        self.W = nn.Parameter(
            torch.normal(0, g/np.sqrt(hidden_size), (hidden_size, hidden_size))
        )
        # bias output
        self.bias_oh = nn.Parameter(
            torch.normal(0, 1/np.sqrt(hidden_size), (1, output_size))
        )
        # bias hidden
        self.bias_hh = nn.Parameter(
            torch.normal(0, 1/np.sqrt(hidden_size), (1, hidden_size))
        )

        # define mask for structural perturbation experiments
        self.struc_p_0 = 0
        self.register_buffer(
            "struc_perturb_mask",
            torch.FloatTensor(self.hidden_size, self.hidden_size).uniform_() > self.struc_p_0,
        )

    def forward(self, input_seq):
        """
        input_seq: (batch_size, time_steps, input_size)
        Returns (outputs, states, noise, None)
        - outputs: (batch_size, time_steps, output_size)
        - states:  (batch_size, time_steps, hidden_size)
        """
        batch_size, time_steps, _ = input_seq.shape

        # init hidden state
        state = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # define process noise
        # shape: (batch_size, time_steps, hidden_size)
        noise = (
            1.41 * self.process_noise *
            torch.randn(batch_size, time_steps, self.hidden_size, device=self.device)
        )

        outputs = []
        states = []

        for t in range(time_steps):
            # compute output from current state
            y_out = state @ self.weight_ho.T + self.bias_oh  # (batch_size, output_size)

            outputs.append(y_out)
            states.append(state)

            # compute RNN update
            fx = -state + self.phi(
                (state @ (self.W * self.struc_perturb_mask)) +
                (input_seq[:,t,:] @ self.weight_ih.T) +
                self.bias_hh +
                self.inv_sqrt_alpha * noise[:,t,:]
            )
            # Euler step
            state = state + self.alpha * fx

        # stack them up
        outputs = torch.stack(outputs, dim=1)  # (batch, time, output_size)
        states = torch.stack(states, dim=1)    # (batch, time, hidden_size)
        return outputs, states, noise, None
    
class VanillaRNNLightning(pl.LightningModule):
    def __init__(self, input_size=3, hidden_size=32, output_size=2, alpha=0.2, g=1.0,
                 nonlinearity="tanh", lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = vRNNLayer(input_size, hidden_size, output_size, alpha, g, nonlinearity)
        self.lr = lr

    def forward(self, x):
        """
        x: (batch, time, input_size)
        returns: (batch, time, output_size)
        """
        outputs, states, noise, _ = self.rnn(x)
        return outputs  # interpret as action logits

    def training_step(self, batch, batch_idx):
        """
        batch => (input_seq, outcome_seq, action_label_seq)
        """
        input_seq, outcome_seq, action_seq = batch  # ignoring outcome_seq for supervised approach
        # input_seq: (batch, time, 3)
        # action_seq: (batch, time) with 0 or 1
        # pass input through RNN
        logits = self.forward(input_seq)  # (batch, time, 2)

        # Flatten for cross entropy
        # We need (batch*time, 2) vs. (batch*time) for action_seq
        batch_size, time_steps, _ = logits.shape
        logits_2d = logits.reshape(batch_size*time_steps, -1)
        target_1d = action_seq.view(-1)

        loss = F.cross_entropy(logits_2d, target_1d)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_seq, outcome_seq, action_seq = batch
        logits = self.forward(input_seq)
        batch_size, time_steps, _ = logits.shape
        logits_2d = logits.reshape(batch_size*time_steps, -1)
        target_1d = action_seq.view(-1)
        loss = F.cross_entropy(logits_2d, target_1d)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_seq, outcome_seq, action_seq = batch
        logits = self.forward(input_seq)
        batch_size, time_steps, _ = logits.shape
        logits_2d = logits.reshape(batch_size*time_steps, -1)
        target_1d = action_seq.view(-1)
        loss = F.cross_entropy(logits_2d, target_1d)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)