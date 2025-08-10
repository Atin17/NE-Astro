import torch
import math
import torch.nn as nn

class CustomActorCriticGRU2Layer:
    """
    A two-layer GRU-based actor-critic network implemented "from scratch",
    without nn.Module or built-in GRU/Linear modules.
    
    We store:
      - 1st layer gating parameters (W_ihz1, W_hhz1, etc.)
      - 2nd layer gating parameters (W_ihz2, W_hhz2, etc.)
      - Policy head (W_policy, b_policy)
      - Value head  (W_value,  b_value)
    
    forward(x, h) => (logits, value, h_new)
      where x: shape (1, 1, input_size)
            h: shape (2, 1, hidden_size) for 2 layers

    forward_sequence(x_seq, h) => unroll over seq_len time steps
      where x_seq: shape (1, seq_len, input_size)
    """

    def __init__(self, input_size=4, hidden_size=32, action_size=2, device='cpu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.device = device

        # ------------------------------------------------------------
        # ========== Layer 1 Parameters ========== 
        self.W_ihz1 = nn.Parameter(torch.randn(input_size, hidden_size, requires_grad=True, device=device)*0.1)
        self.W_hhz1 = nn.Parameter(torch.randn(hidden_size, hidden_size, requires_grad=True, device=device)*0.1)
        self.b_z1   = nn.Parameter(torch.zeros(hidden_size, requires_grad=True, device=device))

        self.W_ihr1 = nn.Parameter(torch.randn(input_size, hidden_size, requires_grad=True, device=device)*0.1)
        self.W_hhr1 = nn.Parameter(torch.randn(hidden_size, hidden_size, requires_grad=True, device=device)*0.1)
        self.b_r1   = nn.Parameter(torch.zeros(hidden_size, requires_grad=True, device=device))

        self.W_ihn1 = nn.Parameter(torch.randn(input_size, hidden_size, requires_grad=True, device=device)*0.1)
        self.W_hhn1 = nn.Parameter(torch.randn(hidden_size, hidden_size, requires_grad=True, device=device)*0.1)
        self.b_n1   = nn.Parameter(torch.zeros(hidden_size, requires_grad=True, device=device))

        # ------------------------------------------------------------
        # ========== Layer 2 Parameters ========== 
        self.W_ihz2 = nn.Parameter(torch.randn(hidden_size, hidden_size, requires_grad=True, device=device)*0.1)
        self.W_hhz2 = nn.Parameter(torch.randn(hidden_size, hidden_size, requires_grad=True, device=device)*0.1)
        self.b_z2   = nn.Parameter(torch.zeros(hidden_size, requires_grad=True, device=device))

        self.W_ihr2 = nn.Parameter(torch.randn(hidden_size, hidden_size, requires_grad=True, device=device)*0.1)
        self.W_hhr2 = nn.Parameter(torch.randn(hidden_size, hidden_size, requires_grad=True, device=device)*0.1)
        self.b_r2   = nn.Parameter(torch.zeros(hidden_size, requires_grad=True, device=device))

        self.W_ihn2 = nn.Parameter(torch.randn(hidden_size, hidden_size, requires_grad=True, device=device)*0.1)
        self.W_hhn2 = nn.Parameter(torch.randn(hidden_size, hidden_size, requires_grad=True, device=device)*0.1)
        self.b_n2   = nn.Parameter(torch.zeros(hidden_size, requires_grad=True, device=device))

        # ------------------------------------------------------------
        # Policy head
        self.W_policy =nn.Parameter(torch.randn(hidden_size, action_size, requires_grad=True, device=device)*0.1)
        self.b_policy =nn.Parameter(torch.zeros(action_size, requires_grad=True, device=device))

        # -----------------------------------------------------------
        # Value head
        self.W_value = nn.Parameter(torch.randn(hidden_size, 1, requires_grad=True, device=device)*0.1)
        self.b_value = nn.Parameter(torch.zeros(1, requires_grad=True, device=device))

        # Gather all parameters
        self.parameters = [
            # layer1
            self.W_ihz1, self.W_hhz1, self.b_z1,
            self.W_ihr1, self.W_hhr1, self.b_r1,
            self.W_ihn1, self.W_hhn1, self.b_n1,
            # layer2
            self.W_ihz2, self.W_hhz2, self.b_z2,
            self.W_ihr2, self.W_hhr2, self.b_r2,
            self.W_ihn2, self.W_hhn2, self.b_n2,
            # policy, value
            self.W_policy, self.b_policy,
            self.W_value,  self.b_value
        ]

    # ---- Single-step GRU gating ----
    def _gru_cell(self, x_t, h_tm1,
                  W_ihz, W_hhz, b_z,
                  W_ihr, W_hhr, b_r,
                  W_ihn, W_hhn, b_n):
        """
        A single-layer GRU cell computation for one time step:
          z = sigma(x_t W_ihz + h_tm1 W_hhz + b_z)
          r = sigma(x_t W_ihr + h_tm1 W_hhr + b_r)
          n = tanh(x_t W_ihn + (r * h_tm1) W_hhn + b_n)
          h_t = (1-z)*h_tm1 + z*n
        """
        z_t = torch.sigmoid(x_t @ W_ihz + h_tm1 @ W_hhz + b_z)
        r_t = torch.sigmoid(x_t @ W_ihr + h_tm1 @ W_hhr + b_r)
        n_t = torch.tanh(x_t @ W_ihn + (r_t * h_tm1) @ W_hhn + b_n)
        h_t = (1 - z_t)*h_tm1 + z_t*n_t
        return h_t

    def forward(self, x, h):
        """
        Single time-step forward:
          x: (1, 1, input_size)
          h: (2, 1, hidden_size) => [h_layer1, h_layer2]
        
        Returns:
          logits: (1, action_size)
          value:  (1,)
          h_new:  (2, 1, hidden_size)
        """
        # shape x => (1, input_size)
        x_2d = x.view(1, self.input_size)

        # Extract layer1 hidden, layer2 hidden
        h_layer1 = h[0]  # shape (1, hidden_size)
        h_layer2 = h[1]  # shape (1, hidden_size)

        # ---- Layer 1 ----
        h1_new = self._gru_cell(
            x_2d, h_layer1,
            self.W_ihz1, self.W_hhz1, self.b_z1,
            self.W_ihr1, self.W_hhr1, self.b_r1,
            self.W_ihn1, self.W_hhn1, self.b_n1
        )
        # ---- Layer 2 ----
        # input = h1_new
        h2_new = self._gru_cell(
            h1_new, h_layer2,
            self.W_ihz2, self.W_hhz2, self.b_z2,
            self.W_ihr2, self.W_hhr2, self.b_r2,
            self.W_ihn2, self.W_hhn2, self.b_n2
        )

        # Final hidden => h2_new used for policy & value
        logits = h2_new @ self.W_policy + self.b_policy  # (1, action_size)
        value  = (h2_new @ self.W_value + self.b_value).squeeze(-1)  # (1,)

        # Combine
        h_new = torch.stack([h1_new, h2_new], dim=0)  # shape (2,1,hidden_size)
        return logits, value, h_new

    def forward_sequence(self, x_seq, h):
        """
        Multi-time-step forward pass:
          x_seq: shape (batch=1, seq_len, input_size)
          h    : shape (2, 1, hidden_size)

        We'll unroll over seq_len steps, returning lists (or stacked Tensors)
        of logits and values at each step, plus the final hidden state.
        """
        seq_len = x_seq.size(1)

        logits_out = []
        values_out = []

        # We'll iterate over each time step
        for t in range(seq_len):
            # Extract x_t => shape (1,1,input_size)
            x_t = x_seq[:, t, :].unsqueeze(0)
            # Single-step forward
            logits_t, val_t, h = self.forward(x_t, h)
            logits_out.append(logits_t)  # shape (1, action_size)
            values_out.append(val_t)     # shape (1,)

        # Concatenate along time dimension => shape (seq_len, batch=1, action_size) for logits
        # shape (seq_len, batch=1) for values
        # We'll do a simple cat along dim=0
        logits_out = torch.cat(logits_out, dim=0)   # (seq_len, action_size)
        values_out = torch.cat(values_out, dim=0)   # (seq_len,)

        # Final hidden is `h`
        return logits_out, values_out, h

    def initial_hidden(self, batch_size=1):
        """
        Return a zero hidden state for 2 layers: shape (2, batch_size, hidden_size).
        """
        return torch.zeros(2, batch_size, self.hidden_size, device=self.device)

    # ------------------------------
    # State dict for saving/loading
    # ------------------------------
    def state_dict(self):
        return {
            'W_ihz1': self.W_ihz1,
            'W_hhz1': self.W_hhz1,
            'b_z1':   self.b_z1,
            'W_ihr1': self.W_ihr1,
            'W_hhr1': self.W_hhr1,
            'b_r1':   self.b_r1,
            'W_ihn1': self.W_ihn1,
            'W_hhn1': self.W_hhn1,
            'b_n1':   self.b_n1,
            'W_ihz2': self.W_ihz2,
            'W_hhz2': self.W_hhz2,
            'b_z2':   self.b_z2,
            'W_ihr2': self.W_ihr2,
            'W_hhr2': self.W_hhr2,
            'b_r2':   self.b_r2,
            'W_ihn2': self.W_ihn2,
            'W_hhn2': self.W_hhn2,
            'b_n2':   self.b_n2,
            'W_policy': self.W_policy,
            'b_policy': self.b_policy,
            'W_value': self.W_value,
            'b_value': self.b_value
        }

    def load_state_dict(self, state_dict):
        self.W_ihz1.data.copy_(state_dict['W_ihz1'])
        self.W_hhz1.data.copy_(state_dict['W_hhz1'])
        self.b_z1.data.copy_(state_dict['b_z1'])
        self.W_ihr1.data.copy_(state_dict['W_ihr1'])
        self.W_hhr1.data.copy_(state_dict['W_hhr1'])
        self.b_r1.data.copy_(state_dict['b_r1'])
        self.W_ihn1.data.copy_(state_dict['W_ihn1'])
        self.W_hhn1.data.copy_(state_dict['W_hhn1'])
        self.b_n1.data.copy_(state_dict['b_n1'])
        self.W_ihz2.data.copy_(state_dict['W_ihz2'])
        self.W_hhz2.data.copy_(state_dict['W_hhz2'])
        self.b_z2.data.copy_(state_dict['b_z2'])
        self.W_ihr2.data.copy_(state_dict['W_ihr2'])
        self.W_hhr2.data.copy_(state_dict['W_hhr2'])
        self.b_r2.data.copy_(state_dict['b_r2'])
        self.W_ihn2.data.copy_(state_dict['W_ihn2'])
        self.W_hhn2.data.copy_(state_dict['W_hhn2'])
        self.b_n2.data.copy_(state_dict['b_n2'])
        self.W_policy.data.copy_(state_dict['W_policy'])
        self.b_policy.data.copy_(state_dict['b_policy'])
        self.W_value.data.copy_(state_dict['W_value'])
        self.b_value.data.copy_(state_dict['b_value'])