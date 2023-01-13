import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        slope = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * slope)  # even dimensions
        pe[:, 1::2] = torch.cos(position * slope)  # odd dimensions
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x is N, L, D
        # pe is 1, maxlen, D
        scaled_x = x * np.sqrt(self.d_model)
        encoded = scaled_x + self.pe[:, :x.size(1), :]
        #  TODO: Check if dropout here helps
        return encoded


class DiscriminatorTransformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    # Constructor
    def __init__(self, n_features, hidden_dim=128, seq_len=288, narrow_attn_heads=0, num_layers=6, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        # LAYERS
        self.positional_encoding = PositionalEncoding(max_len=seq_len, d_model=hidden_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=narrow_attn_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.proj = nn.Linear(n_features, hidden_dim)  # from n_features to encoder hidden dimensions
        self.linear = nn.Linear(hidden_dim, 1)  # from decoder hidden dimensions to classification
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, X):
        device = next(self.parameters()).device
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)
        # Positional encoding - Out size = (batch_size, sequence length, dim_model)
        source_sequence = self.positional_encoding(self.proj(X))
        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer_encoder(source_sequence)
        pool_out = self.pool(transformer_out.permute(0, 2, 1)).squeeze(-1)
        #out = self.linear(transformer_out.mean(dim=1)).squeeze(-1)
        out = self.linear(pool_out).squeeze(-1)
        return out

    def get_target_mask(self, size) -> torch.tensor:
        # Generates a square matrix where each row allows one event more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)