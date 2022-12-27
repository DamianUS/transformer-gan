import torch
import torch.nn
from models.Discriminator import DiscriminatorTransformer
from models.Generator import GeneratorTransformer

class TransformerGAN(torch.nn.Module):
    def __init__(self, num_features, seq_len, batch_size, num_layers, hidden_dim, narrow_attn_heads, dropout=0, noise_length=100):
        super(TransformerGAN, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.noise_length = noise_length
        self.generator = GeneratorTransformer(n_features=num_features, hidden_dim=hidden_dim*2, seq_len=seq_len, narrow_attn_heads=narrow_attn_heads*2, num_layers=num_layers, dropout=dropout, noise_length=noise_length)
        self.discriminator = DiscriminatorTransformer(n_features=num_features, hidden_dim=hidden_dim, seq_len=seq_len, narrow_attn_heads=narrow_attn_heads, num_layers=num_layers, dropout=dropout)

    def forward(self, X, obj='discriminator'):
        assert obj in ['generator','discriminator'], "obj must be either generator or discriminator"
        if obj == 'generator':
            device = next(self.parameters()).device
            noise = torch.randn((self.batch_size, self.noise_length)).float().to(device)
            return self.generator(noise)
        elif obj == 'discriminator':
            return self.discriminator(X)

