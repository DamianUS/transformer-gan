import torch
import torch.nn
from models.Discriminator import DiscriminatorTransformer
from models.Generator import GeneratorTransformer

class TransformerGAN(torch.nn.Module):
    def __init__(self, num_features, seq_len, batch_size, gen_num_layers, dis_num_layers, gen_hidden_dim, dis_hidden_dim, gen_narrow_attn_heads, dis_narrow_attn_heads, gen_dropout=0, dis_dropout=0, noise_length=100):
        super(TransformerGAN, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.noise_length = noise_length
        self.generator = GeneratorTransformer(n_features=num_features, hidden_dim=gen_hidden_dim, seq_len=seq_len, narrow_attn_heads=gen_narrow_attn_heads, num_layers=gen_num_layers, dropout=gen_dropout, noise_length=noise_length)
        self.discriminator = DiscriminatorTransformer(n_features=num_features, hidden_dim=dis_hidden_dim, seq_len=seq_len, narrow_attn_heads=dis_narrow_attn_heads, num_layers=dis_num_layers, dropout=dis_dropout)

    def forward(self, X, obj='discriminator'):
        print([name for name, _ in self.discriminator.named_children()])
        print([name for name, _ in self.discriminator.transformer_encoder.named_children()])
        assert obj in ['generator','discriminator'], "obj must be either generator or discriminator"
        if obj == 'generator':
            device = next(self.parameters()).device
            noise = torch.randn((self.batch_size, self.noise_length)).float().to(device)
            return self.generator(noise)
        elif obj == 'discriminator':
            return self.discriminator(X)

