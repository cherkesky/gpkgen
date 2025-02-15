import mlx.core as mx
import mlx.nn as nn

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=256, beta=0.25):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.beta = beta

    def __call__(self, z):
        # Reshape z -> (batch, height, width, channel) to (batch * height * width, channel)
        z_flattened = z.reshape(-1, z.shape[-1])
        
        # Distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2ze
        d = mx.sum(z_flattened ** 2, axis=1, keepdims=True) + \
            mx.sum(self.embedding.weight ** 2, axis=1) - \
            2 * mx.matmul(z_flattened, self.embedding.weight.T)
        
        # Find nearest encoding
        min_encoding_indices = mx.argmin(d, axis=1)
        z_q = self.embedding(min_encoding_indices).reshape(z.shape)
        
        # Straight through estimator
        z_q = z + mx.stop_gradient(z_q - z)
        
        return z_q, min_encoding_indices

class VQGAN(nn.Module):
    def __init__(self, latent_dim=256, num_embeddings=1024):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, latent_dim, kernel_size=3, stride=1, padding=1)
        )
        
        # Vector Quantizer
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def __call__(self, x):
        # Encode
        z = self.encoder(x.transpose(0, 3, 1, 2))
        
        # Quantize
        z_q, indices = self.quantizer(z.transpose(0, 2, 3, 1))
        
        # Decode
        x_recon = self.decoder(z_q.transpose(0, 3, 1, 2))
        
        return x_recon.transpose(0, 2, 3, 1), z_q, indices 