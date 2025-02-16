import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path
import numpy as np
from PIL import Image
import torch  # For CLIP model loading
import open_clip
import glob
import os
from tqdm import tqdm
import pickle

class VQGAN(nn.Module):
    def __init__(self, 
                 input_channels=3,
                 hidden_dims=[128, 256, 512],
                 n_embeddings=1024,
                 embedding_dim=32):
        super().__init__()
        
        # Encoder
        modules = []
        channels = input_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(min(32, h_dim), h_dim),
                    nn.SiLU()
                )
            )
            channels = h_dim
        
        # Add final convolution to get to embedding dimension
        modules.append(
            nn.Conv2d(channels, embedding_dim, kernel_size=1, stride=1, padding=0)
        )
        
        self.encoder = nn.Sequential(*modules)
        
        # Vector Quantizer
        self.vq_layer = VectorQuantizer(n_embeddings, embedding_dim)
        
        # Decoder
        modules = []
        hidden_dims.reverse()
        
        # Initial convolution to match embedding dimension
        modules.append(
            nn.Conv2d(embedding_dim, hidden_dims[0], kernel_size=1, stride=1, padding=0)
        )
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                     hidden_dims[i + 1],
                                     kernel_size=4,
                                     stride=2,
                                     padding=1),
                    nn.GroupNorm(min(32, hidden_dims[i + 1]), hidden_dims[i + 1]),
                    nn.SiLU()
                )
            )
        
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                 input_channels,
                                 kernel_size=4,
                                 stride=2,
                                 padding=1),
                nn.Tanh()
            )
        )

        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        encoding = self.encoder(x)
        # Preserve spatial dimensions for VQ layer
        b, c, h, w = encoding.shape
        # Reshape to (batch_size * height * width, embedding_dim)
        encoding = encoding.reshape(-1, c)  # Flatten spatial dimensions
        quantized, indices = self.vq_layer.forward(encoding)
        # Reshape back to 4D: (batch, channels, height, width)
        quantized = quantized.reshape(b, c, h, w)
        return quantized, indices
    
    def decode(self, quantized):
        return self.decoder(quantized)
    
    def forward(self, x):
        quantized, indices = self.encode(x)
        return self.decode(quantized), quantized, indices

class VectorQuantizer(nn.Module):
    def __init__(self, n_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        
    def forward(self, z):
        # z shape: (batch_size * height * width, embedding_dim)
        
        # Calculate distances
        z_flattened = z.reshape(-1, self.embedding_dim)
        
        # Calculate distances with correct dimensions
        d = mx.sum(z_flattened ** 2, axis=1, keepdims=True) + \
            mx.sum(self.embedding.weight ** 2, axis=1) - \
            2 * mx.matmul(z_flattened, self.embedding.weight.transpose())
        
        # Get minimum distances
        encoding_indices = mx.argmin(d, axis=1)
        
        # Get quantized vectors directly from embeddings
        quantized = self.embedding.weight[encoding_indices]
        
        return quantized, encoding_indices

def load_and_preprocess_images(image_folder, image_size=224, cache_file="processed_images.pkl"):
    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"Loading cached images from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    image_paths = glob.glob(os.path.join(image_folder, '**/*.jpg'), recursive=True)
    images = []
    
    for path in tqdm(image_paths, desc="Loading images"):
        img = Image.open(path).convert('RGB')
        # Ensure exact size of 224x224 for CLIP
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img = np.array(img) / 127.5 - 1  # Normalize to [-1, 1]
        images.append(img)
    
    images = mx.array(np.stack(images))
    
    # Cache the processed images
    print(f"Caching processed images to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(images, f)
    
    return images

def train_vqgan_clip(
    media_folder: str,
    output_dir: str,
    batch_size: int = 8,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    image_size: int = 224,
    cache_file: str = "processed_images.pkl"
):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CLIP model
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip_model.eval()
    
    # Initialize VQGAN
    vqgan = VQGAN()
    
    # Load dataset with caching - force 224x224
    images = load_and_preprocess_images(media_folder, image_size=224, cache_file=cache_file)
    
    # Setup optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)
    
    # Define loss function for gradient calculation
    def loss_fn(params, batch):
        vqgan.update(params)  # Update model parameters
        reconstructed, quantized, indices = vqgan.forward(batch)
        reconstruction_loss = mx.mean((batch - reconstructed) ** 2)
        
        # Convert MLX arrays to numpy for CLIP
        clip_input = mx.array(reconstructed).astype(mx.float32)
        clip_input = (clip_input + 1) / 2.0
        clip_input = clip_input.transpose(0, 3, 1, 2)
        
        clip_input = torch.tensor(clip_input.tolist())
        if clip_input.shape[-1] != 224 or clip_input.shape[-2] != 224:
            clip_input = torch.nn.functional.interpolate(
                clip_input,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )
        
        with torch.no_grad():
            image_features = clip_model.encode_image(clip_input)
            text_features = clip_model.encode_text(
                tokenizer(["a high quality image"] * batch_size)
            )
        
        # CLIP loss
        clip_loss = mx.mean(
            mx.matmul(
                mx.array(image_features.numpy()),
                mx.array(text_features.numpy()).T
            )
        )
        
        # Add weighting factors
        total_loss = reconstruction_loss + 0.1 * clip_loss
        
        return total_loss

    # Training loop
    try:
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = len(images) // batch_size
            
            for i in range(num_batches):
                batch = images[i * batch_size:(i + 1) * batch_size]
                loss_value, gradients = mx.value_and_grad(loss_fn)(vqgan.parameters(), batch)
                optimizer.update(vqgan, gradients)
                total_loss += loss_value.item()
                
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % 10 == 0:
                # Save parameters as a dictionary of numpy arrays
                checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.npz")
                params_dict = {k: v.numpy() for k, v in vqgan.parameters().items()}
                np.savez(checkpoint_path, **params_dict)
                print(f"Saved checkpoint to {checkpoint_path}")
                
    except Exception as e:
        print(f"Training interrupted: {e}")
        # Save emergency checkpoint
        emergency_path = os.path.join(output_dir, "emergency_checkpoint.npz")
        params_dict = {k: v.numpy() for k, v in vqgan.parameters().items()}
        np.savez(emergency_path, **params_dict)
        print(f"Saved emergency checkpoint to {emergency_path}")
        raise e

if __name__ == "__main__":
    train_vqgan_clip(
        media_folder="media",
        output_dir="vqgan_output",
        batch_size=8,
        num_epochs=100,
        learning_rate=1e-4,
        image_size=224
    )