import mlx.core as mx
import mlx.nn as nn
import open_clip
import numpy as np
from PIL import Image
import torch
from train_vqgan_clip import VQGAN
from pathlib import Path

# # Basic usage
# python inference.py --checkpoint vqgan_output/checkpoint_epoch_100.npz --prompt "a beautiful sunset over mountains"

# # Generate multiple images with custom temperature
# python inference.py \
#     --checkpoint vqgan_output/checkpoint_epoch_100.npz \
#     --prompt "a beautiful sunset over mountains" \
#     --num-images 4 \
#     --temperature 0.8 \
#     --output-dir my_generated_images \
#     --prefix sunset

class VQGANInference:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        # Load CLIP
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.clip_model.eval()
        
        # Load VQGAN
        self.vqgan = VQGAN()
        # Load the saved parameters
        checkpoint = np.load(checkpoint_path)
        params = {k: mx.array(v) for k, v in checkpoint.items()}
        self.vqgan.update(params)
        
    def get_text_embeddings(self, text_prompt: str) -> mx.array:
        """Convert text prompt to CLIP embeddings"""
        with torch.no_grad():
            text_features = self.clip_model.encode_text(self.tokenizer([text_prompt]))
        return mx.array(text_features.numpy())
    
    def sample_latents(self, batch_size: int = 1, temperature: float = 1.0) -> mx.array:
        """Sample latent vectors for generation"""
        # Get spatial dimensions from VQGAN architecture
        spatial_size = 16  # This should match your trained model
        embedding_dim = self.vqgan.vq_layer.embedding_dim
        
        # Sample random latents
        latents = mx.random.normal(
            (batch_size, embedding_dim, spatial_size, spatial_size)
        ) * temperature
        return latents
    
    def generate_images(
        self,
        text_prompt: str,
        num_samples: int = 1,
        temperature: float = 1.0,
        return_pil: bool = True
    ):
        """Generate images from text prompt"""
        # Get text embeddings
        text_embeddings = self.get_text_embeddings(text_prompt)
        
        # Sample latents
        latents = self.sample_latents(num_samples, temperature)
        
        # Generate images
        with mx.eval_mode():
            images = self.vqgan.decode(latents)
        
        # Convert to PIL images if requested
        if return_pil:
            # Convert from [-1, 1] to [0, 255]
            images = ((images + 1) * 127.5).clip(0, 255).astype(mx.uint8)
            images = images.numpy()
            
            pil_images = []
            for img in images:
                pil_images.append(Image.fromarray(img))
            return pil_images
        
        return images

def save_images(images, output_dir: str, prefix: str = "generated"):
    """Save generated images to disk"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for i, img in enumerate(images):
        img.save(output_dir / f"{prefix}_{i}.png")

if __name__ == "__main__":
    # Example usage
    model = VQGANInference(
        checkpoint_path="vqgan_output/checkpoint_epoch_100.npz"
    )
    
    # Generate images
    images = model.generate_images(
        text_prompt="a beautiful sunset over mountains",
        num_samples=4,
        temperature=0.8
    )
    
    # Save the generated images
    save_images(images, "generated_images")