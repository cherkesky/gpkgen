import mlx.core as mx
import mlx.nn as nn
import open_clip
import numpy as np
from PIL import Image
import torch
from train_vqgan_clip import VQGAN
from pathlib import Path
import argparse

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
        
        # Load and inspect the checkpoint
        print("\n=== Checkpoint Inspection ===")
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        
        # Print all available keys
        print("\nCheckpoint keys:")
        for key in checkpoint.files:
            array = checkpoint[key]
            if isinstance(array, np.ndarray):
                print(f"- {key}: shape={array.shape}, dtype={array.dtype}")
            else:
                print(f"- {key}: type={type(array)}")
        
        # Print some basic statistics for a few key arrays
        print("\nKey arrays statistics:")
        for key in checkpoint.files:
            array = checkpoint[key]
            if isinstance(array, np.ndarray) and array.dtype != np.dtype('O'):
                print(f"\n{key}:")
                print(f"  - Min: {np.min(array)}")
                print(f"  - Max: {np.max(array)}")
                print(f"  - Mean: {np.mean(array)}")
                print(f"  - Std: {np.std(array)}")
        
        # Convert parameters recursively
        def convert_to_mlx(v):
            if isinstance(v, np.ndarray):
                if v.dtype == np.dtype('O'):
                    if v.size == 1:
                        return convert_to_mlx(v.item())
                    return [convert_to_mlx(item) for item in v]
                return mx.array(v.astype(np.float32))
            elif isinstance(v, dict):
                return {k: convert_to_mlx(v) for k, v in v.items()}
            elif isinstance(v, list):
                if all(not isinstance(x, (dict, list)) for x in v):
                    return mx.array(v)
                return [convert_to_mlx(item) for item in v]
            return v
            
        params = {k: convert_to_mlx(v) for k, v in checkpoint.items()}
        self.vqgan.update(params)
        print("\n=== Checkpoint Loading Complete ===\n")
        
    def get_text_embeddings(self, text_prompt: str) -> mx.array:
        """Convert text prompt to CLIP embeddings"""
        try:
            with torch.no_grad():
                # Tokenize and encode text
                text = self.tokenizer([text_prompt])
                text_features = self.clip_model.encode_text(text)
                
                # Normalize the features
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Convert to numpy then MLX array
                text_features_np = text_features.cpu().numpy().astype(np.float32)
                text_features_np = text_features_np.squeeze()  # Remove extra dimensions
                
                # Create MLX array and ensure proper shape
                mlx_features = mx.array(text_features_np)
                
                return mlx_features
                
        except Exception as e:
            print(f"Error in get_text_embeddings: {str(e)}")
            raise
    
    def sample_latents(self, batch_size: int = 1, temperature: float = 1.0) -> mx.array:
        """Sample latent vectors for generation"""
        try:
            spatial_size = 16  # This should match your trained model
            embedding_dim = self.vqgan.vq_layer.embedding_dim
            
            # Use truncated normal distribution to avoid extreme values
            np_latents = np.clip(
                np.random.normal(
                    loc=0.0,
                    scale=1.0,  # Increased scale significantly
                    size=(batch_size, embedding_dim, spatial_size, spatial_size)
                ),
                -2,  # Clip values to avoid extremes
                2
            ).astype(np.float32)
            
            latents = mx.array(np_latents)
            
            # Apply temperature without additional scaling
            latents = latents * temperature
            
            return latents
            
        except Exception as e:
            print(f"Error in sample_latents: {str(e)}")
            raise
    
    def generate_images(
        self,
        text_prompt: str,
        num_samples: int = 1,
        temperature: float = 1.0,
        return_pil: bool = True
    ):
        """Generate images using VQGAN"""
        try:
            # Keep text embeddings calculation for future use and debugging
            text_embeddings = self.get_text_embeddings(text_prompt)
            if text_embeddings is None:
                raise ValueError("Text embeddings evaluation returned None")
            print(f"Text embeddings shape: {text_embeddings.shape}")
            print(f"Text embeddings range: {mx.min(text_embeddings):.3f} to {mx.max(text_embeddings):.3f}")
            
            # Sample latents with reduced temperature
            latents = self.sample_latents(num_samples, temperature * 0.5)  # Reduced temperature
            if latents is None:
                raise ValueError("Latents generation returned None")
            print(f"Latents shape before transpose: {latents.shape}")
            print(f"Latents range: {mx.min(latents):.3f} to {mx.max(latents):.3f}")
            
            # Convert to proper format for decoder
            latents = mx.transpose(latents, (0, 2, 3, 1))
            print(f"Latents shape after transpose: {latents.shape}")
            
            # Generate images
            print("Calling VQGAN decode...")
            images = self.vqgan.decode(latents)
            print(f"Decoded images shape: {images.shape}")
            print(f"Decoded images range: {mx.min(images):.3f} to {mx.max(images):.3f}")
            
            # Convert to PIL images if requested
            if return_pil:
                # Convert from [-1, 1] to [0, 255]
                images = ((images + 1) * 127.5)
                images = mx.clip(images, 0, 255).astype(mx.uint8)
                
                # Convert to numpy array for PIL
                images_np = np.array(images)
                print(f"Final image shape: {images_np.shape}")
                print(f"Final image range: {np.min(images_np)} to {np.max(images_np)}")
                
                pil_images = []
                for img in images_np:
                    pil_images.append(Image.fromarray(img))
                return pil_images
            
            return images
            
        except Exception as e:
            print(f"Error in generate_images: {str(e)}")
            raise

def save_images(images, output_dir: str, prefix: str = "generated"):
    """Save generated images to disk"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for i, img in enumerate(images):
        img.save(output_dir / f"{prefix}_{i}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images from text using VQGAN-CLIP')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for image generation')
    parser.add_argument('--num-images', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--output-dir', type=str, default='generated_images', help='Output directory')
    parser.add_argument('--prefix', type=str, default='generated', help='Prefix for output filenames')
    
    args = parser.parse_args()
    
    # Initialize model
    model = VQGANInference(checkpoint_path=args.checkpoint)
    
    # Generate images with adjusted temperature
    images = model.generate_images(
        text_prompt=args.prompt,
        num_samples=args.num_images,
        temperature=args.temperature  # Remove the 0.1 scaling here
    )
    
    # Save the generated images
    save_images(images, args.output_dir, args.prefix)