import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image

class MAETLowLightTransform:
    def __init__(self, alpha=0.3, beta=0.8, noise_level=0.02, device="cpu"):
        """
        Initialize the transformation with specific parameters.
        
        Args:
        - alpha: Exposure degradation factor (lower = darker)
        - beta: Gamma correction factor
        - noise_level: Amount of noise to add
        - device: "cuda" or "cpu" for PyTorch tensor computations
        """
        self.alpha = alpha
        self.beta = beta
        self.noise_level = noise_level
        self.device = device

    def apply_transform(self, image_tensor):
        """
        Apply the physics-based low-light degradation on an image tensor.

        Args:
        - image_tensor: A PyTorch tensor of shape (C, H, W), range [0, 1]
        
        Returns:
        - Transformed low-light image tensor
        """
        # Ensure image is on the correct device
        image_tensor = image_tensor.to(self.device)

        # Step 1: Reduce brightness (Exposure degradation)
        low_light_img = image_tensor * self.alpha

        # Step 2: Apply gamma correction (non-linear transformation)
        low_light_img = torch.pow(low_light_img, self.beta)

        # Step 3: Add Gaussian noise
        noise = torch.randn_like(low_light_img) * self.noise_level
        low_light_img = torch.clamp(low_light_img + noise, 0, 1)

        return low_light_img

# Load and preprocess an image
def load_image(image_path, device="cpu"):
    transform = transforms.Compose([
        transforms.ToTensor()  # Converts to (C, H, W) format with range [0, 1]
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).to(device)

# Convert tensor back to image and save
def save_image(tensor, output_path):
    transform = transforms.ToPILImage()
    image = transform(tensor.cpu())  # Convert to PIL image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure output directory exists
    image.save(output_path)

# Process all images in a folder recursively while maintaining directory structure
def process_folder(input_folder, output_folder, device="cpu"):
    # Initialize the MAET transformation
    maet = MAETLowLightTransform(device=device)
    
    # Walk through the directory
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):  # Supported formats
                input_path = os.path.join(root, file)
                
                # Create corresponding output path
                relative_path = os.path.relpath(input_path, input_folder)  # Preserve structure
                output_path = os.path.join(output_folder, relative_path)

                print(f"Processing: {input_path} -> {output_path}")

                # Load image
                image_tensor = load_image(input_path, device)

                # Apply low-light transformation
                low_light_tensor = maet.apply_transform(image_tensor)

                # Save output image
                save_image(low_light_tensor, output_path)

# Example usage
if __name__ == "__main__":
    input_folder = "animals"  # Folder containing original images
    output_folder = "animals_low_light_images"  # Where to save transformed images
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Processing images from {input_folder} to {output_folder} using {device.upper()}...")
    process_folder(input_folder, output_folder, device)
    print("Processing complete!")
