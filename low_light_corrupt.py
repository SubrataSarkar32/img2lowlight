import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image

class ImageProcessing:
    def __init__(self, gamma=2.2, alpha=0.3, beta=0.8, noise_level=0.02, wb_gains=(1.5, 1.0, 0.8), quant_levels=256, device="cpu"):
        """
        Initialize image processing parameters.

        Args:
        - gamma: Gamma correction factor (for inverse gamma & final gamma correction)
        - alpha: Exposure degradation factor for low-light corruption
        - beta: Gamma correction for low-light simulation
        - noise_level: Gaussian noise level for low-light corruption
        - wb_gains: White balance inversion gains (R, G, B)
        - quant_levels: Levels for quantization (default 256 for 8-bit)
        - device: "cuda" or "cpu"
        """
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.noise_level = noise_level
        self.wb_gains = wb_gains
        self.quant_levels = quant_levels
        self.device = device

    def invert_tone_mapping(self, image_tensor):
        """Inverts tone mapping using an approximation of inverse Reinhard function."""
        return torch.pow(image_tensor, 2.2)

    def invert_gamma_correction(self, image_tensor):
        """Inverts gamma correction."""
        return torch.pow(image_tensor, self.gamma)

    def srgb_to_crgb(self, image_tensor):
        """Converts sRGB to Camera RGB (cRGB) using inverse sRGB transformation."""
        threshold = 0.04045
        return torch.where(image_tensor <= threshold, 
                           image_tensor / 12.92, 
                           torch.pow((image_tensor + 0.055) / 1.055, 2.4))

    def invert_white_balance(self, image_tensor):
        """Inverts white balance using predefined channel-wise gains."""
        r_gain, g_gain, b_gain = self.wb_gains
        wb_matrix = torch.tensor([r_gain, g_gain, b_gain]).view(3, 1, 1).to(self.device)
        return torch.clamp(image_tensor * wb_matrix, 0, 1)

    def perform_low_light_corruption(self, image_tensor):
        """Applies low-light corruption by reducing brightness, adding noise, and adjusting gamma."""
        # Reduce brightness
        low_light_img = image_tensor * self.alpha
        # Apply gamma correction
        low_light_img = torch.pow(low_light_img, self.beta)
        # Add Gaussian noise
        noise = torch.randn_like(low_light_img) * self.noise_level
        return torch.clamp(low_light_img + noise, 0, 1)

    def quantization(self, image_tensor):
        """Applies quantization to reduce color depth."""
        image_tensor = torch.round(image_tensor * (self.quant_levels - 1)) / (self.quant_levels - 1)
        return image_tensor

    def white_balance(self, image_tensor):
        """Applies white balancing by normalizing the color channels."""
        r, g, b = image_tensor[0], image_tensor[1], image_tensor[2]
        avg_r, avg_g, avg_b = torch.mean(r), torch.mean(g), torch.mean(b)

        # Compute gain factors
        gain_r, gain_g, gain_b = avg_g / avg_r, 1.0, avg_g / avg_b
        wb_matrix = torch.tensor([gain_r, gain_g, gain_b]).view(3, 1, 1).to(self.device)

        return torch.clamp(image_tensor * wb_matrix, 0, 1)

    def crgb_to_srgb(self, image_tensor):
        """Converts Camera RGB (cRGB) back to sRGB using standard transformation."""
        threshold = 0.0031308
        return torch.where(image_tensor <= threshold,
                           image_tensor * 12.92,
                           1.055 * torch.pow(image_tensor, 1 / 2.4) - 0.055)

    def gamma_correction(self, image_tensor):
        """Applies gamma correction."""
        return torch.pow(image_tensor, 1 / self.gamma)

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

# Process a single image
def process_image(input_path, output_path, device="cpu"):
    processor = ImageProcessing(device=device)

    # Load image
    image_tensor = load_image(input_path, device)

    # Apply processing steps
    image_tensor = processor.invert_tone_mapping(image_tensor)
    image_tensor = processor.invert_gamma_correction(image_tensor)
    image_tensor = processor.srgb_to_crgb(image_tensor)
    image_tensor = processor.invert_white_balance(image_tensor)
    image_tensor = processor.perform_low_light_corruption(image_tensor)
    image_tensor = processor.quantization(image_tensor)
    image_tensor = processor.white_balance(image_tensor)
    image_tensor = processor.crgb_to_srgb(image_tensor)
    image_tensor = processor.gamma_correction(image_tensor)

    # Save processed image
    save_image(image_tensor, output_path)

# Process all images in a folder recursively while maintaining directory structure
def process_folder(input_folder, output_folder, device="cpu"):
    processor = ImageProcessing(device=device)
    
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_folder)  # Preserve structure
                output_path = os.path.join(output_folder, relative_path)

                print(f"Processing: {input_path} -> {output_path}")

                # Load image
                image_tensor = load_image(input_path, device)

                # Apply processing steps
                image_tensor = processor.invert_tone_mapping(image_tensor)
                image_tensor = processor.invert_gamma_correction(image_tensor)
                image_tensor = processor.srgb_to_crgb(image_tensor)
                image_tensor = processor.invert_white_balance(image_tensor)
                image_tensor = processor.perform_low_light_corruption(image_tensor)
                image_tensor = processor.quantization(image_tensor)
                image_tensor = processor.white_balance(image_tensor)
                image_tensor = processor.crgb_to_srgb(image_tensor)
                image_tensor = processor.gamma_correction(image_tensor)

                # Save processed image
                save_image(image_tensor, output_path)

# Example usage
if __name__ == "__main__":
    input_folder = "animals"  # Folder containing original images
    output_folder = "animals_low_light"  # Where to save transformed images
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Processing images from {input_folder} to {output_folder} using {device.upper()}...")
    process_folder(input_folder, output_folder, device)
    print("Processing complete!")
