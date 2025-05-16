# image_utils.py
# Description: Utility functions for image manipulation using Pillow (PIL) and NumPy.

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import colorsys # For HSL/HSV color space conversions (used in color shift)
# import torch # Only if torch-based functions are to be kept and used.

# --- PIL <-> Tensor Conversion (Potentially Unused if Torch is removed) ---
# These functions were in the original script. If your refactored version
# exclusively uses PIL/OpenCV for image operations and does not involve PyTorch tensors
# for the core image processing pipeline, these can be removed.
# If they are used by some part of the logic you intend to keep, ensure Torch is installed.

# def tensor_to_pil_list_zoomer(tensor_image):
#     """Converts a PyTorch tensor or list of tensors to a list of PIL Images."""
#     if not isinstance(tensor_image, torch.Tensor):
#         if isinstance(tensor_image, list) and all(isinstance(img, Image.Image) for img in tensor_image):
#             return tensor_image
#         if isinstance(tensor_image, Image.Image):
#             return [tensor_image]
#         raise ValueError("Input must be a torch Tensor or a PIL Image/list of PIL Images.")

#     if tensor_image.ndim == 3: # Add batch dimension if single image tensor
#         tensor_image = tensor_image.unsqueeze(0)
#     elif tensor_image.ndim != 4: # Expects (batch, height, width, channels)
#         raise ValueError(f"Expected a 3D or 4D tensor, got {tensor_image.ndim}D")

#     tensor_image_cpu = tensor_image.detach().cpu() # Move to CPU and detach from graph
#     images = []
#     for i in range(tensor_image_cpu.shape[0]): # Iterate over batch
#         img_np = tensor_image_cpu[i].numpy()
#         # Assuming tensor values are in [0, 1], scale to [0, 255] and convert to uint8
#         img_np = (img_np * 255).astype(np.uint8)
        
#         pil_img = None
#         if img_np.shape[2] == 1: # Grayscale
#             pil_img = Image.fromarray(img_np[:, :, 0], 'L').convert("RGB") # Convert to RGB for consistency
#         elif img_np.shape[2] == 3: # RGB
#             pil_img = Image.fromarray(img_np, 'RGB')
#         elif img_np.shape[2] == 4: # RGBA
#             pil_img = Image.fromarray(img_np, 'RGBA').convert('RGB') # Convert to RGB, discarding alpha
#         else:
#             raise ValueError(f"Unsupported number of channels: {img_np.shape[2]}")
#         images.append(pil_img)
#     return images

# def pil_list_to_tensor_zoomer(pil_images):
#     """Converts a list of PIL Images to a PyTorch tensor."""
#     tensor_images = []
#     if not pil_images:
#         print("Warning: image_utils.pil_list_to_tensor_zoomer received an empty list.")
#         # Return an empty tensor with expected channel dimension if needed by downstream torch code
#         return torch.empty(0, 1, 1, 3, dtype=torch.float32) 

#     for image in pil_images:
#         if not isinstance(image, Image.Image):
#             raise ValueError("All items in pil_images must be PIL.Image objects.")
        
#         rgb_image = image.convert("RGB") # Ensure image is RGB
#         img_np = np.array(rgb_image).astype(np.float32) / 255.0 # Convert to float32 and normalize to [0,1]
#         tensor_images.append(torch.from_numpy(img_np)) # Convert NumPy array to tensor
    
#     if not tensor_images: # Should not happen if pil_images was not empty, but as a safeguard
#         return torch.empty(0, 1, 1, 3, dtype=torch.float32)
        
#     return torch.stack(tensor_images) # Stack list of tensors into a single batch tensor


# --- Core Image Transformation Functions ---

def apply_centered_zoom_pil_zoomer(pil_image_rgb, scale_factor, interpolation_str="Lanczos"):
    """
    Applies a centered zoom to a PIL Image.
    Args:
        pil_image_rgb (PIL.Image): Input image, expected to be in RGB mode.
        scale_factor (float): Zoom factor. >1 zooms in, <1 zooms out.
        interpolation_str (str): Resampling filter ("Nearest", "Bilinear", "Bicubic", "Lanczos").
    Returns:
        PIL.Image: The zoomed PIL Image, maintaining original dimensions by cropping/padding.
    """
    interpolation_map = {
        "Nearest": Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST,
        "Bilinear": Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR,
        "Bicubic": Image.Resampling.BICUBIC if hasattr(Image, 'Resampling') else Image.BICUBIC,
        "Lanczos": Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS,
    }
    # Default to Lanczos if the string is not recognized
    interpolation_method = interpolation_map.get(interpolation_str, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

    if abs(scale_factor - 1.0) < 1e-6: # If scale factor is effectively 1, return original
        return pil_image_rgb

    original_width, original_height = pil_image_rgb.size
    
    # Calculate new dimensions after scaling
    scaled_width = int(original_width * scale_factor)
    scaled_height = int(original_height * scale_factor)

    if scaled_width <= 0 or scaled_height <= 0: # Avoid invalid dimensions
        # print(f"image_utils: Warning - Invalid scaled dimensions ({scaled_width}x{scaled_height}). Returning original.")
        return pil_image_rgb # Or return a black image of original size, or raise error

    # Resize the image
    scaled_image = pil_image_rgb.resize((scaled_width, scaled_height), resample=interpolation_method)

    # Calculate coordinates for cropping to maintain original dimensions and center the zoom
    left = (scaled_width - original_width) / 2.0
    top = (scaled_height - original_height) / 2.0
    right = (scaled_width + original_width) / 2.0
    bottom = (scaled_height + original_height) / 2.0
    
    # Crop the scaled image
    # The crop box is (left, top, right, bottom)
    cropped_image = scaled_image.crop((left, top, right, bottom))
    
    return cropped_image

# --- Breathing Effect Visual Transformation Functions ---

def apply_breathing_zoom(pil_image_rgb, base_scale, breath_intensity, max_zoom_factor=0.1, zoom_interpolation_str="Lanczos"):
    """
    Applies zoom to an image based on breath intensity.
    Args:
        pil_image_rgb (PIL.Image): Input RGB image.
        base_scale (float): The baseline scale factor (usually 1.0).
        breath_intensity (float): Normalized breath intensity (0 to 1).
        max_zoom_factor (float): Maximum additional zoom to apply at full intensity (e.g., 0.1 for 10% extra zoom).
        zoom_interpolation_str (str): Interpolation method for zooming.
    Returns:
        PIL.Image: Zoomed image.
    """
    # Calculate the effective scale factor: base_scale + intensity-driven additional zoom
    current_scale_factor = base_scale + (breath_intensity * max_zoom_factor)
    return apply_centered_zoom_pil_zoomer(pil_image_rgb, current_scale_factor, zoom_interpolation_str)

def apply_breathing_brightness(pil_image_rgb, breath_intensity, min_brightness=0.8, max_brightness=1.2):
    """
    Modulates the brightness of an image based on breath intensity.
    Args:
        pil_image_rgb (PIL.Image): Input RGB image.
        breath_intensity (float): Normalized breath intensity (0 to 1).
        min_brightness (float): Brightness factor at zero intensity.
        max_brightness (float): Brightness factor at full intensity.
    Returns:
        PIL.Image: Image with adjusted brightness.
    """
    # Interpolate brightness factor based on intensity
    brightness_factor = min_brightness + (breath_intensity * (max_brightness - min_brightness))
    brightness_factor = max(0.0, brightness_factor) # Ensure brightness is not negative

    enhancer = ImageEnhance.Brightness(pil_image_rgb)
    return enhancer.enhance(brightness_factor)

def apply_breathing_blur(pil_image_rgb, breath_intensity, max_blur_radius=2.0):
    """
    Applies Gaussian blur to an image based on breath intensity.
    Args:
        pil_image_rgb (PIL.Image): Input RGB image.
        breath_intensity (float): Normalized breath intensity (0 to 1).
        max_blur_radius (float): Maximum blur radius (in pixels) at full intensity.
    Returns:
        PIL.Image: Blurred image.
    """
    blur_radius = breath_intensity * max_blur_radius
    if blur_radius < 0.1: # If blur radius is very small, skip filtering for performance
        return pil_image_rgb
    return pil_image_rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))

def apply_breathing_saturation(pil_image_rgb, breath_intensity, min_saturation=0.7, max_saturation=1.3):
    """
    Modulates the color saturation of an image based on breath intensity.
    Args:
        pil_image_rgb (PIL.Image): Input RGB image.
        breath_intensity (float): Normalized breath intensity (0 to 1).
        min_saturation (float): Saturation factor at zero intensity (0=grayscale, 1=original).
        max_saturation (float): Saturation factor at full intensity.
    Returns:
        PIL.Image: Image with adjusted saturation.
    """
    saturation_factor = min_saturation + (breath_intensity * (max_saturation - min_saturation))
    saturation_factor = max(0.0, saturation_factor) # Saturation cannot be negative

    enhancer = ImageEnhance.Color(pil_image_rgb) # Color enhancer adjusts saturation
    return enhancer.enhance(saturation_factor)

def apply_breathing_color_shift(pil_image_rgb, breath_intensity, min_hue_shift=0.0, max_hue_shift=0.15):
    """
    Shifts the hue of an image based on breath intensity.
    Args:
        pil_image_rgb (PIL.Image): Input RGB image.
        breath_intensity (float): Normalized breath intensity (0 to 1).
        min_hue_shift (float): Hue shift at zero intensity (0 to 1, fraction of hue circle).
        max_hue_shift (float): Hue shift at full intensity (0 to 1).
    Returns:
        PIL.Image: Image with shifted hue, converted back to RGB.
    """
    if abs(max_hue_shift - min_hue_shift) < 1e-6 and abs(min_hue_shift) < 1e-6: # No shift to apply
        return pil_image_rgb

    # Calculate the current hue shift based on intensity
    current_hue_shift = min_hue_shift + (breath_intensity * (max_hue_shift - min_hue_shift))
    
    if abs(current_hue_shift) < 1e-6: # If effective shift is negligible
        return pil_image_rgb

    # Convert PIL image to NumPy array for manipulation
    # Ensure it's RGB first, as HSV conversion expects 3 channels
    img_rgb = pil_image_rgb.convert("RGB")
    arr_rgb_float = np.array(img_rgb).astype(np.float32) / 255.0 # Normalize to [0,1] for colorsys

    # Separate R, G, B channels
    r, g, b = arr_rgb_float[..., 0], arr_rgb_float[..., 1], arr_rgb_float[..., 2]
    
    # Vectorize colorsys functions for efficient application to NumPy arrays
    # This converts each pixel's (r,g,b) to (h,s,v)
    hsv_converter = np.vectorize(colorsys.rgb_to_hsv)
    h, s, v = hsv_converter(r, g, b)
    
    # Apply hue shift (h is in [0,1], so add shift and take modulo 1.0)
    h_shifted = (h + current_hue_shift) % 1.0
    
    # Vectorize hsv_to_rgb conversion
    rgb_converter = np.vectorize(colorsys.hsv_to_rgb)
    r_new, g_new, b_new = rgb_converter(h_shifted, s, v)
    
    # Stack new R, G, B channels and convert back to uint8 image format
    arr_rgb_shifted_float = np.stack([r_new, g_new, b_new], axis=-1)
    arr_rgb_shifted_uint8 = np.clip(arr_rgb_shifted_float * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(arr_rgb_shifted_uint8, "RGB")

