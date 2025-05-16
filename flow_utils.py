import numpy as np
import cv2 # OpenCV for image processing
from PIL import Image # Pillow for image manipulation

# Attempt to import project-specific utility modules
try:
    from image_utils import apply_centered_zoom_pil_zoomer # Assumed to be in image_utils.py
except ModuleNotFoundError:
    print("FlowUtils: Warning - 'image_utils.py' or 'apply_centered_zoom_pil_zoomer' not found. Flow target generation might fail.")
    # Define a mock function if image_utils or the specific function is missing
    def apply_centered_zoom_pil_zoomer(pil_image, scale_factor, resample_method_str):
        print("FlowUtils: Mock apply_centered_zoom_pil_zoomer called because original was not found.")
        # Simple resize as a fallback, may not be centered.
        width, height = pil_image.size
        new_width, new_height = int(width * scale_factor), int(height * scale_factor)
        # Map string to PIL resampling enum if possible
        resample_map = {
            "Nearest": Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST,
            "Bilinear": Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR,
            "Bicubic": Image.Resampling.BICUBIC if hasattr(Image, 'Resampling') else Image.BICUBIC,
            "Lanczos": Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS,
        }
        resample_method = resample_map.get(resample_method_str, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
        return pil_image.resize((new_width, new_height), resample=resample_method)


def calculate_optical_flow_optimized_zoomer(img1_pil_rgb, img2_pil_rgb, flow_calc_scale_factor=0.5, interpolation_str_cv2_resize="Linear_cv2"):
    """
    Calculates dense optical flow between two PIL RGB images.
    img1_pil_rgb: Previous frame (PIL Image, RGB)
    img2_pil_rgb: Current frame (PIL Image, RGB)
    flow_calc_scale_factor: Factor to downscale images before flow calculation (0.01 to 1.0).
                            Smaller values are faster but less accurate.
    interpolation_str_cv2_resize: OpenCV interpolation string for resizing flow field back to original.
                                  e.g., "Linear_cv2", "Nearest_cv2".
    Returns: A 2-channel NumPy array (height, width, 2) representing (dx, dy) flow.
    """
    cv2_interpolation_map_resize = {
        "Nearest_cv2": cv2.INTER_NEAREST, 
        "Linear_cv2": cv2.INTER_LINEAR, 
        "Cubic_cv2": cv2.INTER_CUBIC, 
        "Lanczos4_cv2": cv2.INTER_LANCZOS4
    }
    cv2_resize_interpolation = cv2_interpolation_map_resize.get(interpolation_str_cv2_resize, cv2.INTER_LINEAR)
    
    # PIL resampling for downscaling images (if needed)
    # Using Bilinear as a general-purpose, fast method for downscaling before flow.
    pil_resize_interpolation_downscale = Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR

    w1, h1 = img1_pil_rgb.size
    
    # Convert PIL images to OpenCV format (NumPy arrays, BGR, Grayscale)
    img1_gray_cv = cv2.cvtColor(np.array(img1_pil_rgb), cv2.COLOR_RGB2GRAY)
    img2_gray_cv = cv2.cvtColor(np.array(img2_pil_rgb), cv2.COLOR_RGB2GRAY)

    # Downscale images for faster flow calculation if scale_factor < 1.0
    if 0.01 <= flow_calc_scale_factor < 1.0:
        target_w, target_h = int(w1 * flow_calc_scale_factor), int(h1 * flow_calc_scale_factor)
        target_w, target_h = max(1, target_w), max(1, target_h) # Ensure dimensions are at least 1
        
        # Resize grayscale OpenCV images
        img1_small_gray_cv = cv2.resize(img1_gray_cv, (target_w, target_h), interpolation=cv2_resize_interpolation) # Use cv2 resize for np arrays
        img2_small_gray_cv = cv2.resize(img2_gray_cv, (target_w, target_h), interpolation=cv2_resize_interpolation)
    else: # Use original size
        img1_small_gray_cv, img2_small_gray_cv = img1_gray_cv, img2_gray_cv

    # Calculate Farneback optical flow on (potentially downscaled) grayscale images
    # Parameters for cv2.calcOpticalFlowFarneback:
    # prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
    # These are common defaults; might need tuning for specific visual styles.
    flow_small_xy = cv2.calcOpticalFlowFarneback(
        img1_small_gray_cv, img2_small_gray_cv, 
        None,        # flow: output flow image (or None to create)
        0.5,         # pyr_scale: image pyramid or Infinite Impulse Response (IIR) filter scale
        3,           # levels: number of pyramid levels
        15,          # winsize: averaging window size
        3,           # iterations: number of iterations at each pyramid level
        5,           # poly_n: size of the pixel neighborhood used to find polynomial expansion
        1.2,         # poly_sigma: standard deviation of the Gaussian used to smooth derivatives
        0            # flags: operation flags (0 for none)
    )

    # Upscale flow field back to original image dimensions if it was downscaled
    if 0.01 <= flow_calc_scale_factor < 1.0:
        flow_dx_orig_res = cv2.resize(flow_small_xy[...,0], (w1,h1), interpolation=cv2_resize_interpolation) / flow_calc_scale_factor
        flow_dy_orig_res = cv2.resize(flow_small_xy[...,1], (w1,h1), interpolation=cv2_resize_interpolation) / flow_calc_scale_factor
        # Division by flow_calc_scale_factor compensates for the change in coordinate system due to scaling.
        return np.dstack((flow_dx_orig_res, flow_dy_orig_res))
    
    return flow_small_xy # Return flow at original resolution if no downscaling was done


def warp_image_cv2_zoomer(image_np_rgba_uint8, flow_field_xy, interpolation_str_cv2_warp="Linear_cv2", boundary_mode_str_cv2="Reflect_101_cv2"):
    """
    Warps an image using an optical flow field with OpenCV's remap function.
    image_np_rgba_uint8: Input image as a NumPy array (RGBA, uint8).
    flow_field_xy: Optical flow field (height, width, 2) for (dx, dy).
    interpolation_str_cv2_warp: OpenCV interpolation string for cv2.remap.
    boundary_mode_str_cv2: OpenCV boundary mode string for cv2.remap.
    Returns: Warped image as a NumPy array (RGBA, uint8).
    """
    cv2_interpolation_map_warp = {
        "Nearest_cv2": cv2.INTER_NEAREST, 
        "Linear_cv2": cv2.INTER_LINEAR, 
        "Cubic_cv2": cv2.INTER_CUBIC, 
        "Lanczos4_cv2": cv2.INTER_LANCZOS4
    }
    cv2_warp_interpolation = cv2_interpolation_map_warp.get(interpolation_str_cv2_warp, cv2.INTER_LINEAR)

    cv2_boundary_map = {
        "Constant_cv2": cv2.BORDER_CONSTANT,    # Adds a constant colored border
        "Replicate_cv2": cv2.BORDER_REPLICATE,  # Repeats the last pixel
        "Reflect_cv2": cv2.BORDER_REFLECT,      # Reflects the border (e.g., fedcba|abcdefgh|hgfedcb)
        "Wrap_cv2": cv2.BORDER_WRAP,            # Wraps around (e.g., cdefgh|abcdefgh|abcdefg)
        "Reflect_101_cv2": cv2.BORDER_REFLECT_101 # Default, reflects without repeating border pixel (e.g., gfedcb|abcdefgh|gfedcba)
    }
    cv2_warp_boundary_mode = cv2_boundary_map.get(boundary_mode_str_cv2, cv2.BORDER_REFLECT_101)

    height, width = image_np_rgba_uint8.shape[:2]
    
    # Create a grid of (x,y) coordinates corresponding to the pixels of the output image
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    
    # The flow_field_xy (dx, dy) tells us, for each pixel (x,y) in the *original* image (img1),
    # where it moves to in the *next* image (img2).
    # So, img2(x+dx, y+dy) = img1(x,y).
    # For cv2.remap, we need map_x and map_y such that:
    # warped_image(x_out, y_out) = original_image(map_x(x_out, y_out), map_y(x_out, y_out))
    # If we want to apply a forward flow (img1 -> img1_warped_by_flow),
    # the map should define, for each pixel in the *output* warped image,
    # which pixel from the *input* original image it should take its color from.
    # map_x(x,y) = x - flow_x(x,y)
    # map_y(x,y) = y - flow_y(x,y)
    # This means: "To get the color for pixel (x,y) in the warped image, look at pixel
    # (x - dx, y - dy) in the original image."
    
    map_x = grid_x - flow_field_xy[...,0] # Subtract dx from the x-coordinates
    map_y = grid_y - flow_field_xy[...,1] # Subtract dy from the y-coordinates

    # Ensure map_x and map_y are float32 for OpenCV remap
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # Perform the remapping (warping)
    # Note: cv2.remap expects map_x and map_y to be of type CV_32FC1 or CV_32FC2.
    # If flow_field_xy has shape (h, w, 2), then map_x and map_y will be (h,w)
    # and they can be passed directly.
    
    # Ensure image_np_rgba_uint8 is contiguous if issues arise, though usually not needed for remap
    # image_np_rgba_uint8 = np.ascontiguousarray(image_np_rgba_uint8)

    warped_image_np_rgba_uint8 = cv2.remap(
        image_np_rgba_uint8, 
        map_x,  # Map for x-coordinates
        map_y,  # Map for y-coordinates
        interpolation=cv2_warp_interpolation, 
        borderMode=cv2_warp_boundary_mode,
        borderValue=(0,0,0,0) # Value for BORDER_CONSTANT, (B,G,R,A) - alpha 0 for transparent
    )
    
    return warped_image_np_rgba_uint8


def generate_flow_target_frame_zoomer(pil_input_image_rgb, mode="Zoom_In_Flow_Target", amount=0.05, zoom_interpolation_pil_str="Lanczos"):
    """
    Generates a target frame for optical flow calculation, typically by zooming the input image.
    pil_input_image_rgb: The base PIL Image (RGB) to transform.
    mode: "Zoom_In_Flow_Target" or "Zoom_Out_Flow_Target".
    amount: The fractional amount of zoom (e.g., 0.05 for 5% zoom).
    zoom_interpolation_pil_str: PIL resampling method string ("Nearest", "Bilinear", "Bicubic", "Lanczos").
    Returns: A new PIL Image (RGB) representing the transformed target.
    """
    if mode == "Zoom_In_Flow_Target":
        scale_factor = 1.0 + amount
    elif mode == "Zoom_Out_Flow_Target":
        scale_factor = max(0.1, 1.0 - amount) # Ensure scale doesn't go to or below zero
    else: # Default or unrecognized mode
        print(f"FlowUtils: Unrecognized flow target mode '{mode}'. Defaulting to slight zoom in.")
        scale_factor = 1.0 + 0.02 # Default to a minimal zoom in

    # Use the imported (or mocked) apply_centered_zoom_pil_zoomer from image_utils
    # This function is expected to handle the actual zoom and resampling.
    if 'apply_centered_zoom_pil_zoomer' in globals():
        return apply_centered_zoom_pil_zoomer(pil_input_image_rgb, scale_factor, zoom_interpolation_pil_str)
    else:
        # This part should ideally not be reached if the mock is defined correctly at module level
        print("FlowUtils: Critical - apply_centered_zoom_pil_zoomer is not available.")
        # Fallback to a simple, non-centered resize if the proper function is missing.
        width, height = pil_input_image_rgb.size
        new_width, new_height = int(width * scale_factor), int(height * scale_factor)
        resample_map = {
            "Nearest": Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST,
            "Bilinear": Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR,
            "Bicubic": Image.Resampling.BICUBIC if hasattr(Image, 'Resampling') else Image.BICUBIC,
            "Lanczos": Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS,
        }
        resample_method = resample_map.get(zoom_interpolation_pil_str, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
        
        resized_img = pil_input_image_rgb.resize((new_width, new_height), resample=resample_method)
        # If resized smaller, paste onto original size canvas. If larger, crop.
        # This is a crude centering for fallback.
        final_img = Image.new("RGB", (width, height))
        if scale_factor < 1.0: # Pasting smaller image onto center
            paste_x = (width - new_width) // 2
            paste_y = (height - new_height) // 2
            final_img.paste(resized_img, (paste_x, paste_y))
        else: # Cropping larger image from center
            crop_x = (new_width - width) // 2
            crop_y = (new_height - height) // 2
            final_img = resized_img.crop((crop_x, crop_y, crop_x + width, crop_y + height))
        return final_img