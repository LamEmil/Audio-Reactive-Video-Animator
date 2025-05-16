import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import random
import os

# Attempt to import project-specific utility modules
try:
    from audio_utils import (
        analyze_audio_for_peaks_zoomer,
        analyze_audio_for_breathing_envelope,
        get_peak_envelope_multiplier_zoomer,
        # Ensure LIBROSA_AVAILABLE is accessible if needed here, or handled by the caller
    )
except ModuleNotFoundError:
    print("Logic: Critical - 'audio_utils.py' not found. Audio processing will fail.")
    # Define mock functions if audio_utils is missing, so the class can load
    # but functionality will be severely limited.
    def analyze_audio_for_peaks_zoomer(*args, **kwargs): return np.array([False]), 1
    def analyze_audio_for_breathing_envelope(*args, **kwargs): return np.array([0.0]), 1
    def get_peak_envelope_multiplier_zoomer(*args, **kwargs): return 0.0
    # LIBROSA_AVAILABLE = False # This should be managed by audio_utils itself

try:
    from image_utils import (
        apply_centered_zoom_pil_zoomer, apply_breathing_zoom, apply_breathing_brightness,
        apply_breathing_blur, apply_breathing_saturation, apply_breathing_color_shift
        # Ensure these functions are correctly defined in image_utils.py
    )
except ModuleNotFoundError:
    print("Logic: Critical - 'image_utils.py' not found. Image processing will fail.")
    # Define mock functions if image_utils is missing
    def apply_centered_zoom_pil_zoomer(img, *args): return img
    def apply_breathing_zoom(img, *args): return img
    def apply_breathing_brightness(img, *args): return img
    def apply_breathing_blur(img, *args): return img
    def apply_breathing_saturation(img, *args): return img
    def apply_breathing_color_shift(img, *args): return img


try:
    from flow_utils import (
        calculate_optical_flow_optimized_zoomer, warp_image_cv2_zoomer, generate_flow_target_frame_zoomer
    )
except ModuleNotFoundError:
    print("Logic: Critical - 'flow_utils.py' not found. Optical flow effects will fail.")
    # Define mock functions if flow_utils is missing
    def calculate_optical_flow_optimized_zoomer(*args, **kwargs): return None
    def warp_image_cv2_zoomer(img_np, *args): return img_np # Assuming img_np is a numpy array
    def generate_flow_target_frame_zoomer(img, *args): return img


class OpticalFlowZoomerLogic:
    def process_animation(self, input_frames_rgba, params, update_progress_signal):
        # input_frames_rgba: list of PIL RGBA images (for video) or [single image] for static
        # --- Essential Parameters ---
        audio_filepath = params.get("audio_filepath") # Can be None
        fps = params.get("fps", 24)
        override_num_frames = params.get("override_num_frames", 0)
        seed = params.get("seed", -1)

        # --- Parameters for Original Peak/Optical Flow Effects ---
        use_original_peak_effects = params.get("use_original_peak_effects", True)
        peak_threshold_multiplier = params.get("peak_threshold_multiplier", 0.8)
        peak_hold_frames = params.get("peak_hold_frames", 12) # Duration of a peak's influence
        alternate_every_n_peaks = params.get("alternate_every_n_peaks", 1)
        # Envelope for peak effect strength
        peak_attack_ratio = params.get("peak_attack_ratio", 0.2)
        peak_sustain_ratio = params.get("peak_sustain_ratio", 0.5)
        peak_decay_ratio = params.get("peak_decay_ratio", 0.3) # Ensure attack+sustain+decay <= 1.0 if they are proportions of peak_hold_frames
        peak_easing_type = params.get("peak_easing_type", "sine")
        # Optical flow target generation
        flow_target_generation_mode = params.get("flow_target_generation_mode", "Zoom_In_Flow_Target")
        flow_target_transform_amount = params.get("flow_target_transform_amount", 0.15) # e.g., 15% zoom for target
        flow_calc_scale_factor = params.get("flow_calc_scale_factor", 0.5) # Downscale for flow calculation
        # Flow strength application
        idle_flow_strength = params.get("idle_flow_strength", 0.0) # Base flow when no peak
        peak_flow_strength_zoom_in = params.get("peak_flow_strength_zoom_in", 0.4)
        peak_flow_strength_zoom_out = params.get("peak_flow_strength_zoom_out", -0.3)
        # Interpolation/Boundary for warping
        warp_interpolation_cv2_str = params.get("warp_interpolation_cv2", "Linear_cv2")
        zoom_interpolation_pil_str = params.get("zoom_interpolation_pil", "Lanczos") # For generating flow target & breathing zoom
        boundary_mode_cv2_str = params.get("boundary_mode_cv2", "Reflect_101_cv2")

        # --- Parameters for Breathing Effects ---
        enable_breathing_effects = params.get("enable_breathing_effects", False)
        breathing_smoothing_window = params.get("breathing_smoothing_window", 0.5) # seconds
        breathing_target_percentile = params.get("breathing_target_percentile", 75)
        
        enable_breathing_zoom = params.get("enable_breathing_zoom", False)
        breathing_max_zoom_factor = params.get("breathing_max_zoom_factor", 0.05) # e.g., 5% max additional zoom

        enable_breathing_brightness = params.get("enable_breathing_brightness", False)
        breathing_min_brightness = params.get("breathing_min_brightness", 0.9)
        breathing_max_brightness = params.get("breathing_max_brightness", 1.1)

        enable_breathing_blur = params.get("enable_breathing_blur", False)
        breathing_max_blur_radius = params.get("breathing_max_blur_radius", 1.0) # pixels

        enable_breathing_saturation = params.get("enable_breathing_saturation", False)
        breathing_min_saturation = params.get("breathing_min_saturation", 0.8)
        breathing_max_saturation = params.get("breathing_max_saturation", 1.2)

        enable_breathing_color_shift = params.get("enable_breathing_color_shift", False)
        breathing_min_hue_shift = params.get("breathing_min_hue_shift", 0.0) # 0.0 to 1.0
        breathing_max_hue_shift = params.get("breathing_max_hue_shift", 0.05) # e.g., 5% of hue circle

        # Accept first frame for static image, or all frames for video
        if isinstance(input_frames_rgba, list) and len(input_frames_rgba) > 1:
            is_video = True
            pil_initial_image_rgba = input_frames_rgba[0]
            input_frames_list = input_frames_rgba
        else:
            is_video = False
            pil_initial_image_rgba = input_frames_rgba[0] if isinstance(input_frames_rgba, list) else input_frames_rgba
            input_frames_list = [pil_initial_image_rgba]

        if seed >= 0:
            random.seed(seed)
            np.random.seed(seed)

        # --- Audio Analysis ---
        # These will hold audio analysis results, or defaults if no audio/effects
        breathing_envelope_per_frame = np.array([0.0]) # Default: no breath effect
        is_peak_trigger_per_frame = np.array([False]) # Default: no peak triggers
        
        # Calculate total frames based on audio duration if available, else use a default or rely on override
        audio_derived_total_frames = 1 # Default if no audio analysis performed

        if audio_filepath and os.path.exists(audio_filepath):
            # Check if any audio-dependent effect is enabled
            if enable_breathing_effects:
                update_progress_signal.emit("Analyzing audio for breathing envelope...", 3)
                try:
                    breathing_envelope_per_frame, audio_derived_total_frames_breath = analyze_audio_for_breathing_envelope(
                        audio_filepath, fps, breathing_smoothing_window, breathing_target_percentile
                    )
                    audio_derived_total_frames = max(audio_derived_total_frames, audio_derived_total_frames_breath)
                except Exception as e:
                    update_progress_signal.emit(f"Breath audio analysis failed: {e}", 3)
                    # Fallback or re-raise, for now, we'll continue with default envelope

            if use_original_peak_effects:
                update_progress_signal.emit("Analyzing audio for peaks (for flow)...", 5)
                try:
                    is_peak_trigger_per_frame, audio_derived_total_frames_peak = analyze_audio_for_peaks_zoomer(
                        audio_filepath, fps, peak_threshold_multiplier, peak_hold_frames
                    )
                    audio_derived_total_frames = max(audio_derived_total_frames, audio_derived_total_frames_peak)
                except Exception as e:
                    update_progress_signal.emit(f"Peak audio analysis failed: {e}", 5)
                    # Fallback or re-raise
        else: # No audio file, or audio effects disabled that need it
            if enable_breathing_effects or use_original_peak_effects:
                 update_progress_signal.emit("Warning: Audio effects enabled but no audio file. Effects may be static.", 2)


        # Determine final number of frames for the video
        num_total_video_frames = len(input_frames_list) if is_video else audio_derived_total_frames
        if override_num_frames > 0:
            num_total_video_frames = override_num_frames
        
        if num_total_video_frames <= 0: # Ensure there's at least one frame
            num_total_video_frames = 1 
            update_progress_signal.emit("Warning: Frame count is zero or less. Defaulting to 1 frame.", 0)


        # Ensure audio analysis arrays match the number of video frames
        if enable_breathing_effects:
            if len(breathing_envelope_per_frame) < num_total_video_frames:
                breathing_envelope_per_frame = np.pad(breathing_envelope_per_frame, (0, num_total_video_frames - len(breathing_envelope_per_frame)), 'edge')
            elif len(breathing_envelope_per_frame) > num_total_video_frames:
                breathing_envelope_per_frame = breathing_envelope_per_frame[:num_total_video_frames]
        
        if use_original_peak_effects:
            if len(is_peak_trigger_per_frame) < num_total_video_frames:
                # Pad with False (no trigger) or 'wrap' if cyclic behavior is desired for short audio
                is_peak_trigger_per_frame = np.pad(is_peak_trigger_per_frame, (0, num_total_video_frames - len(is_peak_trigger_per_frame)), 'constant', constant_values=False)
            elif len(is_peak_trigger_per_frame) > num_total_video_frames:
                is_peak_trigger_per_frame = is_peak_trigger_per_frame[:num_total_video_frames]


        # --- Base Image and Optical Flow Preparation ---
        pil_initial_rgb = pil_initial_image_rgba.convert('RGB') # For effects not needing alpha yet
        
        base_flow_field_xy = None # Will hold the (dx, dy) flow field
        if use_original_peak_effects:
            update_progress_signal.emit("Calculating base optical flow (if enabled)...", 15)
            try:
                # Generate a target frame (e.g., slightly zoomed version of initial frame)
                pil_flow_target_frame_rgb = generate_flow_target_frame_zoomer(
                    pil_initial_rgb, 
                    flow_target_generation_mode,
                    flow_target_transform_amount, 
                    zoom_interpolation_pil_str
                )
                # Calculate optical flow between initial and target frame
                base_flow_field_xy = calculate_optical_flow_optimized_zoomer(
                    pil_initial_rgb, 
                    pil_flow_target_frame_rgb, 
                    flow_calc_scale_factor, 
                    warp_interpolation_cv2_str # Interpolation used for resizing during flow calculation
                )
                update_progress_signal.emit("Base optical flow calculated.", 25)
            except Exception as e:
                 update_progress_signal.emit(f"Optical flow calculation failed: {e}", 15)
                 use_original_peak_effects = False # Disable if calculation fails


        # --- Frame Generation Loop ---
        output_pil_frames_rgba = []
        
        # State for peak-based flow direction alternation
        current_phase_is_zoom_in = True # Start with zoom-in direction for flow
        actual_peak_trigger_count = 0   # How many peaks have actually triggered an effect
        
        # State for peak effect envelope
        in_peak_effect_phase = False              # True if currently in attack/sustain/decay of a peak
        peak_effect_frames_elapsed_in_phase = 0   # Frames since current peak effect started
        current_peak_target_strength_for_phase = 0.0 # Max strength for current peak phase

        for frame_num in range(num_total_video_frames):
            progress_percentage = 25 + int(70 * frame_num / num_total_video_frames) if num_total_video_frames > 0 else 25
            update_progress_signal.emit(f"Generating frame {frame_num+1}/{num_total_video_frames}", progress_percentage)

            # For video: use the corresponding frame; for image: use initial
            if is_video:
                current_input_pil_rgba = input_frames_list[frame_num]
                current_frame_pil_rgb = current_input_pil_rgba.convert('RGB')
            else:
                current_frame_pil_rgb = pil_initial_rgb.copy()

            # 1. Apply Breathing Effects (modifies current_frame_pil_rgb)
            if enable_breathing_effects and frame_num < len(breathing_envelope_per_frame):
                breath_intensity = breathing_envelope_per_frame[frame_num] # Normalized 0-1

                if enable_breathing_zoom and 'apply_breathing_zoom' in globals():
                    current_frame_pil_rgb = apply_breathing_zoom(current_frame_pil_rgb, 1.0, breath_intensity, breathing_max_zoom_factor, zoom_interpolation_pil_str)
                if enable_breathing_brightness and 'apply_breathing_brightness' in globals():
                    current_frame_pil_rgb = apply_breathing_brightness(current_frame_pil_rgb, breath_intensity, breathing_min_brightness, breathing_max_brightness)
                if enable_breathing_blur and 'apply_breathing_blur' in globals():
                    current_frame_pil_rgb = apply_breathing_blur(current_frame_pil_rgb, breath_intensity, breathing_max_blur_radius)
                if enable_breathing_saturation and 'apply_breathing_saturation' in globals():
                    current_frame_pil_rgb = apply_breathing_saturation(current_frame_pil_rgb, breath_intensity, breathing_min_saturation, breathing_max_saturation)
                if enable_breathing_color_shift and 'apply_breathing_color_shift' in globals():
                    current_frame_pil_rgb = apply_breathing_color_shift(current_frame_pil_rgb, breath_intensity, breathing_min_hue_shift, breathing_max_hue_shift)
            
            # Convert current frame to RGBA before optical flow, as flow might use alpha or it's needed for output
            # The image subject to optical flow will be the one potentially modified by breathing effects.
            canvas_to_warp_pil_rgba = current_frame_pil_rgb.convert("RGBA")


            # 2. Apply Peak-Based Optical Flow Effects (modifies canvas_to_warp_pil_rgba)
            final_frame_pil_rgba = canvas_to_warp_pil_rgba # Default if no flow applied

            if use_original_peak_effects and base_flow_field_xy is not None:
                applied_strength_for_this_iteration = idle_flow_strength
                
                is_new_peak_trigger_this_frame = is_peak_trigger_per_frame[frame_num] if frame_num < len(is_peak_trigger_per_frame) else False
                
                if is_new_peak_trigger_this_frame and not in_peak_effect_phase: # New peak starts
                    in_peak_effect_phase = True
                    peak_effect_frames_elapsed_in_phase = 0
                    actual_peak_trigger_count += 1
                    
                    # Determine direction for this new peak phase
                    if alternate_every_n_peaks > 0 and (actual_peak_trigger_count -1) % alternate_every_n_peaks == 0:
                        if actual_peak_trigger_count > 1 : # Don't flip on the very first peak unless intended
                             current_phase_is_zoom_in = not current_phase_is_zoom_in
                    current_peak_target_strength_for_phase = peak_flow_strength_zoom_in if current_phase_is_zoom_in else peak_flow_strength_zoom_out
                
                if in_peak_effect_phase:
                    peak_effect_frames_elapsed_in_phase += 1
                    # Get envelope multiplier (0-1) based on how far into the peak effect we are
                    envelope_multiplier = get_peak_envelope_multiplier_zoomer(
                        peak_effect_frames_elapsed_in_phase, peak_hold_frames,
                        peak_attack_ratio, peak_sustain_ratio, peak_decay_ratio, peak_easing_type
                    )
                    applied_strength_for_this_iteration = current_peak_target_strength_for_phase * envelope_multiplier
                    
                    # Check if peak effect duration has ended
                    if peak_effect_frames_elapsed_in_phase >= peak_hold_frames:
                        in_peak_effect_phase = False 
                        # Optionally, could decay back to idle_flow_strength over a few frames here
                
                # Apply the optical flow warp if strength is significant
                if abs(applied_strength_for_this_iteration) > 1e-6: # Small threshold to avoid unnecessary computation
                    # Convert PIL RGBA to NumPy array (uint8) for OpenCV warping
                    canvas_to_warp_np_rgba_uint8 = np.array(canvas_to_warp_pil_rgba, dtype=np.uint8)
                    
                    # Scale the base flow field by the current strength
                    current_iter_flow_xy = base_flow_field_xy * applied_strength_for_this_iteration
                    
                    # Warp the image
                    warped_canvas_np_rgba_uint8 = warp_image_cv2_zoomer(
                        canvas_to_warp_np_rgba_uint8, 
                        current_iter_flow_xy,
                        warp_interpolation_cv2_str, # Interpolation for cv2.remap
                        boundary_mode_cv2_str         # Boundary mode for cv2.remap
                    )
                    final_frame_pil_rgba = Image.fromarray(warped_canvas_np_rgba_uint8, 'RGBA')
                # else: final_frame_pil_rgba remains canvas_to_warp_pil_rgba (no significant flow)
            
            output_pil_frames_rgba.append(final_frame_pil_rgba)

        update_progress_signal.emit("Visual frame generation complete.", 95)
        return output_pil_frames_rgba