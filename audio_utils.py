# audio_utils.py
# Description: Utility functions for audio processing, primarily using Librosa.

import numpy as np
import math
import os 

# --- Librosa Import and Availability Check ---
LIBROSA_AVAILABLE = False
LIBROSA_ERROR_MESSAGE = ""
librosa = None 

MOVIEPY_ERROR_MESSAGE_INIT = "" 

try:
    import librosa as lr_module
    librosa = lr_module 
    LIBROSA_AVAILABLE = True
    print("audio_utils.py: Librosa imported successfully.")
except ImportError as e:
    LIBROSA_ERROR_MESSAGE = str(e)
    if "Numba needs NumPy" in LIBROSA_ERROR_MESSAGE or "numpy" in LIBROSA_ERROR_MESSAGE.lower():
        LIBROSA_ERROR_MESSAGE = (
            "CRITICAL ERROR: Librosa (audio processing library) failed to import.\n\n"
            f"Details: '{LIBROSA_ERROR_MESSAGE}'\n\n"
            "This is often due to a Numba/NumPy version incompatibility. \n"
            "Please resolve this by adjusting NumPy/Numba versions.\n\n"
            "Audio-dependent features will be disabled."
        )
    else:
        LIBROSA_ERROR_MESSAGE = (
            f"CRITICAL ERROR: Librosa (audio processing library) could not be imported.\n\nDetails: {e}\n\n"
            "The application's audio processing features will be disabled."
        )
    print(f"audio_utils.py: {LIBROSA_ERROR_MESSAGE}")
except Exception as e_gen: # Catch any other unexpected errors during librosa import
    LIBROSA_ERROR_MESSAGE = (
        f"CRITICAL ERROR: An unexpected error occurred while trying to import Librosa.\n\nDetails: {e_gen}\n\n"
        "The application's audio processing features will be disabled."
    )
    print(f"audio_utils.py: {LIBROSA_ERROR_MESSAGE}")


# --- Audio Analysis Functions ---

def analyze_audio_for_peaks_zoomer(audio_filepath, fps, threshold_multiplier, peak_hold_frames_for_trigger):
    """
    Analyzes audio to detect peaks based on onset strength.
    Args:
        audio_filepath (str): Path to the audio file.
        fps (int): Frames per second of the target video.
        threshold_multiplier (float): Multiplier for dynamic threshold calculation.
        peak_hold_frames_for_trigger (int): Duration a peak trigger should ideally last (used for fallback frame count).
    Returns:
        tuple: (is_peak_trigger_frame (np.ndarray, bool), total_video_frames (int))
               An array indicating if a frame is a peak trigger, and the total frames derived from audio.
    Raises:
        ImportError: If Librosa is not available.
        Exception: If audio loading or processing fails.
    """
    if not LIBROSA_AVAILABLE or librosa is None: # Check the global librosa variable
        raise ImportError("Librosa is not available. Cannot analyze audio for peaks.")
    try:
        # Load audio file
        y, sr = librosa.load(audio_filepath, sr=None, mono=True) # sr=None to preserve original sample rate

        # Calculate onset strength envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        # Normalize onset envelope
        if np.max(onset_env) > 0:
            onset_env_normalized = onset_env / np.max(onset_env)
        else: # Avoid division by zero if onset_env is all zeros
            onset_env_normalized = onset_env

        # Calculate a dynamic threshold for peak picking
        # This threshold adapts to the audio's characteristics
        dynamic_threshold = np.mean(onset_env_normalized) + threshold_multiplier * np.std(onset_env_normalized)
        dynamic_threshold = np.clip(dynamic_threshold, 0.05, 0.95) # Ensure threshold is within a reasonable range

        # Get time points corresponding to the onset envelope frames
        times = librosa.times_like(onset_env, sr=sr) # Using hop_length default of 512 for onset_strength

        # Pick peaks from the onset envelope
        # wait_param: min frames between peaks, derived from a small time window (e.g., 0.15s)
        # hop_length for onset_strength is typically 512 samples.
        # (sr / hop_length) gives frames per second of the onset_env.
        # wait_param_seconds = 0.15 # Minimum time in seconds between peaks
        # wait_param_frames_onset_env = int((sr / 512) * wait_param_seconds)
        wait_param_frames_onset_env = max(1, int( (sr / 512) * 0.15) ) # Ensure at least 1

        detected_peaks_indices = librosa.util.peak_pick(onset_env_normalized,
                                                        pre_max=3, post_max=3,    # Number of frames before/after to be maximal
                                                        pre_avg=3, post_avg=3,    # Number of frames before/after for averaging
                                                        delta=dynamic_threshold,   # Threshold above local average
                                                        wait=wait_param_frames_onset_env) # Min frames between peaks

        # Convert peak indices to times in seconds
        peak_times_seconds = times[detected_peaks_indices]

        # Determine total video frames based on audio duration
        audio_duration_seconds = librosa.get_duration(y=y, sr=sr)
        total_video_frames = math.ceil(audio_duration_seconds * fps)

        # If audio duration is zero but peaks were found (e.g. very short audio),
        # estimate duration from the last peak.
        if total_video_frames == 0 and len(peak_times_seconds) > 0:
            total_video_frames = math.ceil(np.max(peak_times_seconds) * fps) + peak_hold_frames_for_trigger
        
        if total_video_frames == 0: # Still zero, means no audio or no peaks in zero-duration audio
            print("audio_utils: Warning - Could not determine video frames from audio peaks. Defaulting to 1 frame.")
            return np.array([False]), 1 # Return a single frame, no peak

        # Create a boolean array indicating peak trigger frames for the video
        is_peak_trigger_frame = np.zeros(total_video_frames, dtype=bool)
        for peak_s in peak_times_seconds:
            start_frame = math.floor(peak_s * fps) # Frame where the peak occurs
            if 0 <= start_frame < total_video_frames:
                is_peak_trigger_frame[start_frame] = True
        
        return is_peak_trigger_frame, total_video_frames
    except Exception as e:
        print(f"audio_utils: Error loading or processing audio for peaks: {e}")
        # Re-raise the exception so the caller can handle it (e.g., show a message to the user)
        raise e


def analyze_audio_for_breathing_envelope(audio_filepath, fps, smoothing_window_seconds=0.5, target_percentile=75):
    """
    Analyzes audio to create a smoothed "breathing" envelope based on RMS energy.
    This envelope can be used for slow, continuous visual modulations.
    Args:
        audio_filepath (str): Path to the audio file.
        fps (int): Frames per second of the target video.
        smoothing_window_seconds (float): Duration of the smoothing window in seconds.
        target_percentile (int): Percentile of RMS energy used for normalization (1-100).
    Returns:
        tuple: (normalized_envelope (np.ndarray, float), total_video_frames (int))
               A normalized (0-1) envelope array, and total frames derived from audio.
    Raises:
        ImportError: If Librosa is not available.
        Exception: If audio loading or processing fails.
    """
    if not LIBROSA_AVAILABLE or librosa is None: # Check the global librosa variable
        raise ImportError("Librosa is not available. Cannot analyze audio for breathing envelope.")
    try:
        y, sr = librosa.load(audio_filepath, sr=None, mono=True)
        audio_duration_seconds = librosa.get_duration(y=y, sr=sr)
        total_video_frames = math.ceil(audio_duration_seconds * fps)

        if total_video_frames == 0: # Handle very short or empty audio
            return np.array([0.0]), 1 # Default to one frame, zero intensity

        # Calculate RMS energy
        # Use a standard RMS frame length and hop length for consistent analysis
        rms_frame_length_samples = 2048  # A common FFT/RMS frame size
        rms_hop_length_samples = 512     # A common hop size for RMS

        rms_energy = librosa.feature.rms(y=y, frame_length=rms_frame_length_samples, hop_length=rms_hop_length_samples)[0]
        
        # Get time points for the calculated RMS energy
        rms_times = librosa.times_like(rms_energy, sr=sr, hop_length=rms_hop_length_samples, n_fft=rms_frame_length_samples)
        
        # Create time points for each video frame
        video_frame_times = np.arange(total_video_frames) / fps

        # Interpolate RMS energy to match the video frame rate
        if len(rms_energy) > 1 and len(rms_times) > 1 : # Need at least 2 points to interpolate
             interp_rms_energy = np.interp(video_frame_times, rms_times, rms_energy)
        elif len(rms_energy) > 0: # Only one RMS value (e.g., very short audio), replicate it
            interp_rms_energy = np.full(total_video_frames, rms_energy[0])
        else: # No RMS energy calculated (e.g., silent audio shorter than frame_length)
            interp_rms_energy = np.zeros(total_video_frames)


        # Smooth the interpolated RMS energy using a moving average
        smoothing_window_frames = int(smoothing_window_seconds * fps)
        if smoothing_window_frames > 1 and len(interp_rms_energy) > smoothing_window_frames:
            # Use a simple moving average
            smoothed_energy = np.convolve(interp_rms_energy, np.ones(smoothing_window_frames)/smoothing_window_frames, mode='same')
        else: # No smoothing if window is too small or data is shorter than window
            smoothed_energy = interp_rms_energy

        # Normalize the smoothed energy to a 0-1 range
        # Using a percentile for the max value helps make the effect more consistent
        # across audios with different dynamic ranges.
        if np.max(smoothed_energy) > 1e-6: # Check for non-zero energy to avoid division by zero
            positive_energy_values = smoothed_energy[smoothed_energy > 1e-6] # Consider only positive values for percentile
            
            if len(positive_energy_values) > 0:
                 # Calculate the target value for normalization based on percentile
                 target_value_for_norm = np.percentile(positive_energy_values, target_percentile)
                 if target_value_for_norm < 1e-6: # Fallback if percentile is too low (e.g., mostly silence)
                     target_value_for_norm = np.max(smoothed_energy) # Use the absolute max in this case
            else: # All energy is effectively zero
                target_value_for_norm = np.max(smoothed_energy) if np.max(smoothed_energy) > 1e-6 else 1.0 # Default to 1.0 if max is also ~zero

            if target_value_for_norm < 1e-6 : target_value_for_norm = 1.0 # Final check to prevent division by zero

            normalized_envelope = smoothed_energy / target_value_for_norm
            # Allow some overshoot initially, then clip to ensure 0-1 range
            normalized_envelope = np.clip(normalized_envelope, 0, 1.5) 
            # Optional: Apply a power curve for response shaping (e.g., make it more reactive to louder parts)
            # normalized_envelope = np.power(normalized_envelope, 1.5) 
            normalized_envelope = np.clip(normalized_envelope, 0, 1.0) # Final clip to 0-1
        else: # If all energy is effectively zero
            normalized_envelope = np.zeros_like(smoothed_energy)

        return normalized_envelope, total_video_frames

    except Exception as e:
        print(f"audio_utils: Error loading or processing audio for breathing envelope: {e}")
        raise e


def get_peak_envelope_multiplier_zoomer(frames_elapsed_in_phase, peak_hold_frames,
                                 attack_ratio=0.25, sustain_ratio=0.5, decay_ratio=0.25,
                                 easing_type="sine"):
    """
    Calculates an envelope multiplier (0-1) for a peak effect over its duration.
    This creates an Attack-Sustain-Decay envelope.
    Args:
        frames_elapsed_in_phase (int): Current frame number within the peak effect's duration.
        peak_hold_frames (int): Total duration of the peak effect in frames.
        attack_ratio (float): Proportion of peak_hold_frames for the attack phase.
        sustain_ratio (float): Proportion of peak_hold_frames for the sustain phase.
        decay_ratio (float): Proportion of peak_hold_frames for the decay phase.
                               (Note: attack+sustain+decay should ideally sum to 1.0)
        easing_type (str): Type of easing ("linear", "sine", "quad_in", "quad_out").
    Returns:
        float: Envelope multiplier (0.0 to 1.0).
    """
    if peak_hold_frames <= 0: return 0.0 # No duration, no effect

    # Ensure ratios sum up to roughly 1.0 for clarity, though function handles them as proportions of total.
    # For robustness, we calculate absolute frame counts for each phase.
    
    # Clip frames_elapsed to be within the peak_hold_frames duration
    frames_elapsed_in_phase = np.clip(frames_elapsed_in_phase, 0, peak_hold_frames)

    # Calculate duration of each phase in frames
    attack_frames = peak_hold_frames * attack_ratio
    sustain_frames = peak_hold_frames * sustain_ratio
    # Decay_frames is implicitly the remainder, or peak_hold_frames * decay_ratio

    # Define end points of each phase
    attack_end_frame = attack_frames
    sustain_end_frame = attack_frames + sustain_frames
    # Decay phase ends at peak_hold_frames

    current_progress_within_total_hold = float(frames_elapsed_in_phase)

    multiplier = 0.0

    # Attack Phase
    if current_progress_within_total_hold <= attack_end_frame:
        if attack_frames == 0: # If no attack phase, jump to full or start of sustain/decay
            multiplier = 1.0 if (sustain_frames > 0 or (peak_hold_frames * decay_ratio) > 0) else 0.0
        else:
            # Progress within the attack phase (0 to 1)
            attack_phase_progress = current_progress_within_total_hold / attack_frames
            if easing_type == "sine": multiplier = math.sin(attack_phase_progress * math.pi / 2)
            elif easing_type == "quad_in": multiplier = attack_phase_progress**2
            elif easing_type == "quad_out": multiplier = 1 - (1 - attack_phase_progress)**2
            else: multiplier = attack_phase_progress # Linear

    # Sustain Phase
    elif current_progress_within_total_hold <= sustain_end_frame:
        multiplier = 1.0 # Full strength during sustain

    # Decay Phase
    else: # current_progress_within_total_hold > sustain_end_frame
        frames_into_decay_phase = current_progress_within_total_hold - sustain_end_frame
        total_decay_duration_frames = peak_hold_frames - sustain_end_frame # Remaining frames for decay
        
        if total_decay_duration_frames <= 0: # No decay phase defined or already past
            multiplier = 0.0
        else:
            # Progress within the decay phase (0 to 1)
            decay_phase_progress = frames_into_decay_phase / total_decay_duration_frames
            decay_phase_progress = np.clip(decay_phase_progress, 0.0, 1.0) # Ensure it's within 0-1

            if easing_type == "sine": multiplier = math.cos(decay_phase_progress * math.pi / 2) # Decays from 1 to 0
            elif easing_type == "quad_in": multiplier = (1.0 - decay_phase_progress)**2 # (1-x)^2, decays from 1 to 0
            elif easing_type == "quad_out": multiplier = 1.0 - decay_phase_progress**2 # 1 - x^2, decays from 1 to 0
            else: multiplier = 1.0 - decay_phase_progress # Linear decay from 1 to 0
            
    return np.clip(multiplier, 0.0, 1.0)
# --- MoviePy Import and Availability Check ---
try:
    import moviepy.editor 
except ImportError as e:
    MOVIEPY_ERROR_MESSAGE_INIT = (
        f"WARNING: moviepy library not found. Output video may be silent if FFMPEG is not configured.\n\nDetails: {e}\n\n"
        "To include audio, please install moviepy (`pip install moviepy`) and ensure ffmpeg is installed and in your system PATH, or provide the FFMPEG path in the UI."
    )
    # print(f"audio_utils.py: {MOVIEPY_ERROR_MESSAGE_INIT}") # Optional print
except Exception as e_gen: # Catch any other unexpected errors
     MOVIEPY_ERROR_MESSAGE_INIT = (
        f"WARNING: An unexpected error occurred during initial moviepy check in audio_utils.\n\nDetails: {e_gen}\n\n"
        "Output video may be silent if FFMPEG is not configured."
    )
    # print(f"audio_utils.py: {MOVIEPY_ERROR_MESSAGE_INIT}") # Optional print

