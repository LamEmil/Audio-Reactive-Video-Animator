from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
from PIL import Image
import os
import imageio # For reading video input
import cv2 # For writing video output
import tempfile # For temporary file handling
import shutil
import time
import subprocess # For ffmpeg

# Attempt to import project modules
try:
    from logic import OpticalFlowZoomerLogic
    from audio_utils import LIBROSA_AVAILABLE, LIBROSA_ERROR_MESSAGE, MOVIEPY_ERROR_MESSAGE_INIT
except ModuleNotFoundError:
    print("Worker: Critical - Could not import 'logic' or 'audio_utils'. Ensure they are in the Python path.")
    # Provide mock values so the class can be defined, but it will fail at runtime
    class OpticalFlowZoomerLogic: pass # Mock class
    LIBROSA_AVAILABLE = False
    LIBROSA_ERROR_MESSAGE = "CRITICAL: 'audio_utils.py' or 'logic.py' missing, cannot proceed."
    MOVIEPY_ERROR_MESSAGE_INIT = "" # Assuming if audio_utils is missing, moviepy status is unknown


class OpticalFlowZoomerWorker(QThread):
    progress_update = pyqtSignal(str, int)
    finished = pyqtSignal(str, bool, str) # video_path, success, message

    def __init__(self, image_path, audio_path, output_video_path, params, ffmpeg_path=None):
        super().__init__()
        self.image_path = image_path
        self.audio_path = audio_path
        self.output_video_path = output_video_path
        self.params = params
        self.zoomer_logic = OpticalFlowZoomerLogic() # Instantiated here
        self.ffmpeg_custom_path = ffmpeg_path if ffmpeg_path else "ffmpeg"


    def run(self):
        # Access global LIBROSA_AVAILABLE and messages if audio_utils was loaded,
        # otherwise they are the mock values from the try-except block above.
        # No need to declare them global here unless we are re-assigning them.

        # MoviePy is an optional dependency, check for it here.
        local_moviepy_editor = None
        local_MOVIEPY_AVAILABLE = False
        # MOVIEPY_ERROR_MESSAGE_INIT should come from audio_utils, which might also check moviepy.
        # For this worker, we can just try to import it.
        current_moviepy_error_message = MOVIEPY_ERROR_MESSAGE_INIT # From audio_utils (potentially)

        # We don't use moviepy in this version of the worker for ffmpeg calls, using subprocess directly

        # Check for critical missing modules (audio_utils, logic)
        if not hasattr(self.zoomer_logic, 'process_animation'): # Check if it's the real logic class
             self.finished.emit("", False, "Critical Error: 'logic.py' module is not correctly loaded.")
             return
        if "audio_utils.py' missing" in LIBROSA_ERROR_MESSAGE and \
           (self.params.get("enable_breathing_effects", False) or self.params.get("use_original_peak_effects", True)):
            self.finished.emit("", False, LIBROSA_ERROR_MESSAGE) # Propagate the "missing audio_utils" error
            return
        # If audio effects are enabled but Librosa specifically (within audio_utils) is not available
        if not LIBROSA_AVAILABLE and \
           (self.params.get("enable_breathing_effects", False) or self.params.get("use_original_peak_effects", True)):
            self.finished.emit("", False, LIBROSA_ERROR_MESSAGE) # Propagate the "Librosa not available" error
            return


        temp_silent_video_path = None # Renamed from temp_video_file_path to avoid shadowing
        video_writer = None # cv2.VideoWriter object

        try:
            self.progress_update.emit("Loading image or video input...", 0)
            if not self.image_path or not os.path.exists(self.image_path):
                self.finished.emit("", False, "Input image/video file not found or not specified.")
                return

            input_ext = os.path.splitext(self.image_path)[1].lower()
            video_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"]
            is_video_input = input_ext in video_exts

            initial_pil_image_rgba = None # This will be the first frame (or the image itself)
            video_frames = [] # List to hold all frames

            if is_video_input:
                self.progress_update.emit("Reading video input...", 2)
                try:
                    vid_reader = imageio.get_reader(self.image_path)
                    for frame_np in vid_reader:
                        pil_frame = Image.fromarray(frame_np).convert("RGBA")
                        video_frames.append(pil_frame)
                    vid_reader.close()
                except Exception as e:
                    self.finished.emit("", False, f"Error reading video input: {e}")
                    return
                if not video_frames:
                    self.finished.emit("", False, "Could not extract frames from video input.")
                    return
                initial_pil_image_rgba = video_frames[0]
            else: # Static image input
                try:
                    initial_pil_image_rgba = Image.open(self.image_path).convert("RGBA")
                    video_frames = [initial_pil_image_rgba]
                except Exception as e:
                    self.finished.emit("", False, f"Error opening image input: {e}")
                    return
            
            self.progress_update.emit("Image loaded. Preparing parameters...", 5)

            # Handle audio path for logic processor
            if not self.audio_path or not os.path.exists(self.audio_path):
                if self.params.get("enable_breathing_effects", False) or self.params.get("use_original_peak_effects", True):
                    # Audio is required for these effects
                    self.finished.emit("", False, "Audio file not found/specified, but audio-reactive effects are enabled.")
                    return
                else: # No audio effects, or audio is optional
                    self.params["audio_filepath"] = None # Signal to logic that there's no audio
                    self.progress_update.emit("No audio file, proceeding without audio-reactive effects.", 6)
            else:
                self.params["audio_filepath"] = self.audio_path # Pass audio path to logic

            # Call the animation logic
            # The logic module is responsible for handling cases where audio_filepath is None
            output_pil_frames_rgba = self.zoomer_logic.process_animation(
                video_frames, self.params, self.progress_update # Pass list of frames
            )

            if not output_pil_frames_rgba:
                self.finished.emit("", False, "Animation logic did not produce any frames.")
                return

            # Create a temporary file for the silent video
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
                temp_silent_video_path = tmpfile.name
            
            self.progress_update.emit("Writing silent video frames...", 90)
            
            first_frame_pil = output_pil_frames_rgba[0].convert("RGB") # OpenCV needs RGB/BGR
            height, width = np.array(first_frame_pil).shape[:2] # Get dimensions from first frame
            
            user_fps = self.params.get("fps", 24) # Get FPS from params
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
            video_writer = cv2.VideoWriter(temp_silent_video_path, fourcc, user_fps, (width, height))

            if not video_writer.isOpened():
                self.finished.emit("", False, f"Error: Could not open temporary video writer for '{temp_silent_video_path}'")
                return

            for i, frame_pil_rgba in enumerate(output_pil_frames_rgba):
                frame_rgb_np = np.array(frame_pil_rgba.convert("RGB"))
                frame_bgr_np = cv2.cvtColor(frame_rgb_np, cv2.COLOR_RGB2BGR) # OpenCV uses BGR
                video_writer.write(frame_bgr_np)

            video_writer.release()
            video_writer = None # Ensure it's released
            self.progress_update.emit("Silent video written.", 95)

            output_final_video_path = self.output_video_path # Where the final video will be saved

            def ffmpeg_encode_video(input_video, output_video, audio_path=None, ffmpeg_path="ffmpeg"):
                cmd = [
                    ffmpeg_path,
                    "-y",
                    "-i", input_video,
                ]
                if audio_path:
                    cmd += ["-i", audio_path]
                cmd += [
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "veryfast",
                    "-crf", "23",
                ]
                if audio_path:
                    cmd += [
                        "-c:a", "aac",
                        "-b:a", "192k",
                        "-shortest",
                        "-map", "0:v:0",
                        "-map", "1:a:0",
                    ]
                else:
                    cmd += ["-an"] # No audio
                cmd += [
                    "-movflags", "+faststart",
                    "-loglevel", "error",
                    output_video
                ]
                try:
                    process = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    return True, ""
                except subprocess.CalledProcessError as ffmpeg_err:
                    return False, f"FFmpeg encode failed (code {ffmpeg_err.returncode}). Stderr: {ffmpeg_err.stderr}"
                except FileNotFoundError:
                    return False, "ffmpeg executable not found for encoding."

            if self.params.get("audio_filepath") and os.path.exists(self.params["audio_filepath"]):
                audio_input_path = self.params["audio_filepath"]
                self.progress_update.emit("Muxing and encoding video/audio with ffmpeg...", 98)
                output_dir = os.path.dirname(output_final_video_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                success, ffmpeg_err = ffmpeg_encode_video(
                    temp_silent_video_path, output_final_video_path, audio_input_path, self.ffmpeg_custom_path
                )
                if success:
                    self.progress_update.emit("Audio added. Video web-optimized!", 100)
                    self.finished.emit(output_final_video_path, True, "")
                else:
                    shutil.copy(temp_silent_video_path, output_final_video_path)
                    self.progress_update.emit("FFmpeg mux/encode failed. Silent video saved.", 100)
                    self.finished.emit(output_final_video_path, True, f"WARNING: Audio could not be added or video encoding failed. Video is silent or non-standard.\nDetails: {ffmpeg_err[:250]}...")
            else:
                self.progress_update.emit("Encoding video for web compatibility...", 98)
                output_dir = os.path.dirname(output_final_video_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                success, ffmpeg_err = ffmpeg_encode_video(
                    temp_silent_video_path, output_final_video_path, None, self.ffmpeg_custom_path
                )
                if success:
                    self.progress_update.emit("Video (no audio) web-optimized!", 100)
                    self.finished.emit(output_final_video_path, True, "Video generated (no audio was provided or processed).")
                else:
                    shutil.copy(temp_silent_video_path, output_final_video_path)
                    self.progress_update.emit("FFmpeg encode failed. Silent video saved.", 100)
                    self.finished.emit(output_final_video_path, True, f"WARNING: Video encoding failed. Video may be non-standard.\nDetails: {ffmpeg_err[:250]}...")

        except ImportError as ie: # Catch import errors for modules not found during runtime
            import traceback
            error_msg = f"ImportError during processing: {ie}\nThis usually means a required .py file (like audio_utils or image_utils) is missing or not in the Python path.\n{LIBROSA_ERROR_MESSAGE}\nTraceback: {traceback.format_exc()}"
            self.progress_update.emit(f"Import Error: {ie}", 0)
            self.finished.emit("", False, error_msg)
        except Exception as e:
            import traceback
            error_msg = f"General error during video processing: {e}\nTraceback: {traceback.format_exc()}"
            self.progress_update.emit(f"Error: {e}", 0)
            # Try to save silent fallback if it exists
            if temp_silent_video_path and os.path.exists(temp_silent_video_path):
                try:
                    fallback_path = self.output_video_path # User's chosen output path
                    shutil.copy(temp_silent_video_path, fallback_path)
                    self.finished.emit(fallback_path, False, f"{error_msg}\n\nA silent version of the video may have been saved to the output path as a fallback.")
                except Exception as copy_err:
                    self.finished.emit("", False, f"{error_msg}\n\nFailed to save silent fallback video: {copy_err}")
            else:
                self.finished.emit("", False, error_msg)
        finally:
            if video_writer is not None and video_writer.isOpened():
                video_writer.release() # Ensure writer is closed
            
            # Clean up the temporary silent video file
            if temp_silent_video_path and os.path.exists(temp_silent_video_path):
                try:
                    # Add a small delay before removing, sometimes helpful on Windows
                    time.sleep(0.5) 
                    os.remove(temp_silent_video_path)
                except PermissionError as e_perm:
                     print(f"Warning: Could not delete temporary video file {temp_silent_video_path} due to PermissionError: {e_perm}. It might still be in use.")
                except Exception as e_del:
                    print(f"Warning: Could not delete temporary video file {temp_silent_video_path}: {e_del}")