import sys
import os
import time
import imageio
from PIL import Image
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QProgressBar, QTabWidget, QGroupBox, QGridLayout, QMessageBox,
    QSizePolicy, QScrollArea
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget

try:
    from worker import OpticalFlowZoomerWorker
    from audio_utils import LIBROSA_AVAILABLE, LIBROSA_ERROR_MESSAGE, MOVIEPY_ERROR_MESSAGE_INIT
except ModuleNotFoundError:
    # This block is a fallback if modules are not directly found.
    # It's better if all .py files are in the same directory or properly installed.
    print("Warning: Could not import worker or audio_utils directly. Attempting fallback.")
    # Ensure the current directory is in sys.path for the imports below
    # This might be needed if running from an IDE with a different working directory
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    if current_script_path not in sys.path:
        sys.path.append(current_script_path)

    try:
        from worker import OpticalFlowZoomerWorker
    except ModuleNotFoundError:
        QMessageBox.critical(None, "Critical Error", "Failed to load 'worker.py'. Ensure it's in the same directory.")
        sys.exit(1) # Or handle more gracefully

    try:
        from audio_utils import LIBROSA_AVAILABLE, LIBROSA_ERROR_MESSAGE, MOVIEPY_ERROR_MESSAGE_INIT
    except ModuleNotFoundError:
        # Mock these if audio_utils is missing, so the GUI can at least start
        # to show the error about Librosa, but functionality will be broken.
        LIBROSA_AVAILABLE = False
        LIBROSA_ERROR_MESSAGE = "CRITICAL: 'audio_utils.py' is missing. This module is required for audio processing."
        MOVIEPY_ERROR_MESSAGE_INIT = "'audio_utils.py' missing, so MoviePy status is unknown."
        QMessageBox.warning(None, "Missing Module", "Failed to load 'audio_utils.py'. Audio features will be unavailable.")


import random

class OpticalFlowZoomerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Reactive Video Animator")
        self.setGeometry(100, 100, 1150, 820)
        QApplication.setStyle("Fusion")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.controls_panel = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_panel)
        self.controls_panel.setFixedWidth(480)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
        self.controls_layout.addWidget(self.scroll_area)

        # Define these QLineEdit instances before they are used in _create_file_io_group
        self.output_video_path_le = QLineEdit()
        self.ffmpeg_path_le = QLineEdit()


        self._create_file_io_group()
        self._create_parameter_tabs()

        self.generate_button = QPushButton("Generate Video")
        self.generate_button.setFixedHeight(38)
        self.generate_button.setStyleSheet("QPushButton { background-color: #5cb85c; color: white; font-size: 15px; border-radius: 4px; } QPushButton:hover { background-color: #4cae4c; } QPushButton:disabled { background-color: #cfcfcf; color: #777 }")
        self.generate_button.clicked.connect(self.start_zoomer_processing)
        self.controls_layout.addWidget(self.generate_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0,100)
        self.controls_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setFixedHeight(30)
        self.status_label.setWordWrap(True)
        self.controls_layout.addWidget(self.status_label)
        self.main_layout.addWidget(self.controls_panel)

        self.preview_panel = QWidget()
        self.preview_layout = QVBoxLayout(self.preview_panel)

        self.image_preview_label = QLabel("Input Image Preview")
        self.image_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview_label.setMinimumSize(320, 240)
        self.image_preview_label.setStyleSheet("QLabel { border: 1px solid #bbb; background-color: #e9e9e9; }")
        self.preview_layout.addWidget(self.image_preview_label, 1)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(320, 240)
        self.video_widget.setStyleSheet("QVideoWidget { border: 1px solid #bbb; background-color: #111; }")
        self.preview_layout.addWidget(self.video_widget, 1)

        self.media_player = QMediaPlayer()
        self.audio_output_player = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output_player)
        self.media_player.setVideoOutput(self.video_widget)

        self.play_pause_button = QPushButton("Play/Pause Video")
        self.play_pause_button.setEnabled(False)
        self.play_pause_button.clicked.connect(self.toggle_video_playback)
        self.preview_layout.addWidget(self.play_pause_button)
        self.main_layout.addWidget(self.preview_panel)

        self.image_path = None
        self.audio_path = None
        # Consider making ffmpeg_forced_path configurable or auto-detected
        self.ffmpeg_forced_path = "" 
        self.output_video_path_le.setText(os.path.join(os.getcwd(), "animated_output.mp4"))
        self.ffmpeg_path_le.setText(self.ffmpeg_forced_path if self.ffmpeg_forced_path else "ffmpeg")


        if not LIBROSA_AVAILABLE:
            self.generate_button.setEnabled(False)
            self.status_label.setText("Status: CRITICAL - Librosa (audio lib) UNAVAILABLE.")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            if hasattr(self, 'audio_file_le'): # Check if UI elements are created
                self.audio_file_le.setEnabled(False)
                self.audio_browse_btn.setEnabled(False)
                self.audio_file_le.setPlaceholderText("Audio library (Librosa) failed to load.")
            QMessageBox.critical(self, "Critical Dependency Error", LIBROSA_ERROR_MESSAGE)

        if MOVIEPY_ERROR_MESSAGE_INIT and "moviepy.editor" not in MOVIEPY_ERROR_MESSAGE_INIT : # Avoid showing if audio_utils itself is missing
            QMessageBox.warning(self, "Optional Dependency Warning", MOVIEPY_ERROR_MESSAGE_INIT)
            if self.status_label.text() == "Status: Ready":
                 self.status_label.setText("Status: moviepy not found, output may be silent if direct ffmpeg fails.")


    def _create_file_io_group(self):
        file_io_group = QGroupBox("Input / Output & FFMPEG")
        file_io_layout = QGridLayout(file_io_group)

        file_io_layout.addWidget(QLabel("Input Image/Video:"), 0, 0)
        self.image_file_le = QLineEdit()
        self.image_file_le.setPlaceholderText("Select input image or video file")
        file_io_layout.addWidget(self.image_file_le, 0, 1)
        self.image_browse_btn = QPushButton("Browse")
        self.image_browse_btn.clicked.connect(self.browse_image_file)
        file_io_layout.addWidget(self.image_browse_btn, 0, 2)

        file_io_layout.addWidget(QLabel("Audio File (Optional):"), 1, 0)
        self.audio_file_le = QLineEdit()
        self.audio_file_le.setPlaceholderText("Select audio file (.wav, .mp3, .flac)")
        file_io_layout.addWidget(self.audio_file_le, 1, 1)
        self.audio_browse_btn = QPushButton("Browse")
        self.audio_browse_btn.clicked.connect(self.browse_audio_file)
        file_io_layout.addWidget(self.audio_browse_btn, 1, 2)

        file_io_layout.addWidget(QLabel("Output Video:"), 2, 0)
        # self.output_video_path_le is already defined in __init__
        self.output_video_path_le.setPlaceholderText("Specify output video file path (.mp4)")
        file_io_layout.addWidget(self.output_video_path_le, 2, 1)
        self.output_browse_btn = QPushButton("Browse")
        self.output_browse_btn.clicked.connect(self.browse_output_video_path)
        file_io_layout.addWidget(self.output_browse_btn, 2, 2)

        file_io_layout.addWidget(QLabel("FFMPEG Path (Optional):"), 3, 0)
        # self.ffmpeg_path_le is already defined in __init__
        self.ffmpeg_path_le.setPlaceholderText("e.g., C:/ffmpeg/bin/ffmpeg.exe or /usr/bin/ffmpeg")
        file_io_layout.addWidget(self.ffmpeg_path_le, 3, 1)
        self.ffmpeg_browse_btn = QPushButton("Browse")
        self.ffmpeg_browse_btn.clicked.connect(self.browse_ffmpeg_path)
        file_io_layout.addWidget(self.ffmpeg_browse_btn, 3, 2)


        self.scroll_layout.addWidget(file_io_group)

    def _create_parameter_tabs(self):
        self.tabs = QTabWidget()
        self.param_widgets = {}

        tab_gen_audio = QWidget(); layout_ga = QVBoxLayout(tab_gen_audio)
        group_gen = QGroupBox("General Settings"); layout_g = QGridLayout(group_gen)
        self.param_widgets["fps"] = QSpinBox(); self.param_widgets["fps"].setRange(1,300); self.param_widgets["fps"].setValue(24)
        layout_g.addWidget(QLabel("FPS:"),0,0); layout_g.addWidget(self.param_widgets["fps"],0,1)
        self.param_widgets["seed"] = QLineEdit("-1")
        layout_g.addWidget(QLabel("Seed (-1 for random):"),1,0); layout_g.addWidget(self.param_widgets["seed"],1,1)
        self.param_widgets["override_num_frames"] = QSpinBox(); self.param_widgets["override_num_frames"].setRange(0, 999999); self.param_widgets["override_num_frames"].setValue(0)
        layout_g.addWidget(QLabel("Override Frames (0=auto):"),2,0); layout_g.addWidget(self.param_widgets["override_num_frames"],2,1)
        layout_ga.addWidget(group_gen)

        tab_breathing = QWidget(); layout_b = QVBoxLayout(tab_breathing)
        
        group_breath_general = QGroupBox("Breathing General Controls")
        layout_bg = QGridLayout(group_breath_general)
        self.param_widgets["enable_breathing_effects"] = QCheckBox("Enable Breathing Effects")
        self.param_widgets["enable_breathing_effects"].setChecked(True)
        layout_bg.addWidget(self.param_widgets["enable_breathing_effects"], 0, 0, 1, 2)
        self.param_widgets["breathing_smoothing_window"] = QDoubleSpinBox()
        self.param_widgets["breathing_smoothing_window"].setRange(0.0, 10.0); self.param_widgets["breathing_smoothing_window"].setValue(0.5); self.param_widgets["breathing_smoothing_window"].setSingleStep(0.1); self.param_widgets["breathing_smoothing_window"].setDecimals(2)
        layout_bg.addWidget(QLabel("Smoothing Window (s):"), 1, 0); layout_bg.addWidget(self.param_widgets["breathing_smoothing_window"], 1, 1)
        self.param_widgets["breathing_target_percentile"] = QSpinBox()
        self.param_widgets["breathing_target_percentile"].setRange(1,100); self.param_widgets["breathing_target_percentile"].setValue(75)
        layout_bg.addWidget(QLabel("RMS Target Percentile:"), 2, 0); layout_bg.addWidget(self.param_widgets["breathing_target_percentile"], 2, 1)
        layout_b.addWidget(group_breath_general)

        group_breath_zoom = QGroupBox("Breathing Zoom")
        layout_bz = QGridLayout(group_breath_zoom)
        self.param_widgets["enable_breathing_zoom"] = QCheckBox("Enable Zoom"); self.param_widgets["enable_breathing_zoom"].setChecked(True)
        layout_bz.addWidget(self.param_widgets["enable_breathing_zoom"], 0,0,1,2)
        self.param_widgets["breathing_max_zoom_factor"] = QDoubleSpinBox()
        self.param_widgets["breathing_max_zoom_factor"].setRange(0.0, 2.0); self.param_widgets["breathing_max_zoom_factor"].setValue(0.05); self.param_widgets["breathing_max_zoom_factor"].setSingleStep(0.01); self.param_widgets["breathing_max_zoom_factor"].setDecimals(3)
        layout_bz.addWidget(QLabel("Max Zoom Factor:"), 1, 0); layout_bz.addWidget(self.param_widgets["breathing_max_zoom_factor"], 1, 1)
        layout_b.addWidget(group_breath_zoom)

        group_breath_bright = QGroupBox("Breathing Brightness")
        layout_bb = QGridLayout(group_breath_bright)
        self.param_widgets["enable_breathing_brightness"] = QCheckBox("Enable Brightness"); self.param_widgets["enable_breathing_brightness"].setChecked(False)
        layout_bb.addWidget(self.param_widgets["enable_breathing_brightness"], 0,0,1,2)
        self.param_widgets["breathing_min_brightness"] = QDoubleSpinBox(); self.param_widgets["breathing_min_brightness"].setRange(0.0, 2.0); self.param_widgets["breathing_min_brightness"].setValue(0.9); self.param_widgets["breathing_min_brightness"].setSingleStep(0.05); self.param_widgets["breathing_min_brightness"].setDecimals(2)
        layout_bb.addWidget(QLabel("Min Brightness:"),1,0); layout_bb.addWidget(self.param_widgets["breathing_min_brightness"],1,1)
        self.param_widgets["breathing_max_brightness"] = QDoubleSpinBox(); self.param_widgets["breathing_max_brightness"].setRange(0.0, 2.0); self.param_widgets["breathing_max_brightness"].setValue(1.1); self.param_widgets["breathing_max_brightness"].setSingleStep(0.05); self.param_widgets["breathing_max_brightness"].setDecimals(2)
        layout_bb.addWidget(QLabel("Max Brightness:"),2,0); layout_bb.addWidget(self.param_widgets["breathing_max_brightness"],2,1)
        layout_b.addWidget(group_breath_bright)

        group_breath_blur = QGroupBox("Breathing Blur")
        layout_bbl = QGridLayout(group_breath_blur)
        self.param_widgets["enable_breathing_blur"] = QCheckBox("Enable Blur"); self.param_widgets["enable_breathing_blur"].setChecked(False)
        layout_bbl.addWidget(self.param_widgets["enable_breathing_blur"],0,0,1,2)
        self.param_widgets["breathing_max_blur_radius"] = QDoubleSpinBox(); self.param_widgets["breathing_max_blur_radius"].setRange(0.0, 10.0); self.param_widgets["breathing_max_blur_radius"].setValue(1.0); self.param_widgets["breathing_max_blur_radius"].setSingleStep(0.1); self.param_widgets["breathing_max_blur_radius"].setDecimals(2)
        layout_bbl.addWidget(QLabel("Max Blur Radius:"),1,0); layout_bbl.addWidget(self.param_widgets["breathing_max_blur_radius"],1,1)
        layout_b.addWidget(group_breath_blur)

        group_breath_sat = QGroupBox("Breathing Saturation")
        layout_bs = QGridLayout(group_breath_sat)
        self.param_widgets["enable_breathing_saturation"] = QCheckBox("Enable Saturation"); self.param_widgets["enable_breathing_saturation"].setChecked(False)
        layout_bs.addWidget(self.param_widgets["enable_breathing_saturation"],0,0,1,2)
        self.param_widgets["breathing_min_saturation"] = QDoubleSpinBox(); self.param_widgets["breathing_min_saturation"].setRange(0.0, 2.0); self.param_widgets["breathing_min_saturation"].setValue(0.8); self.param_widgets["breathing_min_saturation"].setSingleStep(0.05); self.param_widgets["breathing_min_saturation"].setDecimals(2)
        layout_bs.addWidget(QLabel("Min Saturation:"),1,0); layout_bs.addWidget(self.param_widgets["breathing_min_saturation"],1,1)
        self.param_widgets["breathing_max_saturation"] = QDoubleSpinBox(); self.param_widgets["breathing_max_saturation"].setRange(0.0, 2.0); self.param_widgets["breathing_max_saturation"].setValue(1.2); self.param_widgets["breathing_max_saturation"].setSingleStep(0.05); self.param_widgets["breathing_max_saturation"].setDecimals(2)
        layout_bs.addWidget(QLabel("Max Saturation:"),2,0); layout_bs.addWidget(self.param_widgets["breathing_max_saturation"],2,1)
        layout_b.addWidget(group_breath_sat)

        group_breath_color = QGroupBox("Breathing Color Shift")
        layout_bc = QGridLayout(group_breath_color)
        self.param_widgets["enable_breathing_color_shift"] = QCheckBox("Enable Color Shift")
        self.param_widgets["enable_breathing_color_shift"].setChecked(False)
        layout_bc.addWidget(self.param_widgets["enable_breathing_color_shift"], 0, 0, 1, 2)
        self.param_widgets["breathing_min_hue_shift"] = QDoubleSpinBox()
        self.param_widgets["breathing_min_hue_shift"].setRange(0.0, 1.0)
        self.param_widgets["breathing_min_hue_shift"].setDecimals(3)
        self.param_widgets["breathing_min_hue_shift"].setSingleStep(0.01)
        self.param_widgets["breathing_min_hue_shift"].setValue(0.0)
        layout_bc.addWidget(QLabel("Min Hue Shift:"), 1, 0)
        layout_bc.addWidget(self.param_widgets["breathing_min_hue_shift"], 1, 1)
        self.param_widgets["breathing_max_hue_shift"] = QDoubleSpinBox()
        self.param_widgets["breathing_max_hue_shift"].setRange(0.0, 1.0)
        self.param_widgets["breathing_max_hue_shift"].setDecimals(3)
        self.param_widgets["breathing_max_hue_shift"].setSingleStep(0.01)
        self.param_widgets["breathing_max_hue_shift"].setValue(0.15)
        layout_bc.addWidget(QLabel("Max Hue Shift:"), 2, 0)
        layout_bc.addWidget(self.param_widgets["breathing_max_hue_shift"], 2, 1)
        layout_b.addWidget(group_breath_color)
        
        layout_b.addStretch(1)
        self.tabs.addTab(tab_breathing, "Breathing Effects")

        tab_peak_flow = QWidget(); layout_pf = QVBoxLayout(tab_peak_flow)
        
        self.param_widgets["use_original_peak_effects"] = QCheckBox("Enable Peak/Optical Flow Effects")
        self.param_widgets["use_original_peak_effects"].setChecked(True)
        layout_pf.addWidget(self.param_widgets["use_original_peak_effects"])

        group_audio_peak = QGroupBox("Audio Peak Analysis (for Flow)"); layout_a = QGridLayout(group_audio_peak)
        self.param_widgets["peak_threshold_multiplier"] = QDoubleSpinBox(); self.param_widgets["peak_threshold_multiplier"].setRange(0.0, 5.0); self.param_widgets["peak_threshold_multiplier"].setValue(0.8); self.param_widgets["peak_threshold_multiplier"].setSingleStep(0.1); self.param_widgets["peak_threshold_multiplier"].setDecimals(2)
        layout_a.addWidget(QLabel("Peak Threshold Multiplier:"),0,0); layout_a.addWidget(self.param_widgets["peak_threshold_multiplier"],0,1)
        self.param_widgets["peak_hold_frames"] = QSpinBox(); self.param_widgets["peak_hold_frames"].setRange(0,300); self.param_widgets["peak_hold_frames"].setValue(12)
        layout_a.addWidget(QLabel("Peak Hold Frames:"),1,0); layout_a.addWidget(self.param_widgets["peak_hold_frames"],1,1)
        self.param_widgets["alternate_every_n_peaks"] = QSpinBox(); self.param_widgets["alternate_every_n_peaks"].setRange(0,100); self.param_widgets["alternate_every_n_peaks"].setValue(1)
        layout_a.addWidget(QLabel("Alternate Direction Every N Peaks:"),2,0); layout_a.addWidget(self.param_widgets["alternate_every_n_peaks"],2,1)
        layout_pf.addWidget(group_audio_peak)

        group_peak_env = QGroupBox("Peak Effect Envelope (for Flow)"); layout_p_env = QGridLayout(group_peak_env)
        self.param_widgets["peak_attack_ratio"] = QDoubleSpinBox(); self.param_widgets["peak_attack_ratio"].setRange(0.0,1.0); self.param_widgets["peak_attack_ratio"].setValue(0.2); self.param_widgets["peak_attack_ratio"].setSingleStep(0.05); self.param_widgets["peak_attack_ratio"].setDecimals(2)
        layout_p_env.addWidget(QLabel("Attack Ratio:"),0,0); layout_p_env.addWidget(self.param_widgets["peak_attack_ratio"],0,1)
        self.param_widgets["peak_sustain_ratio"] = QDoubleSpinBox(); self.param_widgets["peak_sustain_ratio"].setRange(0.0,1.0); self.param_widgets["peak_sustain_ratio"].setValue(0.5); self.param_widgets["peak_sustain_ratio"].setSingleStep(0.05); self.param_widgets["peak_sustain_ratio"].setDecimals(2)
        layout_p_env.addWidget(QLabel("Sustain Ratio:"),1,0); layout_p_env.addWidget(self.param_widgets["peak_sustain_ratio"],1,1)
        self.param_widgets["peak_decay_ratio"] = QDoubleSpinBox(); self.param_widgets["peak_decay_ratio"].setRange(0.0,1.0); self.param_widgets["peak_decay_ratio"].setValue(0.3); self.param_widgets["peak_decay_ratio"].setSingleStep(0.05); self.param_widgets["peak_decay_ratio"].setDecimals(2)
        layout_p_env.addWidget(QLabel("Decay Ratio:"),2,0); layout_p_env.addWidget(self.param_widgets["peak_decay_ratio"],2,1)
        self.param_widgets["peak_easing_type"] = QComboBox(); self.param_widgets["peak_easing_type"].addItems(["linear", "sine", "quad_in", "quad_out"]); self.param_widgets["peak_easing_type"].setCurrentText("sine")
        layout_p_env.addWidget(QLabel("Easing Type:"),3,0); layout_p_env.addWidget(self.param_widgets["peak_easing_type"],3,1)
        layout_pf.addWidget(group_peak_env)

        group_flow_target = QGroupBox("Flow Target & Calculation"); layout_ft = QGridLayout(group_flow_target)
        self.param_widgets["flow_target_generation_mode"] = QComboBox(); self.param_widgets["flow_target_generation_mode"].addItems(["Zoom_In_Flow_Target", "Zoom_Out_Flow_Target"]); self.param_widgets["flow_target_generation_mode"].setCurrentText("Zoom_In_Flow_Target")
        layout_ft.addWidget(QLabel("Flow Target Mode:"),0,0); layout_ft.addWidget(self.param_widgets["flow_target_generation_mode"],0,1)
        self.param_widgets["flow_target_transform_amount"] = QDoubleSpinBox(); self.param_widgets["flow_target_transform_amount"].setRange(0.0,2.0); self.param_widgets["flow_target_transform_amount"].setValue(0.15); self.param_widgets["flow_target_transform_amount"].setSingleStep(0.001); self.param_widgets["flow_target_transform_amount"].setDecimals(3)
        layout_ft.addWidget(QLabel("Target Transform Amount:"),1,0); layout_ft.addWidget(self.param_widgets["flow_target_transform_amount"],1,1)
        self.param_widgets["flow_calc_scale_factor"] = QDoubleSpinBox(); self.param_widgets["flow_calc_scale_factor"].setRange(0.01,1.0); self.param_widgets["flow_calc_scale_factor"].setValue(0.5); self.param_widgets["flow_calc_scale_factor"].setSingleStep(0.05); self.param_widgets["flow_calc_scale_factor"].setDecimals(2)
        layout_ft.addWidget(QLabel("Flow Calc Scale Factor:"),2,0); layout_ft.addWidget(self.param_widgets["flow_calc_scale_factor"],2,1)
        layout_pf.addWidget(group_flow_target)

        group_flow_strength = QGroupBox("Flow Strength"); layout_fs = QGridLayout(group_flow_strength)
        self.param_widgets["idle_flow_strength"] = QDoubleSpinBox(); self.param_widgets["idle_flow_strength"].setRange(-2.0,2.0); self.param_widgets["idle_flow_strength"].setValue(0.0); self.param_widgets["idle_flow_strength"].setSingleStep(0.01); self.param_widgets["idle_flow_strength"].setDecimals(3)
        layout_fs.addWidget(QLabel("Idle Flow Strength:"),0,0); layout_fs.addWidget(self.param_widgets["idle_flow_strength"],0,1)
        self.param_widgets["peak_flow_strength_zoom_in"] = QDoubleSpinBox(); self.param_widgets["peak_flow_strength_zoom_in"].setRange(-2.0,2.0); self.param_widgets["peak_flow_strength_zoom_in"].setValue(0.4); self.param_widgets["peak_flow_strength_zoom_in"].setSingleStep(0.05); self.param_widgets["peak_flow_strength_zoom_in"].setDecimals(3)
        layout_fs.addWidget(QLabel("Peak Strength (Zoom In):"),1,0); layout_fs.addWidget(self.param_widgets["peak_flow_strength_zoom_in"],1,1)
        self.param_widgets["peak_flow_strength_zoom_out"] = QDoubleSpinBox(); self.param_widgets["peak_flow_strength_zoom_out"].setRange(-2.0,2.0); self.param_widgets["peak_flow_strength_zoom_out"].setValue(-0.3); self.param_widgets["peak_flow_strength_zoom_out"].setSingleStep(0.05); self.param_widgets["peak_flow_strength_zoom_out"].setDecimals(3)
        layout_fs.addWidget(QLabel("Peak Strength (Zoom Out):"),2,0); layout_fs.addWidget(self.param_widgets["peak_flow_strength_zoom_out"],2,1)
        layout_pf.addWidget(group_flow_strength)

        group_interp = QGroupBox("Interpolation & Boundary (for Flow)"); layout_ip = QGridLayout(group_interp)
        self.param_widgets["warp_interpolation_cv2"] = QComboBox(); self.param_widgets["warp_interpolation_cv2"].addItems(["Nearest_cv2", "Linear_cv2", "Cubic_cv2", "Lanczos4_cv2"]); self.param_widgets["warp_interpolation_cv2"].setCurrentText("Linear_cv2")
        layout_ip.addWidget(QLabel("Warp Interpolation (cv2):"),0,0); layout_ip.addWidget(self.param_widgets["warp_interpolation_cv2"],0,1)
        self.param_widgets["zoom_interpolation_pil"] = QComboBox(); self.param_widgets["zoom_interpolation_pil"].addItems(["Nearest", "Bilinear", "Bicubic", "Lanczos"]); self.param_widgets["zoom_interpolation_pil"].setCurrentText("Lanczos")
        layout_ip.addWidget(QLabel("Zoom Interpolation (PIL):"),1,0); layout_ip.addWidget(self.param_widgets["zoom_interpolation_pil"],1,1)
        self.param_widgets["boundary_mode_cv2"] = QComboBox(); self.param_widgets["boundary_mode_cv2"].addItems(["Constant_cv2", "Replicate_cv2", "Reflect_cv2", "Wrap_cv2", "Reflect_101_cv2"]); self.param_widgets["boundary_mode_cv2"].setCurrentText("Reflect_101_cv2")
        layout_ip.addWidget(QLabel("Boundary Mode (cv2):"),2,0); layout_ip.addWidget(self.param_widgets["boundary_mode_cv2"],2,1)
        layout_pf.addWidget(group_interp)
        
        layout_pf.addStretch(1)
        self.tabs.addTab(tab_peak_flow, "Peak/Flow Effects")

        self.tabs.insertTab(0, tab_gen_audio, "General Settings")


        self.scroll_layout.addWidget(self.tabs)

    def browse_image_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Image or Video", self.image_path or "",
            "Image/Video Files (*.png *.jpg *.jpeg *.bmp *.gif *.mp4 *.avi *.mov *.mkv *.webm *.wmv *.flv);;All Files (*)"
        )
        if path:
            self.image_path = path
            self.image_file_le.setText(path)
            self.update_image_preview(path)
            self.status_label.setText(f"Input: {os.path.basename(path)}"); self.status_label.setStyleSheet("")

    def browse_audio_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", self.audio_path or "", "Audio Files (*.wav *.mp3 *.flac);;All Files (*)"
        )
        if path:
            self.audio_path = path
            self.audio_file_le.setText(path)
            self.status_label.setText(f"Audio: {os.path.basename(path)}"); self.status_label.setStyleSheet("")

    def browse_output_video_path(self):
        default_path = self.output_video_path_le.text() or os.path.join(os.getcwd(), "animated_output.mp4")
        path, _ = QFileDialog.getSaveFileName(self, "Save Output Video", default_path, "Video Files (*.mp4)")
        if path:
            if not path.lower().endswith(".mp4"): path += ".mp4"
            self.output_video_path_le.setText(path)

    def browse_ffmpeg_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select FFMPEG Executable", self.ffmpeg_path_le.text() or "", "All Files (*)")
        if path:
            self.ffmpeg_path_le.setText(path)


    def update_image_preview(self, image_path_to_preview):
        try:
            ext = os.path.splitext(image_path_to_preview)[1].lower()
            video_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"]
            if ext in video_exts:
                # Using imageio to get the first frame
                try:
                    vid = imageio.get_reader(image_path_to_preview)
                    frame = vid.get_data(0) # Get the first frame
                    vid.close()
                    img = Image.fromarray(frame) # Convert numpy array to PIL Image
                    # Convert PIL Image to QImage
                    # Ensure image is RGB for QImage.Format_RGB888
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    qimg = QImage(img.tobytes(), img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                except Exception as e:
                    self.image_preview_label.setText(f"Video Preview Error:\n{e}")
                    print(f"Error previewing video frame: {e}")
                    return
            else: # It's an image
                pixmap = QPixmap(image_path_to_preview)

            if pixmap.isNull():
                self.image_preview_label.setText("Cannot load preview")
                return
            # Scale pixmap to fit the label while keeping aspect ratio
            scaled_pixmap = pixmap.scaled(self.image_preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_preview_label.setPixmap(scaled_pixmap)
        except Exception as e:
            self.image_preview_label.setText(f"Preview Error:\n{e}")
            print(f"Error updating image preview: {e}")


    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image_path and os.path.exists(self.image_path) and self.image_preview_label.pixmap():
             self.update_image_preview(self.image_path)


    def collect_parameters(self):
        params = {}
        for name, widget in self.param_widgets.items():
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)): params[name] = widget.value()
            elif isinstance(widget, QComboBox): params[name] = widget.currentText()
            elif isinstance(widget, QCheckBox): params[name] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                try:
                    val = int(widget.text())
                    if name == "seed" and val < 0: # Seed specifically can be -1
                        params[name] = -1
                    else:
                        params[name] = val
                except ValueError: # Handle cases where text is not a valid integer
                    params[name] = 0 # Default or error value
                    if name == "seed": params[name] = -1 # Default seed
                    QMessageBox.warning(self, f"Invalid Input for {name}", f"Parameter '{name}' expects an integer. Using default value.")
        return params

    def start_zoomer_processing(self):
        if not LIBROSA_AVAILABLE and (self.param_widgets["enable_breathing_effects"].isChecked() or self.param_widgets["use_original_peak_effects"].isChecked()):
            QMessageBox.critical(self, "Cannot Proceed", "Librosa (audio library) is not available, but audio-reactive effects are enabled.\nPlease install Librosa or disable audio-reactive effects.\n" + LIBROSA_ERROR_MESSAGE); return

        self.status_label.setText("Status: Starting..."); self.status_label.setStyleSheet("")
        self.progress_bar.setValue(0); self.generate_button.setEnabled(False); self.play_pause_button.setEnabled(False)
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState: self.media_player.pause()

        current_params = self.collect_parameters()
        output_video_file = self.output_video_path_le.text()
        
        # Use ffmpeg path from QLineEdit, fallback to forced path, then fallback to 'ffmpeg'
        ffmpeg_custom_path = self.ffmpeg_path_le.text().strip()
        if not ffmpeg_custom_path and self.ffmpeg_forced_path:
            ffmpeg_custom_path = self.ffmpeg_forced_path
        if not ffmpeg_custom_path: # If still empty, worker will use "ffmpeg"
            ffmpeg_custom_path = None


        if not self.image_path: QMessageBox.warning(self, "Input Error", "Please select an input image."); self.generate_button.setEnabled(True); self.status_label.setText("Status: Ready"); return
        
        audio_effects_enabled = current_params.get("enable_breathing_effects", False) or \
                                current_params.get("use_original_peak_effects", False)
        
        current_audio_path = self.audio_file_le.text().strip()
        if audio_effects_enabled and not current_audio_path:
            QMessageBox.warning(self, "Input Error", "Audio-reactive effects are enabled, but no audio file selected."); self.generate_button.setEnabled(True); self.status_label.setText("Status: Ready"); return
        
        self.audio_path = current_audio_path if current_audio_path else None


        if not output_video_file: QMessageBox.warning(self, "Output Error", "Please specify an output video path."); self.generate_button.setEnabled(True); self.status_label.setText("Status: Ready"); return

        output_dir = os.path.dirname(output_video_file)
        if output_dir and not os.path.exists(output_dir):
            try: os.makedirs(output_dir)
            except OSError as e: QMessageBox.critical(self, "Output Error", f"Could not create output directory: {output_dir}\n{e}"); self.generate_button.setEnabled(True); self.status_label.setText("Status: Ready"); return

        self.worker = OpticalFlowZoomerWorker(
            self.image_path, self.audio_path, output_video_file, current_params, ffmpeg_custom_path
        )
        self.worker.progress_update.connect(self.update_progress)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.start()

    def update_progress(self, message, value):
        self.status_label.setText(f"Status: {message}"); self.status_label.setStyleSheet("")
        self.progress_bar.setValue(value)

    def on_processing_finished(self, video_path, success, error_message):
        self.generate_button.setEnabled(True)
        if success and video_path: # Check if video_path is not empty
            final_message = "Video generated successfully!"
            if "WARNING:" in error_message.upper() or "SILENT" in error_message.upper():
                final_message = error_message # Show warning from worker
                self.status_label.setStyleSheet("color: orange; font-weight:bold;")
            else:
                self.status_label.setStyleSheet("color: green; font-weight:bold;")

            self.status_label.setText(f"Status: {final_message.splitlines()[0]}") # Show first line of message

            # Attempt to play the video
            self.media_player.setSource(QUrl()) # Clear previous source
            time.sleep(0.1) # Short delay
            self.media_player.setSource(QUrl.fromLocalFile(os.path.abspath(video_path))) # Use absolute path
            self.play_pause_button.setEnabled(True)

            # Check media status before playing
            if self.media_player.mediaStatus() == QMediaPlayer.MediaStatus.LoadedMedia or \
               self.media_player.mediaStatus() == QMediaPlayer.MediaStatus.BufferedMedia:
                if self.media_player.error() == QMediaPlayer.Error.NoError:
                    self.media_player.play()
                else:
                    print(f"MediaPlayer Error before play: {self.media_player.errorString()} (Code: {self.media_player.error()}) for source {video_path}")
                    QMessageBox.warning(self, "Playback Warning", f"Video generated, but could not auto-play: {self.media_player.errorString()}")
            elif self.media_player.mediaStatus() == QMediaPlayer.MediaStatus.NoMedia and video_path:
                 QMessageBox.warning(self, "Playback Warning", f"Video generated at {video_path}, but media player could not load it. It might be invalid or an issue with codecs.")
            else:
                # Suppress popup for LoadingMedia, just show generation complete
                print(f"MediaPlayer Status not ready for play: {self.media_player.mediaStatus()} for source {video_path}")
                self.status_label.setText("Status: Generation complete. Video saved.")
                # No popup here
        
        elif not success: # Processing failed
            self.status_label.setText(f"Status: Error - {error_message.splitlines()[0]}"); self.status_label.setStyleSheet("color: red; font-weight:bold;")
            QMessageBox.critical(self, "Processing Error", f"An error occurred:\n{error_message}")
            self.progress_bar.setValue(0); self.play_pause_button.setEnabled(False); self.media_player.setSource(QUrl())
        else: # Success but no video_path (should not happen if success is true)
            self.status_label.setText("Status: Finished, but no video path returned."); self.status_label.setStyleSheet("color: orange;")
            self.progress_bar.setValue(100)


    def toggle_video_playback(self):
        if self.media_player.source().isEmpty(): return

        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause(); self.status_label.setText("Status: Video paused")
        else:
            # Before attempting to play, check for errors on the current source
            if self.media_player.error() != QMediaPlayer.Error.NoError:
                print(f"MediaPlayer Error on attempting play: {self.media_player.errorString()} for {self.media_player.source().toLocalFile()}")
                # Optionally, try to reload the source
                # current_source_url = self.media_player.source()
                # self.media_player.setSource(QUrl())
                # time.sleep(0.1)
                # self.media_player.setSource(current_source_url)
                # if self.media_player.error() != QMediaPlayer.Error.NoError: # Check again
                QMessageBox.warning(self, "Playback Error", f"Cannot play video: {self.media_player.errorString()}\nSource: {self.media_player.source().toLocalFile()}")
                self.status_label.setText("Status: Playback error")
                return # Don't try to play if there's an error

            self.media_player.play(); self.status_label.setText("Status: Video playing")
            if self.media_player.error() != QMediaPlayer.Error.NoError: # Check error *after* play attempt
                QMessageBox.warning(self, "Playback Error", f"Error occurred during play: {self.media_player.errorString()}")
                self.status_label.setText("Status: Playback error after play attempt")


        self.status_label.setStyleSheet("")


    def closeEvent(self, event):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, 'Confirm Exit', "Processing is ongoing. Are you sure you want to exit?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.quit() # Request graceful exit
                if not self.worker.wait(3000): # Wait up to 3 seconds
                    self.worker.terminate() # Force terminate if still running
                    self.worker.wait() # Wait for termination
                event.accept()
            else: event.ignore()
        else: event.accept()

if __name__ == "__main__":
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'): QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'): QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    main_window = OpticalFlowZoomerApp()
    main_window.show()
    sys.exit(app.exec())