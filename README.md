# AI Based Anti Sleep Alarm Detector for Driver

A real-time drowsiness detection system that uses a webcam, OpenCV Haarcascade
face/eye detectors, and a Pygame audio alert to warn a driver when signs of
sleepiness (eyes closed for a sustained period) are detected. Live drowsiness
metrics are also plotted with Matplotlib.

## Project Structure

```
AI_Based_Anti_Sleep_Alarm_Detector_For_Driver/
├── drowsiness_detector.py   # Main application
├── requirements.txt         # Python dependencies
├── sounds/
│   └── buzz.mp3.wav          # Alert sound (ADD THIS FILE YOURSELF)
└── README.md
```

## How It Works

1. Captures live video from the default webcam.
2. Uses Haarcascade classifiers (bundled with OpenCV) to detect the face and
   eyes in each frame.
3. Tracks how often eyes are detected as "closed" vs "open" over a rolling
   45-second window.
4. If the eye-closure ratio in that window is >= 0.2 (20%), it triggers an
   audio alert on a background thread so the video loop isn't blocked.
5. Also counts blinks (a closed → open transition) and plots both the
   eye-closure ratio and blink count live in a Matplotlib window.
6. Press **ESC** while the camera window is focused to quit.

## Setup

1. **Install Python 3.8+** if you don't already have it.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add the alert sound file:**
   This project expects an audio file at `sounds/buzz.mp3.wav`. That file is
   NOT included in this package (no bundled audio) — add any short `.wav`
   alarm/buzzer sound of your own and place it at that path, or edit the
   `play_alert()` function in `drowsiness_detector.py` to point to a
   different file/path.

4. **Run the program:**
   ```bash
   python drowsiness_detector.py
   ```

   Make sure your webcam is connected and not in use by another
   application.

## Notes / Possible Improvements

- Haarcascades are lightweight but can be less accurate than modern
  landmark-based approaches (e.g. dlib's 68-point facial landmarks or
  MediaPipe Face Mesh) for computing a proper Eye Aspect Ratio (EAR).
- The current logic treats "no eyes detected by the cascade" as "eyes
  closed," which can also happen due to poor lighting, glasses, or head
  angle — worth tuning `detectMultiScale` parameters or switching to EAR
  based detection for more robustness.
- Consider adding a config section (thresholds, check interval, camera
  index) at the top of the script for easier tuning.
- Consider logging session data (CSV) alongside the live graph for later
  analysis/reporting.

## Requirements

- Python 3.8+
- A working webcam
- opencv-python
- pygame
- matplotlib
