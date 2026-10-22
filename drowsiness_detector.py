
import cv2
import time
import os
import threading
import pygame
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from collections import deque

# ==========================================
# 1. SETTINGS
# ==========================================
EAR_THRESHOLD = 0.21
DROWSY_SECONDS = 1.5
MIN_BLINK_SECONDS = 0.08
MAX_BLINK_SECONDS = 1.0
GRAPH_SECONDS = 30

# ==========================================
# 2. INITIALIZE BUZZER
# ==========================================
SOUND_READY = False

try:
    pygame.mixer.init()

    # Load buzz.wav from this script's folder
    script_folder = os.path.dirname(os.path.abspath(__file__))
    sound_path = os.path.join(script_folder, "buzz.wav")

    if not os.path.isfile(sound_path):
        raise FileNotFoundError(
            f"buzz.wav not found in: {script_folder}"
        )

    pygame.mixer.music.load(sound_path)
    SOUND_READY = True
    print("Buzzer loaded successfully.")

except Exception as e:
    print("Buzzer could not be initialized:", e)
    print("Place a valid WAV file named buzz.wav beside this script.")

alarm_lock = threading.Lock()
alarm_playing = False


def start_alarm():
    global alarm_playing

    if not SOUND_READY:
        return

    with alarm_lock:
        if not alarm_playing:
            try:
                pygame.mixer.music.play(-1)
                alarm_playing = True
            except pygame.error as e:
                print("Alarm playback error:", e)


def stop_alarm():
    global alarm_playing

    with alarm_lock:
        if alarm_playing and pygame.mixer.get_init():
            pygame.mixer.music.stop()
            alarm_playing = False


# ==========================================
# 3. EYE ASPECT RATIO (EAR)
# ==========================================
def calculate_ear(landmarks, indices, width, height):
    points = []

    for index in indices:
        point = landmarks[index]
        points.append(
            np.array([point.x * width, point.y * height])
        )

    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ==========================================
# 4. INITIALIZE FACE MESH AND CAMERA
# ==========================================
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    face_mesh.close()
    pygame.mixer.quit()
    raise RuntimeError(
        "Cannot open webcam. Check camera permissions."
    )

# ==========================================
# 5. TRACKING VARIABLES
# ==========================================
closed_start = None
blink_count = 0
start_time = time.time()
last_graph_update = 0

graph_time = deque()
graph_ear = deque()
graph_blinks = deque()

# ==========================================
# 6. PREPARE LIVE GRAPHS
# ==========================================
plt.ion()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
fig.suptitle("Real-Time Driver Drowsiness Monitoring")

line_ear, = ax1.plot([], [], label="Average EAR")
ax1.axhline(
    EAR_THRESHOLD,
    linestyle="--",
    label="EAR threshold"
)
ax1.set_ylabel("Eye Aspect Ratio")
ax1.set_title("Eye Opening Over Time")
ax1.grid(True)
ax1.legend()

line_blinks, = ax2.plot([], [], label="Blink Count")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Total blinks")
ax2.set_title("Blink Count Over Time")
ax2.grid(True)
ax2.legend()

# ==========================================
# 7. REAL-TIME DETECTION LOOP
# ==========================================
try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Camera frame unavailable.")
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        now = time.time()
        elapsed = now - start_time

        status = "NO FACE DETECTED"
        color = (0, 165, 255)
        ear = None

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear = calculate_ear(
                landmarks, LEFT_EYE, width, height
            )
            right_ear = calculate_ear(
                landmarks, RIGHT_EYE, width, height
            )
            ear = (left_ear + right_ear) / 2.0

            # Draw eye landmark points
            for index in LEFT_EYE + RIGHT_EYE:
                point = landmarks[index]
                px = int(point.x * width)
                py = int(point.y * height)
                cv2.circle(
                    frame, (px, py), 2, (0, 255, 0), -1
                )

            # ----------------------------------
            # DETECT EYE CLOSURE
            # ----------------------------------
            eyes_closed = ear < EAR_THRESHOLD

            if eyes_closed:
                if closed_start is None:
                    closed_start = now

                closed_duration = now - closed_start

                if closed_duration >= DROWSY_SECONDS:
                    status = "DROWSINESS ALERT!"
                    color = (0, 0, 255)
                    start_alarm()
                else:
                    status = "EYES CLOSING"
                    color = (0, 165, 255)

            else:
                if closed_start is not None:
                    closed_duration = now - closed_start

                    if (
                        MIN_BLINK_SECONDS
                        <= closed_duration
                        <= MAX_BLINK_SECONDS
                    ):
                        blink_count += 1

                closed_start = None
                stop_alarm()
                status = "AWAKE - MONITORING"
                color = (0, 255, 0)

            # ----------------------------------
            # UPDATE GRAPH DATA
            # ----------------------------------
            graph_time.append(elapsed)
            graph_ear.append(ear)
            graph_blinks.append(blink_count)

            while (
                graph_time
                and elapsed - graph_time[0] > GRAPH_SECONDS
            ):
                graph_time.popleft()
                graph_ear.popleft()
                graph_blinks.popleft()

        else:
            # Stop alarm if the face disappears
            closed_start = None
            stop_alarm()

        # ======================================
        # 8. DISPLAY LIVE STATUS
        # ======================================
        cv2.putText(
            frame, status, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

        if ear is not None:
            cv2.putText(
                frame, f"EAR: {ear:.3f}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 0), 2
            )

        cv2.putText(
            frame, f"Blinks: {blink_count}", (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 0), 2
        )

        if closed_start is not None:
            duration = now - closed_start
            cv2.putText(
                frame, f"Eye closure: {duration:.1f}s", (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

        cv2.imshow("Real-Time Drowsiness Detector", frame)

        # Refresh graphs twice per second
        if now - last_graph_update >= 0.5 and graph_time:
            line_ear.set_data(
                list(graph_time), list(graph_ear)
            )
            line_blinks.set_data(
                list(graph_time), list(graph_blinks)
            )

            left = max(0, elapsed - GRAPH_SECONDS)
            right = max(GRAPH_SECONDS, elapsed)

            ax1.set_xlim(left, right)
            ax2.set_xlim(left, right)

            ax1.relim()
            ax1.autoscale_view(scalex=False, scaley=True)
            ax2.relim()
            ax2.autoscale_view(scalex=False, scaley=True)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            last_graph_update = now

        plt.pause(0.001)

        # Press ESC in the webcam window to stop
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    stop_alarm()
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    # Safely close the graph window
    try:
        plt.ioff()
        plt.close("all")
    except Exception:
        pass

    if pygame.mixer.get_init():
        pygame.mixer.quit()

    print("Drowsiness detection stopped.")