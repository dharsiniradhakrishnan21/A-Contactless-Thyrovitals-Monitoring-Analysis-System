import tkinter as tk
from tkinter import messagebox
import threading
import cv2
from PIL import Image, ImageTk
import numpy as np
import time
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
import mediapipe as mp
import matplotlib.pyplot as plt

mp_face_mesh = mp.solutions.face_mesh

def capture_video_frames(num_frames=240, width=320, height=240, fps=20):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    frames, timestamps = [], []
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        timestamps.append(time.time())
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
    cap.release()
    return frames, timestamps

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    y = filtfilt(b, a, data)
    return y

def detect_eyebrows(img):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                               refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return img, None

        face_landmarks = results.multi_face_landmarks[0]
        ih, iw, _ = img.shape

        left_ids = [55, 65, 52, 53, 46, 124, 156, 70, 63, 105]
        right_ids = [285, 295, 282, 283, 276, 354, 386, 310, 300, 334]

        left_pts = [(int(face_landmarks.landmark[i].x * iw), int(face_landmarks.landmark[i].y * ih)) for i in left_ids]
        right_pts = [(int(face_landmarks.landmark[i].x * iw), int(face_landmarks.landmark[i].y * ih)) for i in right_ids]

        img_out = img.copy()
        for pts in [left_pts, right_pts]:
            for i in range(len(pts) - 1):
                cv2.line(img_out, pts[i], pts[i + 1], (0, 255, 0), 2)

        return img_out, left_pts + right_pts

def analyze_eyebrow_thinning(img, points, threshold=500):
    if points is None:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min = max(min(xs) - 5, 0)
    x_max = min(max(xs) + 5, img.shape[1])
    y_min = max(min(ys) - 5, 0)
    y_max = min(max(ys) + 5, img.shape[0])

    roi = img[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    variance = gray.var()
    thinning = variance < threshold
    return thinning, variance

def analyze_and_display(output_box):
    output_box.insert(tk.END, "Processing video frames...\n")
    frames, timestamps = capture_video_frames()
    if len(frames) < 30:
        output_box.insert(tk.END, "Not enough frames captured.\n")
        return

    green_signal = []
    for frame in frames:
        h, w, _ = frame.shape
        roi = frame[h//8:h//8+h//6, w//3:w//3+w//3]
        green_mean = np.mean(roi[:, :, 1])
        green_signal.append(green_mean)

    green_signal = np.array(green_signal)
    elapsed_time = timestamps[-1] - timestamps[0]
    fs = len(green_signal) / elapsed_time

    # Detrend
    window_size = 5
    moving_avg = np.convolve(green_signal, np.ones(window_size)/window_size, mode='same')
    detrended = green_signal - moving_avg

    # Heart and breathing signal filtering
    hr_signal = bandpass_filter(detrended, 0.8, 2.5, fs)     # 48 - 150 bpm
    br_signal = bandpass_filter(detrended, 0.1, 0.4, fs)     # 6 - 24 breaths/min

    # Peak detection
    hr_peaks, _ = find_peaks(hr_signal, distance=fs/2.5)
    br_peaks, _ = find_peaks(br_signal, distance=fs*2.5)     # Min 2.5s between breaths

    # Compute Rates
    heart_rate = 60 / np.mean(np.diff(hr_peaks) / fs) if len(hr_peaks) > 1 else None
    breathing_rate = 60 / np.mean(np.diff(br_peaks) / fs) if len(br_peaks) > 1 else None

    if heart_rate:
        output_box.insert(tk.END, f"Heart Rate: {heart_rate:.1f} bpm\n")
    else:
        output_box.insert(tk.END, "Heart rate not detected\n")

    if breathing_rate:
        output_box.insert(tk.END, f"Breathing Rate: {breathing_rate:.1f} breaths/min\n")
    else:
        output_box.insert(tk.END, "Breathing rate not detected\n")

    # Eyebrow analysis
    annotated_img, eyebrow_points = detect_eyebrows(frames[-1])
    thinning_result = analyze_eyebrow_thinning(annotated_img, eyebrow_points)
    if thinning_result:
        thinning, variance = thinning_result
        if thinning:
            output_box.insert(tk.END, f"Eyebrow Thinning Detected (Variance: {variance:.2f})\n")
        else:
            output_box.insert(tk.END, f"Eyebrow Normal (Variance: {variance:.2f})\n")
    else:
        output_box.insert(tk.END, "Eyebrow analysis failed.\n")

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.plot(np.array(timestamps) - timestamps[0], green_signal, label="Green Signal")
    plt.plot(np.array(timestamps) - timestamps[0], detrended, label="Detrended")
    plt.title("Green Channel Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------- GUI Setup -----------------
def start_analysis():
    output_box.delete(1.0, tk.END)
    threading.Thread(target=analyze_and_display, args=(output_box,), daemon=True).start()

root = tk.Tk()
root.title("Health Analyzer (Heart, Breath, Eyebrow)")
root.geometry("600x400")

start_button = tk.Button(root, text="Start Health Analysis", command=start_analysis, font=("Arial", 14), bg="green", fg="pink")
start_button.pack(pady=20)

output_box = tk.Text(root, height=15, width=70, font=("Consolas", 10))
output_box.pack(padx=10, pady=10)

root.mainloop()
