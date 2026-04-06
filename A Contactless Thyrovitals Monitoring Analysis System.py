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

# ---------------------- Bandpass Filter ----------------------
def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    y = filtfilt(b, a, data)
    return y

# ---------------------- EAR (Blink Detection) ----------------------
def eye_aspect_ratio(landmarks, eye_ids, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_ids]
    # Vertical distances
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # Horizontal distance
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    ear = (A + B) / (2.0 * C)
    return ear

# ---------------------- Eyebrow Detection ----------------------
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

# ---------------------- Eyebrow Thinning Analysis ----------------------
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

# ---------------------- Main Analysis ----------------------
def analyze_health(output_box):
    output_box.insert(tk.END, "Capturing video for 15 seconds...\n")
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    frames, timestamps = [], []

    blink_count = 0
    ear_threshold = 0.21
    consecutive_frames = 0

    with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
        while time.time() - start_time < 15:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            timestamps.append(time.time())

            # Process with MediaPipe for blink detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = results.multi_face_landmarks[0].landmark

                # EAR for blink detection (left eye landmarks)
                left_eye_ids = [33, 160, 158, 133, 153, 144]
                ear = eye_aspect_ratio(landmarks, left_eye_ids, w, h)

                if ear < ear_threshold:
                    consecutive_frames += 1
                else:
                    if consecutive_frames >= 2:  # Blink detected
                        blink_count += 1
                    consecutive_frames = 0

    cap.release()

    # ---------------------- Heart & Breath Analysis ----------------------
    if len(frames) < 30:
        output_box.insert(tk.END, "Not enough frames captured.\n")
        return

    green_signal = []
    for frame in frames:
        h, w, _ = frame.shape
        roi = frame[h//8:h//8+h//6, w//3:w//3+w//3]
        green_signal.append(np.mean(roi[:, :, 1]))

    green_signal = np.array(green_signal)
    elapsed_time = timestamps[-1] - timestamps[0]
    fs = len(green_signal) / elapsed_time

    # Detrend
    moving_avg = np.convolve(green_signal, np.ones(5)/5, mode='same')
    detrended = green_signal - moving_avg

    # Filter signals
    hr_signal = bandpass_filter(detrended, 0.8, 2.5, fs)
    br_signal = bandpass_filter(detrended, 0.1, 0.4, fs)

    # Peaks
    hr_peaks, _ = find_peaks(hr_signal, distance=fs/2.5)
    br_peaks, _ = find_peaks(br_signal, distance=fs*2.5)

    heart_rate = 60 / np.mean(np.diff(hr_peaks) / fs) if len(hr_peaks) > 1 else None
    breathing_rate = 60 / np.mean(np.diff(br_peaks) / fs) if len(br_peaks) > 1 else None

    # ---------------------- Eyebrow Analysis ----------------------
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

    # ---------------------- Output ----------------------
    if heart_rate:
        output_box.insert(tk.END, f"Heart Rate: {heart_rate:.1f} bpm\n")
    else:
        output_box.insert(tk.END, "Heart rate not detected\n")

    if breathing_rate:
        output_box.insert(tk.END, f"Breathing Rate: {breathing_rate:.1f} breaths/min\n")
    else:
        output_box.insert(tk.END, "Breathing rate not detected\n")

    output_box.insert(tk.END, f"Blink Count: {blink_count} in 15 sec\n")
    blink_rate = blink_count * 4  # per minute
    output_box.insert(tk.END, f"Blink Rate: {blink_rate} blinks/min\n")

    # ---------------------- Stress & Fatigue ----------------------
    if heart_rate and blink_rate:
        if heart_rate > 100 or blink_rate > 25:
            output_box.insert(tk.END, "⚠ Stress Detected\n")
        else:
            output_box.insert(tk.END, "✅ Stress Normal\n")

        if blink_rate < 10:
            output_box.insert(tk.END, "⚠ Fatigue Detected (low blink rate)\n")
        else:
            output_box.insert(tk.END, "✅ Fatigue Normal\n")

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(np.array(timestamps) - timestamps[0], green_signal, label="Green Signal")
    plt.plot(np.array(timestamps) - timestamps[0], detrended, label="Detrended")
    plt.title("Green Channel Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------- GUI ----------------------
def start_analysis():
    output_box.delete(1.0, tk.END)
    threading.Thread(target=analyze_health, args=(output_box,), daemon=True).start()

root = tk.Tk()
root.title("Health Analyzer (Full Version)")
root.geometry("650x450")

start_button = tk.Button(root, text="Start Health Analysis", command=start_analysis,
                         font=("Arial", 14), bg="green", fg="white")
start_button.pack(pady=20)

output_box = tk.Text(root, height=18, width=75, font=("Consolas", 10))
output_box.pack(padx=10, pady=10)

root.mainloop()