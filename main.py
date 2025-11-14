import cv2
import numpy as np
import face_recognition
import os
from deepface import DeepFace
import pygame
from collections import deque, Counter

# Initialize pygame mixer for beep sound
pygame.mixer.init()
beep_sound = pygame.mixer.Sound("alert.wav")  # ensure this file exists
beep_channel = None  # to track if sound is playing

# Load wanted faces
known_encodings = []
known_names = []

wanted_dir = "wanted_faces"
for file in os.listdir(wanted_dir):
    path = os.path.join(wanted_dir, file)
    if file.lower().endswith(("png", "jpg", "jpeg")):
        img = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(img)
        if enc:
            known_encodings.append(enc[0])
            name = os.path.splitext(file)[0]
            known_names.append(name)
            print(f"Loaded wanted face: {file} -> '{name}'")

# Buffers for stabilizing gender prediction
gender_history = {}

# Function to draw labels (black text on yellow background)
def draw_label(frame, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw, y), (0, 255, 255), -1)  # yellow box
    cv2.putText(frame, text, (x, y - 2), font, scale, (0, 0, 0), thickness)  # black text

# Start webcam
cap = cv2.VideoCapture(0)
print("Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    alert_triggered = False  # reset per frame

    for i, (top, right, bottom, left) in enumerate(face_locations):
        face_enc = face_encodings[i]
        matches = face_recognition.compare_faces(known_encodings, face_enc, tolerance=0.45)
        name = "Unknown"
        color = (0, 255, 0)  # green for unknown

        if True in matches:
            idx = matches.index(True)
            name = known_names[idx]
            color = (0, 0, 255)  # red for wanted
            alert_triggered = True  # wanted triggers alert

        # Scale back up
        top, right, bottom, left = [v * 4 for v in (top, right, bottom, left)]

        # Gender + emotion detection
        try:
            face_img = frame[top:bottom, left:right]
            if face_img.size > 0:
                analysis = DeepFace.analyze(face_img, actions=['gender', 'emotion'], enforce_detection=False, silent=True)
                gender_pred = analysis[0]['dominant_gender']
                emotion = analysis[0]['dominant_emotion']

                # Stabilize gender using history
                if i not in gender_history:
                    gender_history[i] = deque(maxlen=5)  # store last 5 predictions

                gender_history[i].append(gender_pred)
                stable_gender = Counter(gender_history[i]).most_common(1)[0][0]

                # Women show emotions, men only show gender
                if stable_gender.lower() == "woman":
                    draw_label(frame, f"{name} | {stable_gender} | {emotion}", left, top)
                    if emotion.lower() in ["fear", "sad", "angry", "disgust"]:
                        alert_triggered = True
                else:
                    draw_label(frame, f"{name} | {stable_gender}", left, top)
            else:
                draw_label(frame, name, left, top)
        except Exception:
            draw_label(frame, name, left, top)

        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Handle beep sound
    if alert_triggered:
        if beep_channel is None or not beep_channel.get_busy():
            beep_channel = beep_sound.play(-1)  # loop beep
    else:
        if beep_channel is not None and beep_channel.get_busy():
            beep_channel.stop()
            beep_channel = None

    # Show video
    cv2.imshow("Women Safety Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if beep_channel is not None:
    beep_channel.stop()
pygame.mixer.quit()
print("Exited cleanly.")





