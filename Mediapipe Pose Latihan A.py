import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Landmark index
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16

cap = cv2.VideoCapture(0)  # Webcam

while True:
    success, img = cap.read()
    if not success:
        break

    # Konversi warna BGR ke RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Proses pose
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        lm = results.pose_landmarks.landmark

        # Ambil koordinat pundak dan pergelangan
        shoulder_y = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2
        wrist_y_min = min(lm[LEFT_WRIST].y, lm[RIGHT_WRIST].y)

        # Deteksi tangan terangkat
        if wrist_y_min < shoulder_y:
            cv2.putText(img, "Tangan Terangkat", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Deteksi Tangan Terangkat", img)

    # Tekan q untuk keluar
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()