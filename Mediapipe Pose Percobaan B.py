import cv2
import mediapipe as mp
mpose = mp.solutions.pose #inisia media pipe pose
pose = mpose.Pose()
mdraw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0) #video dari webcam
while True:

    success, img = cap.read() #ppembacaan image
    imgrgb= cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #konversi wwarna dari bgr ke rgb
    hasil = pose.process(imgrgb) #ekstraksi dari image
    if hasil.pose_landmarks:
        mdraw.draw_landmarks(img, hasil.pose_landmarks, mpose.POSE_CONNECTIONS) #menggambar koneksi landmark

        for id,im in enumerate(hasil.pose_landmarks.landmark):
            print(id, im.x, im.y) #ekstraksi id, posisi x, posisi y
        cv2.imshow("webcam",img)
        cv2.waitKey(10)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break

cap.release() #Tutup webcam dan jendela tampilan saaat q ditekan
cv2.destroyAllWindows()