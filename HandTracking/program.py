import cv2
from mediapipe.python.solutions import drawing_utils
from mediapipe.python.solutions import hands as Hands

cap = cv2.VideoCapture(0)
mpHands = Hands
hands = mpHands.Hands()
mpDraw = drawing_utils

while True:
    success, image = cap.read()
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            
            for id, lm in enumerate(handLms.landmark):
                h,w,c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id == Hands.HandLandmark.THUMB_TIP:
                    cv2.circle(image, (cx, cy), 30, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image",image)
    if cv2.waitKey(33) == 27:
        break
