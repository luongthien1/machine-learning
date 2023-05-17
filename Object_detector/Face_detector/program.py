import cv2
import numpy
import tensorflow
from keras.models import load_model as loadModel

loaded_model = loadModel("Object_detector/Face_detector/facetracking/")
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, camera_frame = cap.read()
    rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
    frame = tensorflow.image.resize(rgb, (120,120)) 
    predict = loaded_model.predict(numpy.expand_dims(frame/255,0), verbose = 0)
    predict_coord = predict[1][0]

    if predict[0][0] > 0.5:
        cv2.rectangle(camera_frame, tuple(numpy.multiply(predict_coord[:2], [640,480]).astype(int)),
                        tuple(numpy.multiply(predict_coord[2:], [640,480]).astype(int)), (255,0,0), 2)
        cv2.rectangle(camera_frame, 
                      tuple(numpy.add(numpy.multiply(predict_coord[:2], [640,480]).astype(int), 
                                    [0,-30])),
                      tuple(numpy.add(numpy.multiply(predict_coord[:2], [640,480]).astype(int),
                                    [80,0])), 
                            (255,0,0), -1)
        cv2.putText(camera_frame, f'face {predict[0][0]}', tuple(numpy.add(numpy.multiply(predict_coord[:2], [640,480]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow("frame", camera_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()