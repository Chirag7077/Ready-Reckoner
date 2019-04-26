import numpy as np
import cv2
import matplotlib.pyplot as plt


HAAR_CASCADE = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    cv2.flip(frame, flipCode=-1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = HAAR_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
    cv2.imshow('Face Detection', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()


# Useful Links
#https://github.com/frankenstein32/Real-TIme-Face-Recognition/blob/master/Face_Recognition.py
