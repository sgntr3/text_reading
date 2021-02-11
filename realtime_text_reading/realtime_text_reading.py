import cv2
import numpy as np 
import pytesseract
from PIL import ImageGrab
import imutils
import reading_function as rf


vid = cv2.VideoCapture(0)

def ss():
    if cv2.waitKey(10) & 0xFF == ord('s'):
        cv2.imwrite(r"realtime_text_reading\ss1.jpg", reading_frame)
        print("screenshot alındı!")

def read():
    if cv2.waitKey(10) & 0xFF == ord('r'):
        rf.metinOku()

while True:
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)
    reading_frame = frame[200:300, 100:300]
    reading_area = cv2.rectangle(frame, (100,200), (300,300), (0, 0, 255), 3)

    ss()    
    read()
    cv2.imshow("mask", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



vid.release()
cv2.destroyAllWindows()