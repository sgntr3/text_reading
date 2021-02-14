import sys, os, cv2, pytesseract
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QTextEdit, QLabel, QApplication, QWidget, QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets 
import numpy as np
from PIL import Image
import reading_function as rf




class window(QWidget):

    def __init__(self):
        super().__init__()
        self.init_ui()
    

    def init_ui(self):
        self.buton2 = QPushButton("Cam")
        self.buton3 = QPushButton("Face Detecting")
        self.buton4 = QPushButton("Cat Detecting")
        self.buton5 = QPushButton("Text Reading")
        self.buton6 = QPushButton("Blur")

        vbox = QVBoxLayout()
        vbox.addWidget(self.buton2)
        vbox.addWidget(self.buton3)
        vbox.addWidget(self.buton4)
        vbox.addWidget(self.buton5)
        vbox.addWidget(self.buton6)
        vbox.addStretch()

        hbox = QHBoxLayout()
        hbox.addStretch()
        hbox.addLayout(vbox)
        hbox.addStretch()        

        self.setLayout(hbox)
        self.buton2.clicked.connect(self.cam)
        self.buton3.clicked.connect(self.face_detecting)
        self.buton4.clicked.connect(self.cat_face)
        self.buton5.clicked.connect(self.text_reading)
        self.buton6.clicked.connect(self.blurr)
        self.show()


    def cam(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret,frame = cap.read() #ret = return 
            # # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            cv2.imshow("frame",frame)
    
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def face_detecting(self):
        face = cv2.CascadeClassifier(r"opencv\advanced\falan\cascades\haarcascade_frontalface_default.xml")
        eyes = cv2.CascadeClassifier(r"opencv\advanced\falan\cascades\haarcascade_eye.xml")

        cap = cv2.VideoCapture(0)

        while True:
            ret,frame = cap.read()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            face_detect = face.detectMultiScale(gray,1.3,5)
            for x,y,w,h in face_detect:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,139,120),2)
                roi_gri = gray[y:y+h,x:x+w]
                roi_frame = frame[y:y+h,x:x+w]
            cv2.imshow("Face Cam",frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def cat_face(self):
        cat_face = cv2.CascadeClassifier(r"opencv\advanced\falan\cascades\frontalcatface.xml")
        cap = cv2.VideoCapture(0)

        while True:
            ret,frame = cap.read()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            cat_detecting = cat_face.detectMultiScale(gray,1.3,5)
            for x,y,w,h in cat_detecting:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,139,120),2)
                roi_gri = gray[y:y+h,x:x+w]
                roi_frame = frame[y:y+h,x:x+w]
            cv2.imshow("Cat Cam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
        cap.release()
        cv2.destroyAllWindows()


    pytesseract.pytesseract.tesseract_cmd ="C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
    kaynak = ""

    def text_reading(self):
        vid = cv2.VideoCapture(0)

        def ss():
            if cv2.waitKey(10) & 0xFF == ord('s'):
                cv2.imwrite(r"realtime_text_reading\ss1.jpg", reading_frame)
                print("Screenshot has been captured!")

        def read():
            if cv2.waitKey(10) & 0xFF == ord('r'):
                rf.read()

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



    def blurr(self):
        file_path = QFileDialog.getOpenFileName(self,"Open File",os.getenv("Desktop"))
        img = cv2.imread(file_path[0])

        blur = cv2.GaussianBlur(img,(3,3),2)
        cv2.imshow("blur",blur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


app = QApplication(sys.argv)
window = window()
sys.exit(app.exec_())