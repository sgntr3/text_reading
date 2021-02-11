import cv2
import numpy as np
import pytesseract
import imutils

pytesseract.pytesseract.tesseract_cmd ="C:\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"


def metinOku():
    img = cv2.imread(r"realtime_text_reading\ss1.jpg")
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #filtered = cv2.bilateralFilter(gray, 5, 500, 500)
    edged = cv2.Canny(gray, 30, 200)


    contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(contours) 
    cnts = sorted(cnts, key= cv2.contourArea, reverse=True)[:10]
    screen = 0

    for c in cnts:
        epsilon = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c, 0.02 * epsilon, True)
        if len(approx) == 4:
            screen = approx
            break

    mask = np.zeros(img.shape[0:2], np.uint8)


    new_img = cv2.drawContours(mask, [screen], 0, (255, 255, 255), -1) 
    new_img = cv2.bitwise_and(img, img, mask= mask)

    (x,y) = np.where(mask == 255) 
    (top_x, top_y) = (np.min(x), np.min(y))
    (bottom_x, bottom_y) = (np.max(x), np.max(y))
    cropped = img[top_x: bottom_x + 1, top_y: bottom_y +1]



    cv2.waitKey(0)

    text = pytesseract.image_to_string(cropped, lang="eng")
    print("text: ", text)

    if len(text) < 1:
        print("Metin Okunamadı Tekrar SS Alıp Okutun")
