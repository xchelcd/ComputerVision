import cv2
import numpy as np
import imutils
import pytesseract as ts
from pytesseract import Output
import os
from matplotlib import pyplot as plt

ts.pytesseract.tesseract_cmd = r'D:\Descargas\Tesseract\tesseract\tesseract'

def toGrayScale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
def toBin(frame):
    _, img = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    return img
def toCanny(frame):
    return cv2.Canny(frame, 100, 150)
def toBlur(frame):
    i=3
    return cv2.blur(frame, (i,i))
def blur(frame):
    for i in range(3,6):
        cv2.imwrite("{}/id_blur_k{}.png".format(version,i),cv2.blur(frame, (i,i)))
        cv2.imwrite("{}/id_2d_k{}.png".format(version,i),  cv2.filter2D(roi,-1,np.ones((5,5),np.float32)/25))
def erodeImage(frame):
    return cv2.erode(frame, (3,3), iterations=2)
def dilateImage(frame):
    return cv2.dilate(frame, (3,3), iterations=2)
def gauss(frame):
    return cv2.GaussianBlur(frame,(5,5),0.35)
def putText(src, text, loc):
    cv2.putText(src,text,loc,cv2.FONT_ITALIC,0.5,(0,0,255),1,cv2.LINE_AA)
def drawRect(frame):
    frameCopy = frame.copy()
    x1,y1 = 100,100
    x2,y2 = 450,325
    roi = frameCopy[y1:y2,x1:x2]
    cv2.rectangle(frame, (x1,y1), (x2,y2), 255, 1)
    return roi
def getImageRect(roi):
    cv2.imwrite("{}/id_org.png".format(version), roi)
    #cv2.imwrite("v{}-id_gray.png".format(version), toGrayScale(roi))
    blur(roi)
def saveImg(roi, version):
    if not os.path.exists(version):    
        os.makedirs(version)
    getImageRect(roi)
def findText(frame):
    return ts.image_to_string(frame)

def camshift(frame_, src, flag):
    frame = frame_.copy()
    x, y, w, h = 300, 200, 275, 125
    track_window = (x, y, w, h)
    roi = src
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv_roi, np.array((50., 60.,5.)), np.array((30.,255.,180.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    #roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1 )
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv2.polylines(frame,[pts],True, 255,2)
    cv2.imshow('img2',img2)
    cv2.imshow('hsv_roi',hsv_roi)
    cv2.imshow('mask',mask)

def findContours(frame, src):
    print("find")

    
v = 4
version = "v{}".format(v)

vAux = "v3"
k=3 #3<k<5

name_O = "id_org"
name_blur = "id_blur_k{}".format(k)#
name_2d = "id_2d_k{}".format(k)

imgName_O = name_O
imgName_blur = name_blur
imgName_2d = name_2d

img_O = cv2.imread("{}/{}.png".format(vAux,imgName_O))
img_blur = cv2.imread("{}/{}.png".format(vAux,imgName_blur))
img_2d = cv2.imread("{}/{}.png".format(vAux,imgName_2d))

#cv2.imshow("img_O", img_O)
#cv2.imshow("img_blur", img_blur)
#cv2.imshow("img_2d", img_2d)

src = img_O
#cv2.imshow("src", src)
#cv2.imshow("bin", toBin(src))
#cv2.imshow("gray", toGrayScale(src))
#cv2.imshow("canny", toCanny(toGrayScale(src)))
              
video = cv2.VideoCapture(0)
    
flag = True
while True:
    _, frame = video.read()

    key = cv2.waitKey(1)
    if key & 0xFF == ord('e'):
        print("Break")
        break
    
    #roi = drawRect(frame)
    if key == ord('s'):
        print("ScreenShot_{}".format(version))
        saveImg(roi, version)
        v = v + 1
        version = "v{}".format(v)
    if key == ord('t'):
        print("Take photo")
    #camshift(frame, src, flag)
    if key == ord('c'):
        copy = frame.copy()
        #cv2.imshow("O", copy)
        copy = toGrayScale(copy)
        _,copy = cv2.threshold(copy, 120,175, cv2.THRESH_BINARY)

        #cv2.imwrite("bin_0.png", copy)
        #cv2.imshow("test", copy)

        
        copy = cv2.morphologyEx(copy, cv2.MORPH_CLOSE, None, iterations=10)
        copy = cv2.dilate(copy, (5,5), iterations=5)
        #cv2.imshow("erode", copy)
        #cv2.imwrite("close-dialte_0.png", copy)
        
        #kernel = np.ones((3,3),np.uint8)
        #frame_ = frame.copy()
        #cannyFrame = toCanny(toBlur(toGrayScale(src)))#frame
        
        #cannyFrame  = cv2.dilate(cannyFrame,(5,5),iterations=15)
        #cannyFrame  = cv2.erode(cannyFrame,(5,5),iterations=15)
        #cv2.imshow("filter_", cannyFrame)
        #cannyFrame  = cv2.dilate(cannyFrame,(5,5),iterations=3)
        #cannyFrame = cv2.morphologyEx(cannyFrame, cv2.MORPH_CLOSE, (5,5))
        #cv2.imshow("filter", cannyFrame)
        
        contours, _ = cv2.findContours(copy,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 2000:
                epsilon = 0.051*cv2.arcLength(c,True)
                approx = cv2.approxPolyDP(c,epsilon,True)
                if len(approx) == 4:
                    cv2.drawContours(frame,[approx],0,(0,255,0),5)
                    x,y,w,h = cv2.boundingRect(c)
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    roi = frame[y:y+h, x:x+w]
                    h, w, c = roi.shape
                    #print(roi.shape)
                    #print(w)
                    #print(h)
                    p1 = (int(0.28*w),int(0.2*h))
                    p2 = (int(0.6*w),int(0.4*h))
                    cv2.rectangle(roi,p1,p2,(0,0,255),3)
                    cv2.imshow("roi", roi)
                    #cv2.imwrite("roi_0.png", roi)
                    #print("test")
                    #print(p1,p2)
                    #print(p1[1])
                    #print(p1[0])
                    #print(p2[1])
                    #print(p2[0])
                    #ones = np.ones((p1[1],p2[1]))
                    #cv2.imshow("ones", ones)
                    tesseract = roi[int(0.2*h):int(0.4*h),int(0.28*w):int(0.6*w)]
                    original = tesseract
                    gray = toGrayScale(tesseract)
                    gauss = gauss(toGrayScale(tesseract))
                    cv2.imshow("original", original)
                    cv2.imshow("gauss", gauss)
                    cv2.imshow("gray", gray)

                    
                    #name = findText(tesseract)
                    print("Original:{}".format(findText(original)))
                    print("Gray:{}".format(findText(gray)))
                    print("Gauss:{}".format(findText(gauss)))
            #cv2.drawContours(frame,[c],-1,(0,0,255),3)

        #cv2.imshow("Frame_", frame_)
        #cv2.imshow("cannyFrame", cannyFrame)
        cv2.waitKey(1)

    flag = False
    cv2.imshow("video", frame)
    #cv2.imshow("canny", toCanny(toGrayScale(frame)))
    #cv2.imshow("roi", roi)


cv2.waitKey(0)
video.release()
cv2.destroyAllWindows()

