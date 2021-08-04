import os
import cv2

print("to save images...")

root = 'data/'
#root = 'data_test/'

rockData = 'rock'
paperData = 'paper'
scissorsData = 'scissors'
okData = 'ok'
okntData = 'oknt'

dataPosition = 0
dataName = [rockData,paperData,scissorsData,okData,okntData]
#dataName = paperData
#dataName = scissorsData
#dataName = okData
#dataName = okntData

path = root + dataName[dataPosition]

#if not os.path.exists(path):
#    os.makedirs(path)

p1 = (330,50)
p2 = (630,380)
count = 0
imgSize = (100,100)
totalImg = 500

video = cv2.VideoCapture(0)

flag = False
string = None

while True:
    _, frame = video.read()
    frame = cv2.flip(frame,1)
    
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    copy = frame.copy()

    cv2.rectangle(frame,p1,p2,(255,255,255),3)

    roi = copy[p1[1]:p2[1], p1[0]:p2[0]]
    imgResized = cv2.resize(roi,imgSize,interpolation=cv2.INTER_CUBIC)

    if flag:
        path = root + dataName[dataPosition]
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(path + '/{}_{}.jpg'.format(dataName[dataPosition],count),imgResized)
        count += 1
        
    key = cv2.waitKey(1)
    if count == totalImg:
        count = 0
        flag = False
        print("finished {}.".format(dataName[dataPosition]))
        print("press 's' to save {}".format(dataName[dataPosition]))
        if dataPosition == 4: break
        dataPosition += 1
        #break;
    if key == ord('s'):
        flag = True
    if key == 27:
        print("interrupted")
        break

    cv2.putText(frame,"{}: {}".format(dataName[dataPosition],count),(10,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
    if count == 0: string = "press 's' to save"
    else: string = "saving"
    cv2.putText(frame,"{} {}".format(string,dataName[dataPosition]),(10,60),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
    cv2.imshow("frame",frame)
    #cv2.imshow("roi",roi)
    #cv2.imshow("imgResized",imgResized)

video.release()
cv2.destroyAllWindows()
    
