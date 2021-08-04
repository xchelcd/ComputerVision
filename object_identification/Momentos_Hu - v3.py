import cv2
import imutils
import numpy as np
import time

count = 0
i_0 = 0
i_1 = 0
i_2 = 0
i_2 = 0
i_3 = 0
i_4 = 0
i_5 = 0
i_6 = 0
i_7 = 0
i_8 = 0
i_9 = 0
i_a = 0
i_b = 0
imgList = []
promList = []
auxListCount_ = [0,0,0,0,0,0,0,0,0,0,0,0]
img_0 = 0
img_1 = 0
img_2 = 0
img_3 = 0
img_4 = 0
img_5 = 0
img_6 = 0
img_7 = 0
img_8 = 0
img_9 = 0
img_a = 0
img_b = 0
'''
animalList = ['aguila',
              'leon',
              'caballo',
              'girasol',
              'auto',
              'pinguino',
              'colibri',
              'corazon',
              'hongo',
              'tigre',
              'cheeta',
              'elefafnte']
'''
animalList = ['aguila',
              'girasol',
              'auto',
              'hongo',
              'colibri',
              'pinguino',
              'caballo',
              'flecha',
              'corazon',
              'tenis']

def imShow(img):
    i = 0
    for img_ in img:
        cv2.imshow("img_{}".format(i), img_)
        i = i+1

def putText(src, text, loc):
    cv2.putText(src, "{}".format((text)),
                loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,0,255), 1, cv2.LINE_AA)

def prepareImage2(img):
    k = 3
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(img, (k,k))
    _, bin_ = cv2.threshold(blur, 100, 101, cv2.THRESH_BINARY_INV)
    #cv2.imshow("A", bin_)
    canny = cv2.Canny(bin_,100,200)#100,200
    #cv2.imshow("prepareImage", bin_)
    return bin_
    
def prepareImage(img):
    k = 3
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (k,k))
    _, bin_ = cv2.threshold(blur, 100, 101, cv2.THRESH_BINARY_INV)
    #cv2.imshow("A", bin_)
    canny = cv2.Canny(bin_,100,200)#100,200
    #cv2.imshow("prepareImage", bin_)
    return bin_

def improveImgMorph(img):
    #cv2.imshow("o", img)
    k = 5#5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    open_ = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #cv2.imshow("open", open_)
    erode = cv2.erode(open_, kernel, iterations = 1)#2
    #cv2.imshow("erode", erode)
    dilate = cv2.dilate(img, kernel, iterations = 1)#2
    #cv2.imshow("dilate", dilate)
    close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("close", close)
    #cv2.imshow("improveImgMorph", img)
    return erode

def findContours(src, bin_):
    count = 0
    aux = src
    contours, hierarchy  = cv2.findContours(
        bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#SIMPLE
    #print(len(contours))
    numberImage = 0
    for i in range (len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        area = cv2.contourArea(contours[i])
        if area > 1000:
            numberImage = numberImage + 1
	    #######################################################
            cv2.drawContours(src, contours, i, (0,255,0), 1)#,cv2.FILLED)
            count = segmentImage(aux, contours[i], (x,y,w,h), i, count)

def segmentImage(src, img, params, index, count):
    x,y,w,h = params
    rect = cv2.minAreaRect(img)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #cv2.drawContours(src,[box],0,(0,0,255),2)
    a = 10
    roi = src[y+a:y+h+a, x+a:x+w+a]
    #roi = src[y:y+h, x:x+w]
    #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    morph = improveImgMorph(roi)
    bin_ = prepareImage(morph)
    roi = bin_
    
    indexHuMoments = 0
    test_1=checkHuMoments(huMoments(roi))
    test_2=checkHuMoments2(huMoments(roi))
    #print("1->{}\n".format(m,n))
    greater = max(test_1)
    pos = test_1.index(greater)
    putText(src, "{}".format(animalList[pos]), (x,y))
    #putText(src, "{}".format(animalList[test_2]), (x,y+15))
    
    print(test_1)
    button = 0#input("enter")
    if button == '1':
        cv2.imwrite("frame_1.png", frame)
    '''
    test = checkHuMoments(huMoments(roi))
    greater = max(test)
    pos = test.index(greater)
    print(test)
    print("{}->{}->{}".format(greater,pos,animalList[pos]))
    putText(src, "{}".format(animalList[pos]), (x,y))
    '''
    '''
    if huMoments(img_0)[indexHuMoments].round(4) - offset < huMoments(roi)[0].round(4) < huMoments(img_0)[indexHuMoments].round(4) + offset:
        putText(src, "{}".format("aguila"), (x,y))
    elif huMoments(img_1)[indexHuMoments].round(4) - offset < huMoments(roi)[0].round(4) < huMoments(img_1)[indexHuMoments].round(4) + offset:
        putText(src, "{}".format("leon"), (x,y))
    elif huMoments(img_2)[indexHuMoments].round(4) - offset < huMoments(roi)[0].round(4) < huMoments(img_2)[indexHuMoments].round(4) + offset:
        putText(src, "{}".format("caballo"), (x,y))
    elif huMoments(img_3)[indexHuMoments].round(4) - offset < huMoments(roi)[0].round(4) < huMoments(img_3)[indexHuMoments].round(4) + offset:
        putText(src, "{}".format("girasol"), (x,y))
    elif huMoments(img_4)[indexHuMoments].round(4) - offset < huMoments(roi)[0].round(4) < huMoments(img_4)[indexHuMoments].round(4) + offset:
        putText(src, "{}".format("auto"), (x,y))
    elif huMoments(img_5)[indexHuMoments].round(4) - offset < huMoments(roi)[0].round(4) < huMoments(img_5)[indexHuMoments].round(4) + offset:
        putText(src, "{}".format("pinguino"), (x,y))
    elif huMoments(img_6)[indexHuMoments].round(4) - offset < huMoments(roi)[0].round(4) < huMoments(img_6)[indexHuMoments].round(4) + offset:
        putText(src, "{}".format("colibri"), (x,y))
    elif huMoments(img_7)[indexHuMoments].round(4) - offset < huMoments(roi)[0].round(4) < huMoments(img_7)[indexHuMoments].round(4) + offset:
        putText(src, "{}".format("corazon"), (x,y))
    elif huMoments(img_8)[indexHuMoments].round(4) - offset < huMoments(roi)[0].round(4) < huMoments(img_8)[indexHuMoments].round(4) + offset:
        putText(src, "{}".format("hongo"), (x,y))
    elif huMoments(img_9)[indexHuMoments].round(4) - offset < huMoments(roi)[0].round(4) < huMoments(img_9)[indexHuMoments].round(4) + offset:
        putText(src, "{}".format("tigre"), (x,y))
    elif huMoments(img_a)[indexHuMoments].round(4) - offset < huMoments(roi)[0].round(4) < huMoments(img_a)[indexHuMoments].round(4) + offset:
        putText(src, "{}".format("cheeta"), (x,y))
    elif huMoments(img_b)[indexHuMoments].round(4) - offset < huMoments(roi)[0].round(4) < huMoments(img_b)[indexHuMoments].round(4) + offset:
        putText(src, "{}".format("elefante"), (x,y))
    '''
    return count

def checkHuMoments(huMoments):
    count = 0
    auxListSmaller = [float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0)]#,float(0),float(0)]
    auxListSmaller_ = [float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0)]#,float(0),float(0)]
    auxListCount = [0,0,0,0,0,0,0,0,0,0]#,0,0]
    imgList = [img_0,
               img_1,
               img_2,
               img_3,
               img_4,
               img_5,
               img_6,
               img_7,
               img_8,
               img_9
               #, img_a,
               #img_b
               ]
    i = 0
    for hm in huMoments:
        j = 0
        for img in imgList:
            hu = cv2.HuMoments(cv2.moments(img))[i]
            huDif = abs(hm - hu)
            auxListSmaller[j] = huDif
            j = 1 + j
        smaller = min(auxListSmaller,key=lambda x:float(x))
        position = auxListSmaller.index(smaller)
        auxListCount[position] = auxListCount[position] + 1
        i = 1 + i
    #print("método 1\n{}".format(auxListCount))
    #print()
    return auxListCount

def checkHuMoments2(huMoments):
    count = 0
    auxListSmaller = [float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0)]#,float(0),float(0)]
    auxListSmaller_ = [float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0)]#,float(0),float(0)]
    auxListCount = [0,0,0,0,0,0,0,0,0,0]#,0,0]
    imgList = [img_0,
               img_1,
               img_2,
               img_3,
               img_4,
               img_5,
               img_6,
               img_7,
               img_8,
               img_9
               #img_a,
               #img_b
               ]
    i = 0
    for img in imgList:
        j = 0
        total = 0
        for hm in huMoments:
            hu = cv2.HuMoments(cv2.moments(img))[j]
            dif = abs(hm-hu)
            total +=  dif
            j = j + 1
        auxListSmaller[i] = total
        i = i + 1
    smaller = min(auxListSmaller,key=lambda x:float(x))
    position = auxListSmaller.index(smaller)
    return position


def huMoments(img):
    return cv2.HuMoments(cv2.moments(img))


path_0 = "img/copy/"
path_ = "img/new/"
imgName_0 = "img_0"#aguila
imgName_1 = "img_1"#león
imgName_2 = "img_2"#caballo
imgName_3 = "img_3"#girasol
imgName_4 = "img_4"#auto
imgName_5 = "img_5"#pinguino
imgName_6 = "img_6"#colibrí
imgName_7 = "img_7"#corazón
imgName_8 = "img_8"#hongo
imgName_9 = "img_9"#tigre
imgName_a = "img_a"#cheeta
imgName_b = "img_b"#elefafnte
imgName_c = "img_all_0"
imgName_d = "img_all_1"
imgName_t = "img_0_1"#TEST

img_0 = cv2.imread("{}{}.tif".format(path_,imgName_0))
img_1 = cv2.imread("{}{}.tif".format(path_,imgName_1))
img_2 = cv2.imread("{}{}.tif".format(path_,imgName_2))
img_3 = cv2.imread("{}{}.tif".format(path_,imgName_3))
img_4 = cv2.imread("{}{}.tif".format(path_,imgName_4))
img_5 = cv2.imread("{}{}.tif".format(path_,imgName_5))
img_6 = cv2.imread("{}{}.tif".format(path_,imgName_6))
img_7 = cv2.imread("{}{}.tif".format(path_,imgName_7))
img_8 = cv2.imread("{}{}.tif".format(path_,imgName_8))
img_9 = cv2.imread("{}{}.tif".format(path_,imgName_9))
#img_a = cv2.imread("{}{}.tif".format(path_,imgName_a))
#img_b = cv2.imread("{}{}.tif".format(path_,imgName_b))
#img_c = cv2.imread("{}{}.tif".format(path_,imgName_c))
#img_d = cv2.imread("{}{}.tif".format(path_,imgName_d))
#img_t = cv2.imread("{}{}.tif".format(path_0,imgName_t))#TEST

morph = improveImgMorph(img_0)
bin_ = prepareImage(morph)
img_0 = bin_

morph = improveImgMorph(img_1)
bin_ = prepareImage(morph)
img_1 = bin_

morph = improveImgMorph(img_2)
bin_ = prepareImage(morph)
img_2 = bin_

morph = improveImgMorph(img_3)
bin_ = prepareImage(morph)
img_3 = bin_

morph = improveImgMorph(img_4)
bin_ = prepareImage(morph)
img_4 = bin_

morph = improveImgMorph(img_5)
bin_ = prepareImage(morph)
img_5 = bin_

morph = improveImgMorph(img_6)
bin_ = prepareImage(morph)
img_6 = bin_

morph = improveImgMorph(img_7)
bin_ = prepareImage(morph)
img_7 = bin_

morph = improveImgMorph(img_8)
bin_ = prepareImage(morph)
img_8 = bin_

morph = improveImgMorph(img_9)
bin_ = prepareImage(morph)
img_9 = bin_

#morph = improveImgMorph(img_a)
#bin_ = prepareImage(morph)
#img_a = bin_

#morph = improveImgMorph(img_b)
#bin_ = prepareImage(morph)
#img_b = bin_

#imgList = [img_0,img_1,img_2,img_3,img_4,img_5,
#           img_6,img_7,img_8,img_9,img_a,img_b]

imgList = [img_0,img_1,img_2,img_3,img_4,img_5,
           img_6,img_7,img_8,img_9]

'''
print("{}\n{}\n".format("aguila", huMoments(img_0)))#.round(10)))#cheeta
print("{}\n{}\n".format("leon", huMoments(img_1)))#.round(10)))#cheeta
print("{}\n{}\n".format("caballo", huMoments(img_2)))#.round(10)))#cheeta
print("{}\n{}\n".format("girasol", huMoments(img_3)))#.round(10)))#cheeta
print("{}\n{}\n".format("auto", huMoments(img_4)))#.round(10)))#cheeta
print("{}\n{}\n".format("pinguino", huMoments(img_5)))#.round(10)))#cheeta
print("{}\n{}\n".format("colibrí", huMoments(img_6)))#.round(10)))#cheeta
print("{}\n{}\n".format("corazon", huMoments(img_7)))#.round(10)))#cheeta
print("{}\n{}\n".format("hongo", huMoments(img_8)))#.round(10)))#cheeta
print("{}\n{}\n".format("tigre", huMoments(img_9)))#.round(10)))#cheeta
#print("{}\n{}\n".format("cheeta", huMoments(img_a)))#.round(10)))#cheeta
#print("{}\n{}\n".format("elefante", huMoments(img_b)))#.round(10)))#cheeta
'''

#imShow([img_0, img_1, img_2, img_3, img_4, img_5,
#        img_6, img_7, img_8, img_9
#        #, img_a, img_b
#        ])

video = cv2.VideoCapture(1)

frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)
result = cv2.VideoWriter('obj_1.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                        10, size)

while video.isOpened():
    
    _, frame = video.read()
    
    morph = improveImgMorph(frame)
    bin_ = prepareImage(morph)

    findContours(frame, bin_)


    result.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break
    cv2.imshow("Video", frame)
    
video.release()
result.release()
cv2.destroyAllWindows()







