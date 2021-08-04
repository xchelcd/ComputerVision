import cv2
import numpy as np
import time
import imutils
import mediapipe as mp
#import ctypes
from datetime import datetime
import os

#user32 = ctypes.windll.user32
#screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

virtualDrawing = 'virtual_drawing-'
imgRoot = 'icons'


#Rango mínimo y máximo del color HSV que se desea detectar en el puntero
minColorHSV = np.array([70,150,150])#40
maxColorHSV = np.array([97,255,255])#75

#Valor actual y valor previo del puntero para realizar los trazos
pointer = None
prePointer = None

'''
Tiempo inicial y tiempo final para saber la duración en (s) del puntero
en las áreas donde se pueden seleccionar
'''
t1 = None
t2 = None
timeFlag = True

#Matriz auxiliar del tamaño de frame donde se pintan todos los trazos
canva = None
'''
Frame auxiliar donde NO se van a pintar los recuadros de opciones
y se va a guardar únicamente lo que muestra la cámara y los trazos
hechos por el usuario
'''
ss = None
ssToSave = None

rawCols = 640
rawRows = 480

#Puntos de cada opción seleccionable
colorP1 = (0,0)
colorP2 = (50,50)
thicknessP1 = (51,0)
thicknessP2 = (101,50)
eraseP1 = (102,0)
eraseP2 = (152,50)
figureP1 = (rawCols-52,2)#top|left
figureP2 = (rawCols-2,52)
backP1 = (0,rawRows-50)#bottom|left
backP2 = (50,rawRows)
aceptP1 = (rawCols-50,rawRows-50)#bottom|right
aceptP2 = (rawCols,rawCols)
ssP1 = (rawCols-52,50+3)
ssP2 = (rawCols-2,100+3)
recordP1 = (0,51)
recordP2 = (50,101)

imgOptionSize = 50

#Banderas y valores de cada 'menú'. Sólo uno puede ser verdadero
mainMenu = True
mainMenuValue = '0'
colorMenu = False
colorMenuValue = '1'
thicknessMenu = False
thicknessMenuValue = '2'
eraseMenu = False
eraseMenuValue = '3'
figureMenu = False
figureMenuValue = '4'
ssMenu = False
ssMenuValue = '7'
recordMenu = False
recordMenuValue = '8'

aceptValue = '5'
backValue = '6'

#mainMenu = True
#secondaryMenu = False
#auxMenu = False

#Valores para realizar los trazos
maxValueThickness = 50
thickness = 5
colorR = 255#173
colorG = 255#35
colorB = 255#233
r,g,b = None,None,None

prepareSS = False

cFigure = None

eraseFlag = False

isRecording = False
startRecord = False
result = None

#TEST
def showRoi(frame):
    cols=frame.shape[1]#640
    rows=frame.shape[0]#480

    p1 = (cols//5,rows//5)
    p2 = (4*cols//5,4*rows//5)
    
    roi = frame[p1[1]:p2[1],p1[0]:p2[0]]#detectar contornos y buscar puntos
    cv2.imshow("roi",roi)

'''
Cada vez que pasa 1s en cada opción se llama está función para
cambiar las banderas. Recibe el valor de menú correspondiente
y le asigna verdadero o falso a cada menú
'''
def selectMenu(key):
    #if key == ord('1') or ord('2') or ord('3') or ord('0') or ord('4') or ord('5') or ord('6'):
    if key == ord(mainMenuValue) or ord(colorMenuValue) or ord(thicknessMenuValue):# or ord(figureMenuValue) or ord(backValue) or ord(aceptValue):
        global mainMenu
        global colorMenu
        global thicknessMenu
        global figureMenu

        global eraseMenu
        global eraseFlag
        global ssMenu
        ##global secondaryMenu
        ##global auxMenu

        global recordMenu
        global startRecord
        global isRecording
        
        ##mainMenu = key == ord('1')
        ##secondaryMenu = key == ord('2')
        ##auxMenu = key == ord('3')
        #print(mainMenuValue,aceptValue,backValue)
        mainMenu = key == ord(mainMenuValue)# or ord(aceptValue)or ord(backValue)
        colorMenu = key == ord(colorMenuValue)
        thicknessMenu = key == ord(thicknessMenuValue)
        figureMenu = key == ord(figureMenuValue)
        if key == ord(eraseMenuValue):
            eraseFlag = True
            mainMenu = True
            eraseMenu = True
        if key == ord(ssMenuValue):
            mainMenu = True
            ssMenu = True
        if key == ord(recordMenuValue):
            mainMenu = True
            recordMenu = True
            startRecord = not startRecord
            #print(isRecording)
            if isRecording:
                isRecording = False
    pass

#Dibuja todas las opciones del menú principal
def mainMenuSelected(frame):
    cols=frame.shape[1]#640
    rows=frame.shape[0]#480

    

    #selectColor,selectThickness,erase,findImage
    #cv2.rectangle(frame,(2,2),(50,50),(255,255,255),3)
    #cv2.rectangle(frame,(55,2),(103,50),(255,255,255),3)
    #cv2.rectangle(frame,(108,2),(156,50),(255,255,255),3)
    #cv2.rectangle(frame,(cols-50,2),(cols-2,50),(255,255,255),3)
    #cv2.rectangle(frame,ssP1,ssP2,(255,255,255),3)

    putImages(frame)

    #TEST
    #selectRGBColor(frame)
    #selectThickness(frame)
    pass
def putImages(frame):
    #aceptName = 'acept.jpeg'
    #cancelName = 'cancel.jpeg'
    colorRGBName = 'color_RGB.jpg'
    copyName = 'copy.jpeg'
    eraseName = 'erase.png'
    ssName = 'ss.jpg'
    thicknessName = 'thickness.png'
    saveRecordingName = 'savevideo.jpg'
    

    #aceptImg = cv2.imread("{}/{}".format(imgRoot,aceptName))
    #cancelImg = cv2.imread("{}/{}".format(imgRoot,cancelName))
    colorRGBImg = cv2.imread("{}/{}".format(imgRoot,colorRGBName))
    copyImg = cv2.imread("{}/{}".format(imgRoot,copyName))
    eraseImg = cv2.imread("{}/{}".format(imgRoot,eraseName))
    ssImg = cv2.imread("{}/{}".format(imgRoot,ssName))
    thicknessImg = cv2.imread("{}/{}".format(imgRoot,thicknessName))
    saveRecordingImg = cv2.imread("{}/{}".format(imgRoot,saveRecordingName))

    thicknessImg = cv2.resize(thicknessImg,(imgOptionSize,imgOptionSize))
    #aceptImg = cv2.resize(aceptImg,(imgOptionSize,imgOptionSize))
    #cancelImg = cv2.resize(cancelImg,(imgOptionSize,imgOptionSize))
    colorRGBImg = cv2.resize(colorRGBImg,(imgOptionSize,imgOptionSize))
    copyImg = cv2.resize(copyImg,(imgOptionSize,imgOptionSize))
    eraseImg = cv2.resize(eraseImg,(imgOptionSize,imgOptionSize))
    ssImg = cv2.resize(ssImg,(imgOptionSize,imgOptionSize))
    thicknessImg = cv2.resize(thicknessImg,(imgOptionSize,imgOptionSize))
    saveRecordingImg = cv2.resize(saveRecordingImg,(imgOptionSize,imgOptionSize))
    
    frame[colorP1[1]:colorP2[1],colorP1[0]:colorP2[0]] = colorRGBImg
    frame[thicknessP1[1]:thicknessP2[1],thicknessP1[0]:thicknessP2[0]] = thicknessImg
    frame[eraseP1[1]:eraseP2[1],eraseP1[0]:eraseP2[0]] = eraseImg
    frame[figureP1[1]:figureP2[1],figureP1[0]:figureP2[0]] = copyImg
    #frame[backP1[1]:backP2[1],backP1[0]:backP2[0]] = cancelImg
    #frame[aceptP1[1]:aceptP2[1],aceptP1[0]:aceptP2[0]] = aceptImg
    frame[ssP1[1]:ssP2[1],ssP1[0]:ssP2[0]] = ssImg
    frame[recordP1[1]:recordP2[1],recordP1[0]:recordP2[0]] = saveRecordingImg

    pass
    
#Dibuja el slider de thickness
def thicknessSelected(frame):
    global pointer
    global thickness
    global colorB,colorG,colorR
    separate = 5
    height = 70
    distX = 70
    
    cols=frame.shape[1]#640
    rows=frame.shape[0]#480
    
    p1 = (distX,-(height//2)+(rows//2))
    p2 = (cols-distX,height//2+rows//2)

    cv2.rectangle(frame,p1,p2,(colorB,colorG,colorR),thickness)

    cv2.line(frame,(p1[0],p2[1]-height//2),((((thickness*(cols-2*distX))//maxValueThickness)+70),p2[1]-height//2),(colorB,colorG,colorR),5)

    putText(frame,'1',(50,5+rows//2),(255,255,255))
    putText(frame,maxValueThickness,(cols-50-12,5+rows//2),(255,255,255))

    aceptName = 'acept.jpeg'
    aceptImg = cv2.imread("{}/{}".format(imgRoot,aceptName))
    aceptImg = cv2.resize(aceptImg,(imgOptionSize,imgOptionSize))
    frame[aceptP1[1]:aceptP2[1],aceptP1[0]:aceptP2[0]] = aceptImg

    putText(frame,thickness,(cols//2,-15+rows//2),(255,255,255))
    if pointer is not None:
        val = ((pointer[0]-70)*100//(cols-(2*distX)))*maxValueThickness//100
        if p1[0] < pointer[0] < p2[0]+5 and p1[1] < pointer[1] < p2[1]:
            if val == 0: val = 1
            thickness = val
            np.save('data',[colorB,colorG,colorR,thickness])
    pointer = None
    #aceptColorRGB
    cv2.rectangle(frame,(cols-50,-52+rows),(cols-2,rows-2),(255,255,255),3)
    pass
#Dibuja los 3 sliders de los colores RGB
def rgbColorSelected(frame):
    global pointer
    global colorR,colorG,colorB
    global thickness
    
    separate = 5
    height = 70
    distX = 70
    
    cols=frame.shape[1]#640
    rows=frame.shape[0]#480

    p1R = (distX,-separate-height-(height//2)+rows//2)
    p2R = (cols-distX,-separate-(height//2)+rows//2)
    
    p1G = (distX,-height//2+rows//2)
    p2G = (cols-distX,height//2+rows//2)
    
    p1B = (distX,separate+height//2+rows//2)
    p2B = (cols-distX,separate+height+(height//2)+rows//2)
    
    cv2.rectangle(frame,p1R,p2R,(0,0,255),3)
    cv2.rectangle(frame,p1G,p2G,(0,255,0),3)
    cv2.rectangle(frame,p1B,p2B,(255,0,0),3)
    

    putText(frame,'0',(50,-height+rows//2),(0,0,255))
    putText(frame,'255',(cols-50-12,-height+rows//2),(0,0,255))
    
    putText(frame,'0',(50,5+rows//2),(0,255,0))
    putText(frame,'255',(cols-50-12,5+rows//2),(0,255,0))
    
    putText(frame,'0',(50,12+height+rows//2),(255,0,0))
    putText(frame,'255',(cols-50-12,12+height+rows//2),(255,0,0))
    
    putText(frame,colorR,(cols//2,-height-separate-15-3+rows//2),(0,0,255))
    putText(frame,colorG,(cols//2,-15+rows//2),(0,255,0))
    putText(frame,colorB,(cols//2,+separate+height-3-15+rows//2),(255,0,0))

    cv2.line(frame,(p1R[0],p2R[1]-height//2),((((colorR*(cols-2*distX))//255)+70),p2R[1]-height//2),(0,0,255),5)
    cv2.line(frame,(p1G[0],p2G[1]-height//2),((((colorG*(cols-2*distX))//255)+70),p2G[1]-height//2),(0,255,0),5)
    cv2.line(frame,(p1B[0],p2B[1]-height//2),((((colorB*(cols-2*distX))//255)+70),p2B[1]-height//2),(255,0,0),5)

    cv2.circle(frame,(cols//2,rows-height),25,(colorB,colorG,colorR),-1)

    aceptName = 'acept.jpeg'
    aceptImg = cv2.imread("{}/{}".format(imgRoot,aceptName))
    aceptImg = cv2.resize(aceptImg,(imgOptionSize,imgOptionSize))
    frame[aceptP1[1]:aceptP2[1],aceptP1[0]:aceptP2[0]] = aceptImg

    #aceptColorRGB
    cv2.rectangle(frame,(cols-50,-52+rows),(cols-2,rows-2),(255,255,255),3)
    
    if pointer is not None:
        val = ((pointer[0]-70)*100//(cols-(2*distX)))*255//100
        if p1R[0] < pointer[0] < p2R[0]+5 and p1R[1] < pointer[1] < p2R[1]:
            colorR = val
        elif p1G[0] < pointer[0] < p2G[0]+5 and p1G[1] < pointer[1] < p2G[1]:
            colorG = val
        elif p1B[0] < pointer[0] < p2B[0]+5 and p1B[1] < pointer[1] < p2B[1]:
            colorB = val
        np.save('data',[colorB,colorG,colorR,thickness])
    pass
#Resume la llamada a la función de opencv
def putText(src, text, loc, color):
    cv2.putText(src, "{}".format((text)),
                loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1, cv2.LINE_AA)

#Dibuja la región donde se va a buscar la figura que se desee trazar
def figureMenuSelected(frame):
    global pointer
    copy = frame.copy()
    
    cols=frame.shape[1]#640
    rows=frame.shape[0]#480

    #guardar roi
    #cv2.rectangle(frame,(cols-50,2),(cols-2,50),(255,255,255),3)
    cv2.rectangle(frame,(2,rows-52),(52,rows-2),(255,255,255),3)#back

    cancelName = 'cancel.jpeg'
    cancelImg = cv2.imread("{}/{}".format(imgRoot,cancelName))
    cancelImg = cv2.resize(cancelImg,(imgOptionSize,imgOptionSize))
    frame[backP1[1]:backP2[1],backP1[0]:backP2[0]] = cancelImg
    putText(frame,'Press "s" to save',(50,50),(0,255,0))

    p1 = (cols//5,rows//5)
    p2 = (4*cols//5,4*rows//5)

    #if guardarRoiBbutton->saveImage(takeScreenShot)
    #t2-t1 > 2s -> 'clicked'
    cv2.rectangle(frame,p1,p2,(0,0,255),3)
    pass
'''
Obtiene la figura encontrada
Se usaron algoritmos similares a las adas de segmentación
'''
def getImage(frame):
    global cFigure
    
    kernel = np.ones((5,5),np.uint8)
    
    cols=frame.shape[1]#640
    rows=frame.shape[0]#480

    cFigure = np.zeros_like(frame)

    p1 = (cols//5,rows//5)
    p2 = (4*cols//5,4*rows//5)
    
    roi = frame[p1[1]:p2[1],p1[0]:p2[0]]
    #DEBUG
    #cv2.imshow("roi",roi)
    
    grayScale = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(grayScale,180,255,cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    canny = cv2.Canny(opening,100,150)

    #img = canny
    #DEBUG
    #cv2.imshow("img",canny)
    saveFigure(canny,cFigure)
    pass
#Busca el contorno deseado y lo imprime en pantalla
def saveFigure(roi,cFigure):
    copy = cFigure.copy()

    contours, hierarchy = cv2.findContours(roi,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    #TEST
    #cv2.drawContours(test,contours,0,(0,0,255), 1)
    if (len(contours)) > 0:
        if (len(contours[0])) > 0:
            for i in range (len(contours[0])):
                c = contours
                #DEBUG
                #cv2.drawContours(roi,c,-1,(colorB,colorG,colorR),thickness)
                if i+1 < (len(contours[0])):
                    p1 = c[0][i][0]
                    p2 = c[0][i+1][0]
                    cv2.line(cFigure,(p1[0],p1[1]),(p2[0],p2[1]),(255,0,0),3)
                    
    else:
        #Avisar que no se detectó ningún contorno
        print("Sin contorno")
    #DEBUG
    #cv2.imshow("cFigure",cFigure)
    
    global canva
    canva = cv2.add(canva,copy)
    
    
    #cFigure = np.zeros_like(cFigure)
    #cFigure = copy
    #cv2.imshow("canva",canva)
    #DEBUG
    #cv2.imshow("roi",roi)
    pass

#Convierte el lápiz en borrador
def eraseMenuSelected():
    global colorR,colorG,colorB
    global r,g,b
    global eraseFlag

    if pointer is None: return
    
    if (pointer[0] < eraseP1[0] or pointer[0] > eraseP2[0]) or pointer[1] > eraseP2[1]:
        #print(eraseP1,eraseP2)
        if eraseFlag:
            #putImageToDraw
            #print("Drawing")
            eraseFlag = False
            if colorR == 0 and colorG == 0 and colorB == 0:
                if r is None or g is None or b is None:
                    getData()
                else:
                    colorR,colorG,colorB = r,g,b
            else:
                r,g,b = colorR,colorG,colorB
                colorR,colorG,colorB = 0,0,0 
        else:
            #putImageToErase
            #print("Erasing")
            pass
            pass
    pass

#Busca y dibuja el puntero para hacer los trazos
def findPointer(frame, shape):
    global prePointer
    global pointer

    kernel = np.ones((5,5),np.uint8)
    
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,minColorHSV,maxColorHSV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 3)
    mask = cv2.dilate(mask,kernel,iterations = 5)
    #DEBUG
    #cv2.imshow("debug", mask)

    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours,key=cv2.contourArea,reverse=True)[:1]

    mask = cv2.erode(mask,kernel,iterations=2)
    #DEBUG
    #cv2.imshow("debug", mask)

    if (len(contours)) == 0:
        prePointer = None
        pointer = None

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        pointer = ((x+(w//4),y+(h//5)))
        cv2.circle(frame,pointer,1,(colorB,colorG,colorR),thickness)
        cv2.circle(frame,pointer,thickness,(255,255,255),2)
        #DEBUG
        #putText(frame,"{}".format(pointer),(x,y),(0,0,0))
    pass
#Devuelve la opción donde se encuentra el puntero
def positionPointer(pointer):
    red = (0,0,255)
    p1 = None
    p2 = None
    val = None
    if pointer[0] < colorP2[0] and pointer[1] < colorP2[1]:
        p1,p2 = colorP1,colorP2
        val = ord(colorMenuValue)
    elif thicknessP1[0] < pointer[0] < thicknessP2[0] and pointer[1] < thicknessP2[1]:
        p1,p2 = thicknessP1,thicknessP2
        val = ord(thicknessMenuValue)
    elif pointer[0] > figureP1[0] and pointer[1] < figureP2[1]:
        p1,p2 = figureP1,figureP2
        val = ord(figureMenuValue)
    elif pointer[0] < backP2[0] and pointer[1] > backP1[1]:
        p1,p2 = backP1,backP2
        val = ord(mainMenuValue)
    elif pointer[0] > aceptP1[0] and pointer[1] > aceptP1[1]:
        p1,p2 = aceptP1,aceptP2
        val = ord(mainMenuValue)
    elif eraseP1[0] < pointer[0] < eraseP2[0] and pointer[1] < eraseP2[1]:
        p1,p2 = eraseP1,eraseP2
        val = ord(eraseMenuValue)
    elif pointer[0] > ssP1[0] and (ssP1[1] < pointer[1] < ssP2[1]):
        p1,p2 = ssP1,ssP2
        val = ord(ssMenuValue)
    elif recordP1[0] < pointer[0] < recordP2[0] and recordP1[1] < pointer[1] < recordP2[1]:
        p1,p2 = recordP1,recordP2
        val = ord(recordMenuValue)
    else: val = None
    if p1 is not None or p2 is not None:
        cv2.rectangle(frame,p1,p2,red,1)
    return val
#Crea los trazos de acuerdo a pointer y prePointer
def drawing(frame):
    global prePointer
    global canva
    global ss

    umbral = 15

    flag = True

    if pointer is not None:
        if (pointer[0] < eraseP2[0] and pointer[1] < eraseP2[1]) or (pointer[0] > ssP1[0] and pointer[1] < ssP2[1]):              
            flag = False
    
    if prePointer is None: prePointer = pointer
    else:
        #if colorB < umbral or colorG < umbral or colorR < umbral:
        if flag:
            #print(colorB,colorG,colorR)
            canva = cv2.line(
                canva,prePointer,pointer,(colorB,colorG,colorR),thickness)

    copy = cv2.cvtColor(canva,cv2.COLOR_BGR2GRAY)
    _,copy = cv2.threshold(copy,10,255,cv2.THRESH_BINARY)
    copy = cv2.bitwise_not(copy)
    frame = cv2.bitwise_and(frame,frame,mask=copy)
    frame = cv2.add(frame,canva)
    prePointer = pointer

    #To save SS
    ss = cv2.bitwise_and(ss,ss,mask=copy)
    ss = cv2.add(ss,canva)

    global ssToSave
    ssToSave = ss
    
    #cv2.imshow("canva",canva)
    #cv2.imshow("ss",ss)
    #cv2.line(canva,(0,0),pointer,(colorB,colorG,colorR),thickness)
    return frame

def ssMenuSelected(frame):
    ssMenu = False
    #print("ssMenuSelected")
    
    global ss
    global ssToSave
    
    #if ss is None: return 0

    cv2.rectangle(frame,ssP1,ssP2,(0,0,255),5)

    #DEBUG
    #cv2.imshow("ss",ssToSave)

    now = datetime.now()
    date = now.strftime("%d-%m-%Y_%H-%M-%S")
    #print(dt_string)
    
    path = "ss"
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(path + '/{}{}.jpg'.format(virtualDrawing,date),ssToSave)

    #DEBUG
    #print(path + '/{}{}.jpg'.format(virtualDrawing,date))
    pass
def getData():
    global colorB,colorG,colorR,thickness
    try:
        data = np.load('data.npy')
        if data is not None or (len(data)) > 0:
            colorB,colorG,colorR = int(data[0]),int(data[1]),int(data[2])
            thickness = int(data[3])
    except IOError:
        pass
    except ValueError:
        pass

def savingRecord(frame, video):
    global result
    global isRecording
    global startRecord
    path = 'video/'
    date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    #virtualDrawing
    if startRecord:
        if not os.path.exists(path):
            os.makedirs(path)
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        size = (frame_width, frame_height)
        #print("{}{}{}".format(path,virtualDrawing,date))
        if result is None:
            result = cv2.VideoWriter("{}{}{}.avi".format(path,virtualDrawing,date),
                                 cv2.VideoWriter_fourcc(*'MJPG'),10, size)
        else:
            #print("end record")
            result.release()
            result = None
        isRecording = True
        startRecord = False
        #print("startRecord")
    #else: print("1")
    if result is not None:
        if isRecording:
            #print("recording")
            putText(frame,'rec',(cols//2-35,rows-5),(0,0,255))
            cv2.circle(frame,(cols//2-40,rows-8),3,(0,0,255),-1)
            result.write(ssToSave)
            #DEBUG
            #cv2.imshow("rec",ssToSave)
        else:
            #print("end record")
            result.release()
            result = None
    #else: print("2")
################################## Main #######################################


video = cv2.VideoCapture(0)

#Trata de obtener los datos guardados de los colores y el thickness
getData()

#window_name = "frame"
#cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


#i = 0#DEBUG
'''
min_detection_confidence=0.9
'''
with mp_hands.Hands(max_num_hands=2,
                    min_detection_confidence=0.8) as hands:
    while True:
        #i = i+1#DEBUG
        ret, frame = video.read()
        frame = cv2.flip(frame,1)
        #frame = imutils.resize(frame,screensize[1])
        
        copy = frame.copy()
        ss = copy
        key = cv2.waitKey(1)
    
        cols = frame.shape[1]#640
        rows = frame.shape[0]#480
    
        if canva is None:
            #ss = copy
            canva = np.zeros_like(frame)

        #findPointer
        findPointer(frame,(cols,rows))

        '''
        Corrobora si se encuentra adentro de un recuadro de alguna
        opción y mide el tiempo que ha estado ahí para saber si
        se selecciona o no
        '''
        if pointer is not None:
            posPointer = positionPointer(pointer)
            if posPointer is not None:
                if timeFlag:
                    t1 = time.time()
                    timeFlag = False
                t2 = time.time()
                if t1 is not None and t2 is not None:
                    tInside = t2-t1
                    putText(frame,round(tInside,2),(cols//2,25),(255,255,255))
                    if tInside > 1:
                        prePointer = None
                        if pointer[0] > eraseP1[0] and pointer[0] < eraseP2[0] and pointer[1] < eraseP2[1]:
                            if tInside > 3:
                                canva = np.zeros_like(frame)
                        else:
                            t1 = None
                            t2 = None
                        
                        selectMenu(posPointer)
            else:
                timeFlag = True

        '''Detecta si hay alguna mano dentro del cuadro de la cámara'''
        results = hands.process(frame)
        if results.multi_hand_landmarks is not None:
            for hands_landmarks in results.multi_hand_landmarks:
                if results.multi_handedness is not None:
                    obj = results.multi_handedness[0]
                    hand = obj.classification[0].index
                    

                    #DEBUG
                    xI = int(hands_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*cols)
                    yI = int(hands_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*rows)

                    xM = int(hands_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x*cols)
                    yM = int(hands_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y*rows)
                    
                    #print((xI,yI),(xM,yM))
                    
                    if hand == 0:
                        #print(obj)
                        #print((xI,yI),(xM,yM))
                        #print(i)#DEBUG
                        cv2.circle(frame,(25,125),20,(0,0,255),-1)
                        #mp_drawing.draw_landmarks(
                        #    frame,hands_landmarks,mp_hands.HAND_CONNECTIONS)
                        pointer = None
                        prePointer = None

        '''
        Corrobora donde se encuentra el puntero y llama a las funciones
        correspondientes para aparecer el menú
        '''
        
        if key == ord(mainMenuValue) or mainMenu:
            #print(eraseMenu,ssMenu)
            if key == ord(eraseMenuValue) or eraseMenu: eraseMenuSelected()
            if key == ord(ssMenuValue) or ssMenu:
                ssMenuSelected(frame)
                ssMenu = False
            if key == ord(recordMenuValue) or recordMenu:
                savingRecord(frame, video)
            selectMenu(ord(mainMenuValue))
            mainMenuSelected(frame)
            frame = drawing(frame)
        elif key == ord(colorMenuValue) or colorMenu:
            selectMenu(ord(colorMenuValue))
            rgbColorSelected(frame)
        elif key == ord(thicknessMenuValue) or thicknessMenu:
            selectMenu(ord(thicknessMenuValue))
            thicknessSelected(frame)
        elif key == ord(figureMenuValue) or figureMenu:
            selectMenu(ord(figureMenuValue))
            figureMenuSelected(frame)
            prepareSS = True
        #elif key == ord(eraseMenuValue) or eraseMenu:
        #    eraseMenuSelected()
        else:
            prepareSS = False
            prePointer = None

        #TEST findPointer
        #findPointer(frame,(cols,rows))
       
        #TEST
        if key == ord('s') and prepareSS:
            #getImage(frame.copy())
            getImage(frame)
    
        if cFigure is not None:
            frame = cv2.add(frame,cFigure)
        if key == 27: break

        
        
        
        cv2.imshow("frame", frame)
        #DEBUG
        #cv2.imshow("copy",copy)

video.release()
cv2.destroyAllWindows()
