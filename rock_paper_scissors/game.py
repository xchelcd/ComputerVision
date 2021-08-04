from keras.models import load_model
import cv2
import mediapipe as mp
import time
import numpy as np
from random import choice
#choice(['rock','paper','scissors'])

results = None
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands(max_num_hands=1,min_detection_confidence=0.8)

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "ok",
    4: "oknt"
}
def mapper(val):
    return REV_CLASS_MAP[val]

imgPath = 'images/'
#scissorsImg = cv2.imread("{}{}.png".format(pathImg,mapper(2)))
#paperImg = cv2.imread("{}{}.png".format(pathImg,mapper(1)))
#rockImg = cv2.imread("{}{}.png".format(pathImg,mapper(0)))

#scissorsImg = cv2.resize(scissorsImg,(480,480))
#paperImg = cv2.resize(paperImg,(480,480))
#rockImg = cv2.resize(rockImg,(480,480))
#TEST
#cv2.imshow("a",paperImg)

'''
Selección del menú
'''
optionMenu = True
optionMenuVal = '0'
gameMenu = False
gameMenuVal = '1'
plusVal = '2'
lessVal = '3'

boxSize = 50
rawCols = 640
rawRows = 480
    
#p1Start = (3,3)
#p2Start = (boxSize+3,boxSize+3)
p1Start = (rawCols//2-boxSize//2,3)
p2Start = (rawCols//2+boxSize//2,boxSize+3)


p1Less = (rawCols//2-2*boxSize,rawRows//2+boxSize//3)
p2Less = (rawCols//2-boxSize,rawRows//2+boxSize//3+boxSize)
p1Plus = (rawCols//2+boxSize//2,rawRows//2+boxSize//3)
p2Plus = (rawCols//2+boxSize//2+boxSize,rawRows//2+boxSize//3+boxSize)

attempts = 1
isStarted = False

t1 = None
t2 = None
timeFlag = True

timePlayFlag = True
t1Play,t2Play=None,None

pointer = None

mePoints = 0
botPoints = 0
me,bot = None,None

thereWinner = None

timeToThink = 5
timeToSelect = 1

xI,yI=None,None

isDetectedOk = False

'''
Simplifica la función de opencv
'''
def putText(src, text, loc, color):
    cv2.putText(src, "{}".format((text)),
                loc, cv2.FONT_HERSHEY_SIMPLEX, 1,
                color, 1, cv2.LINE_AA)

'''
Distingue si el juego empezó o no para saber qué es lo que se va a dibujar
'''
def drawMenu(img):
    global mePoints,botPoints,thereWinner
    white = (255,255,255)
    green = (0,255,0)
    red = (0,0,255)
    blue = (255,0,0)
    color = None

    cols = img.shape[1]
    rows = img.shape[0]
    
    if isStarted is False: color = red
    else: color = green
    
    
    cv2.rectangle(img,p1Start,p2Start,color,4)
    if isStarted is False:
        putText(img,"El mejor de",(cols//3,rows//2),white)
        putText(img,attempts,(cols//2-boxSize//2,rows//2+boxSize),red)
        
        cv2.rectangle(img,p1Less,p2Less,white,2)
        cv2.rectangle(img,p1Plus,p2Plus,white,2)
    else:
        putText(img,"Me {}".format(mePoints),(10,30),blue)
        putText(img,"{} Bot".format(botPoints),(cols-130,30),blue)
        putText(img,"The best of {}".format(attempts),(cols//3,rows-10),red)
    
'''
Se detectan las manos en copy y las dibuja en img esto
para que el menú dibujado en img no interfiera en la
detección de manos, copy es únicamente lo que captura
la cámara
'''
def findHands(img,copy):
    global results, pointer
    global me
    imgRGB = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    cols = img.shape[1]#640
    rows = img.shape[0]#480
    
    if results.multi_hand_landmarks:
        for hands_landmarks in results.multi_hand_landmarks:
            '''
            xI e yI obtienen el punto del dedo índide
            '''
            xI = int(hands_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x*cols)
            yI = int(hands_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y*rows)
            cv2.circle(img,(xI,yI),8,(0,0,255),-1)
            #DEBUG
            #putText(img,"{},{}".format(xI,yI),(xI,yI),(0,0,0))
            pointer = (xI,yI)
            #DEBUG
            #mpDraw.draw_landmarks(img, hands_landmarks,
            #                      mpHands.HAND_CONNECTIONS)
    else:
        '''
        Si no detecta manos el puntero es None para evitar que
        se mantenga en la última posición cuando ya no hayan
        manos en el frame (en caso de que la última posición
        sea en una de las opciones)
        '''
        pointer = None
        me = None
    return img

'''
Las manos ya se detectan, pero se requiere extrar esa región de interés.
Se buscan los valores min y max de cada punto en las manos para dibujar
un rectángulo #DEBUG. Este rectángulo se extrae de la imagen y es el que
posteriormente se identifica lo que hay en él (piedra, papel, tijeras,
ok, oknt)
'''
def drawRectHand(img):
    global results
    global pointer
    xList = []
    yList = []
    bbox = []
    lmList = []
    results = hands.process(img)
    p1,p2 = None, None
    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[0]
        for id, lm in enumerate(myHand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            xList.append(cx)
            yList.append(cy)
            lmList.append([id, cx, cy])
            
        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        boxW, boxH = xmax - xmin, ymax - ymin
        bbox = xmin, ymin, boxW, boxH

        a = 50
        p1 = (bbox[0] - a, bbox[1] - a)
        p2 = (bbox[0] + bbox[2] + a, bbox[1] + bbox[3] + a)
        #DEBUG
        #cv2.rectangle(frame, p1,p2,(255, 255, 0), 2)
    else: pointer = None
    if p1 is not None and p2 is not None:
        '''
        roi
        '''
        return img[p1[1]:p2[1],p1[0]:p2[0]]

'''
Busca donde se encuentra el puntero, si se encuentra en alguna
de las opciones que tiene el usuario devuelve se activa la bandera
de dicha opción.
Las opciones se encuentran en un recuadro y se tiene que llevar el
dedo índice al recuadro
'''
def positionPointer(pointer):
    global isStarted, attempts
    global mePoints,botPoints
    p1 = None
    p2 = None
    val = None
    red = (0,0,255)
    
    if p1Start[0] < pointer[0] < p2Start[0] and pointer[1] < p2Start[1]:
        p1,p2 = p1Start,p2Start
        if isStarted: val = ord(optionMenuVal)
        else:
            mePoints = 0
            botPoints = 0
            val = ord(gameMenuVal)
    elif not isStarted:
        if p1Less[0] < pointer[0] < p2Less[0] and p1Less[1] < pointer[1] < p2Less[1]:
            p1,p2 = p1Less,p2Less
            val = ord(lessVal)
        elif p1Plus[0] < pointer[0] < p2Plus[0] and p1Plus[1] < pointer[1] < p2Plus[1]:
            p1,p2 = p1Plus,p2Plus
            val = ord(plusVal)
    if p1 is not None or p2 is not None:
        cv2.rectangle(frame,p1,p2,red,1)
        
    return val

'''
Cambia la bandera del menú según la opción elegida
'''
#DEPRECATED
#def menuSelected(key):
#    if key == ord(optionMenuVal) or ord(gameMenuVal):
#        global optionMenu, gameMenu
#        optionMenu = key == optionMenuVal
#        gameMenu = key == gameMenuVal

'''
Toma el tiempo que se ha tenido el dedo índice en las opciones,
si supera 1s entonces se tomará como que el usuario decidió
seleccionar dicha opción
'''
def isSelectedOption():
    global timeFlag
    global rawCols
    global t1,t2
    global isStarted
    global attempts
    
    if pointer is not None:
        posPointer = positionPointer(pointer)
        if posPointer is not None:
            '''
            timeFlag es True solo la primera vez que entra a la opción,
            luego se pasa a False para no tener una diferencia de tiempos 0
            '''
            if timeFlag:
                t1 = time.time()
                timeFlag = False
            t2 = time.time()
            if t1 is not None and t2 is not None:
                '''
                tiempo en seg
                '''
                tIn = t2-t1
                putText(frame,round(tIn,2),(rawCols//3,25),(255,255,255))
                if tIn > timeToSelect:
                    '''
                    Se checa dónde se encuentra el puntero
                    '''
                    if posPointer == ord(optionMenuVal) or posPointer == ord(gameMenuVal):
                        isStarted = not isStarted
                    elif posPointer == ord(lessVal):
                        if attempts == 1: pass
                        else: attempts = attempts -2
                    elif posPointer == ord(plusVal):
                        if attempts == 9: pass
                        else: attempts = attempts +2
                    '''
                    Si se cumple 1s en la opción, se reinician los tiempos
                    para que se tome otra opción si se cambia rápido a otra
                    opción
                    '''
                    t1 = None
                    t2 = None
        else: timeFlag = True

'''
Obtiene la roi y detecta que hay
'''
def selectGun(roi):
    global me
    global isDetectedOk
    #DEBUG
    #cv2.imshow("roi",roi)
    roi = cv2.resize(roi, (227, 227))
    pred = model.predict(np.array([roi]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)
    me = user_move_name
    #DEBUG
    #putText(frame,"{}".format(user_move_name),(50,50),(255,0,0))

    if user_move_name == 'ok':
        isDetectedOk = True
    
    #DEBUG
    global pointer
    putText(frame,"{}".format(user_move_name),(pointer[0],pointer[1]),(255,0,0))

'''
Define al ganador
'''
def winner(me,bot):
    global mePoints,botPoints
    global isStarted
    if me == 'rock':
        if bot == 'paper': botPoints = botPoints +1
        elif bot == 'scissors': mePoints = mePoints +1
    elif me == 'paper':
        if bot == 'scissors': botPoints = botPoints +1
        elif bot == 'rock': mePoints = mePoints +1
    elif me == 'scissors': 
        if bot == 'rock': botPoints = botPoints +1
        elif bot == 'paper': mePoints = mePoints +1
    elif me == 'oknt':
        isStarted = False
    else: botPoints = botPoints +1
    
'''
Inicia el juego y se espera 5s para que el usuario decida su movimiento.
Cuando pasan los 5s la computadora escoge entre piedra, papel y tijeras
y dependiendo el resulado se le suma 1 punto al ganador.
Si el movimiento del usuario no se reconoce, se le da el punto a la computadora
'''
#img = None
def toPlay(img):
    global timePlayFlag
    global t1Play,t2Play
    global me,bot
    global mePoints, botPoints
    global attempts
    #global img
    timeLeft = None

    if not isDetectedOk: return
    
    if timePlayFlag:
        t1Play = time.time()
        timePlayFlag = False
    t2Play = time.time()
    if t1Play is not None and t2Play is not None:
        timeLeft = timeToThink-(t2Play-t1Play)
        if timeLeft < 0:
            #if img is not None:
            #    frame[0:480,640:640+480] = img
            #TEST
            print(me,bot)
            
            winner(me,bot)
            t1Play = None
            t2Play = None
            timePlayFlag = True
            cv2.waitKey(750)
    if timeLeft is not None:
        putText(img,"Start in {}".format(round(timeLeft,1)),(rawCols//3,rawRows//2),(255,255,255))
############################################################

video = cv2.VideoCapture(0)

'''
Se carga el modelo
'''
model = load_model("model/rock-paper-scissors-model.h5")
aux = None
string = ''

srcPath = 'src/'
plus = cv2.resize(cv2.imread("{}plus.jpg".format(srcPath)),(50,50))
less = cv2.resize(cv2.imread("{}less.jpg".format(srcPath)),(50,50))
play = cv2.resize(cv2.imread("{}play.jpg".format(srcPath)),(50,50))
stop = cv2.resize(cv2.imread("{}stop.jpg".format(srcPath)),(50,50))

while True:
        
    ret, frame = video.read()
    frame = cv2.flip(frame,1)
    copy = frame.copy()

    key = cv2.waitKey(1)

    drawMenu(frame)
    isSelectedOption()

    game = findHands(frame,copy)
    #roi = drawRectHand(game)
    roi = drawRectHand(copy)

    if not isStarted:
        isDetectedOk = False
        game[p1Start[1]:p2Start[1], p1Start[0]:p2Start[0]] = play
        game[p1Less[1]:p2Less[1], p1Less[0]:p2Less[0]] = less
        game[p1Plus[1]:p2Plus[1], p1Plus[0]:p2Plus[0]] = plus
    else:
        game[p1Start[1]:p2Start[1], p1Start[0]:p2Start[0]] = stop

    '''
    Se corrobora que exista roi
    '''
    if roi is not None and isStarted:
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            selectGun(roi)

    finalFrame = cv2.hconcat([game,np.zeros((480,480,3),dtype=np.uint8)])
    if isStarted:
        #toPlay(game)
        string = ''
        toPlay(finalFrame)

        '''
        Se escoge una opción al azar para la computadora
        '''
        bot = choice(['rock','paper','scissors'])
        imgString = "{}{}.png".format(
            imgPath,bot)
        img = cv2.resize(cv2.imread(imgString),(480,480))

        '''
        Al frame original se le agrega el espacio para que se muestre
        la jugada de la computadaora
        '''
        finalFrame[0:480,640:640+480] = img
        if mePoints == attempts//2+1:
            #TEST
            print("win")
            string = 'Winner {}-{}'.format(mePoints,botPoints)
            isStarted = False
            thereWinner = True
        elif botPoints == attempts//2+1:
            thereWinner = True
            isStarted = False
            #TEST
            print("lose")
            string = 'Loser {}-{}'.format(mePoints,botPoints)
        #TEST
        else:
            if thereWinner == True:
                string = '{}-{}'.format(mePoints,botPoints)
            #thereWinner = None       
    else:
        #if mePoints > 0 or botPoints > 0:
        #    string = '{}-{}'.format(mePoints,botPoints)
        if thereWinner is not None and thereWinner is False:
            string = '{}-{}'.format(mePoints,botPoints)
        mePoints = 0
        botPoints = 0
    
    putText(finalFrame,string,(rawCols//2-boxSize//2,3*boxSize),(0,0,255))
    
    if key == 27: break
    
    cv2.imshow("Game",finalFrame)
    #TEST
    #cv2.imshow("Game",game)

video.release()
cv2.destroyAllWindows()
