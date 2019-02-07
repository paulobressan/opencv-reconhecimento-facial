import cv2
import numpy as np
import os

detectFace = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
webcam = cv2.VideoCapture(0)
count = 1
numberCount = 25
width, heigth = 220, 220
idUser = input('Digite o id do usuário : ')
while True:
    conect, image = webcam.read()
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detectedFace = detectFace.detectMultiScale(
        imageGray, scaleFactor=1.5, minSize=(100, 100))

    for x, y, l, a in detectedFace:
        cv2.rectangle(image, (x, y), (x + l, y + a), (0, 0, 255))
        if count >= numberCount + 1:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            if np.average(imageGray) > 110:
                print('detected face')
                imageFace = cv2.resize(
                    imageGray[y:y+a, x:x+l], (width, heigth))
                if not os.path.isdir('unlock/user/'):
                    os.mkdir('unlock/user/')
                cv2.imwrite(f'unlock/user/{str(idUser)}.{count}.jpg', imageFace)
                count += 1
    cv2.imshow('Face', image)
    cv2.waitKey(1)

webcam.release()
cv2.destroyAllWindows()
