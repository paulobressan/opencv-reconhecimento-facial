import cv2
import numpy as np
import os

classificador = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
camera = cv2.VideoCapture(0)
amostra = 1
numeroAmostra = 25
largura, altura = 220, 220
nomeUsuario = input('Digite o nome do usuário : ')
while True:
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    faceDetectadas = classificador.detectMultiScale(
        imagemCinza, scaleFactor=1.5, minSize=(100, 100))

    for x, y, l, a in faceDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255))
        regiao = imagem[y:y + a, x:x + l]
        if amostra >= numeroAmostra + 1:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            if np.average(imagemCinza) > 110:
                print('Face detectada')
                imagemFace = cv2.resize(
                    imagemCinza[y:y+a, x:x+l], (largura, altura))
                if not os.path.isdir('user/'):
                    os.mkdir('user/')
                cv2.imwrite(f'{nomeUsuario}.{amostra}.jpg', imagemFace)
                amostra += 1
    cv2.imshow('Face', imagem)
    cv2.waitKey(1)

camera.release()
cv2.destroyAllWindows()
