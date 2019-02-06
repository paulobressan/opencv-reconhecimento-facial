import cv2


def nomePorId(id):
    if id == 1:
        return 'Paulo'
    elif id == 2:
        return 'Boiola'
    else:
        return ''


# detector de faces
detectorFace = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
# criando o reconhecedor
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
# Carregando para o reconhecedor o arquivo treinado
reconhecedor.read('classificadorLbph.yml')
# definindo as dimensões da imagem
largura, altura = 220, 220
# font usada para escrever na tela
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# Iniciar a captura de imagem com a webcam
camera = cv2.VideoCapture(0)

while True:
    # conectando e capturando uma imagem da webcam
    conectado, imagem = camera.read()
    # convertendo a imagem para a tonalidade/escalas cinza
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # detectando as faces da imagemCinza com a escala de 1.5 e o tamanho minimo de 30 x 30
    facesDetectadas = detectorFace.detectMultiScale(
        imagemCinza, scaleFactor=1.5, minSize=(30, 30))

    # percorrendo as faces detectadas
    for x, y, l, a in facesDetectadas:
        # resize na imagem detectada para o tamanho da largura e altura
        imagemFace = cv2.resize(imagemCinza[y:y+a, x:x + l], (largura, altura))
        # Adicionando o retandulo ao redor da face
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        # reconhecendo a face com o lbph treinado no treinamento.py
        id, confianca = reconhecedor.predict(imagemFace)
        # escrevendo o texto na imagem
        cv2.putText(imagem, nomePorId(id),
                    (x, y + (a + 30)), font, 2, (0, 0, 255))
        cv2.putText(imagem, str(confianca),
                    (x, y + (a + 50)), font, 2, (0, 0, 255))

    cv2.imshow("Face", imagem)
    cv2.waitKey(1)

camera.release()
cv2.destroyAllWindows()
