import cv2

# Classificador de imagem, o arquivo é um treinamento para detecção de imagem
classificador = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")

# captura da imagem pela webcam, o metodo espera a camera que sera utilizada para a captura. As cameras disponiveis na maquina são numeradas
camera = cv2.VideoCapture(0)

while True:
    # capturar imagem pela webcam
    conectado, imagem = camera.read()

    # Converter a imagem capturada para cinza porque o algoritimo tende a ser melhor com a escala de sinza.
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Detectar as faces da imagem capturada pela webcam.
    # Como parametros vamos passar a imagem convertida para cinza, o scaleFactor é a escala que desejamos utilizar da imagem,
    # o minSize é o tamanho minimo para fazer a detecção de faces
    # Na variavel facesDetectadas estão todas as faces detectadas
    facesDetectadas = classificador.detectMultiScale(
        imagemCinza, scaleFactor=1.5, minSize=(100, 100))

    # Colocar o quadro ao redor da face que for detectado
    # posição x de onde começa uma face, posição y de onde começa uma face, largura, altura
    for x, y, l, a in facesDetectadas:
        # Para definir um retangulo ao redor da face detectada, temos que passar a imagem colorida, os dois eixos x e y,
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255))
        cv2.putText(imagem, "PAULO BRESSAN", (x, y + l),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # exibir a imagem capturada pela webcam
    cv2.imshow("Face", imagem)
    cv2.waitKey(1)

# Liberar a memória do uso da camera
camera.release()
# destruir todas janelas criada a cima
cv2.destroyAllWindows()
