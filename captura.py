import cv2
import os

# Classificador de imagem, o arquivo é um treinamento para detecção de imagem
classificador = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')

# captura da imagem pela webcam, o metodo espera a camera que sera utilizada para a captura. As cameras disponiveis na maquina são numeradas
camera = cv2.VideoCapture(0)

# variavel para controlar quantas fotos foram tiradas
amostra = 1

# capturar 25 fotos de cada pessoa
numeroAmostras = 25

# input libera a entrada de dados pelo terminal
id = input('Digite o seu identificador: ')
# controlar o tamanho da imagem
largura, altura = 220, 220
print(f'capturando imagens do identificador {id}')

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
        # Se as amostrar chegar na quantidade de numeroAmostras, vamos parar o for
        if amostra >= numeroAmostras + 1:
            break
        # Se o programa estiver esperando uma tecla E(&) a tecla Q for teclada(comparar com hexadecimal) vamos capturar uma imagem e salva-la
        # o ord converte caracteres para hexadecimal, 0xFF = tecla "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # imagem redimensionada
            imagemFace = cv2.resize(
                # redimensionar a imagem para capturar a região detectada
                imagemCinza[y:y + a, x:x + l], (largura, altura))
            # Se não existir a pasta fotos vamos criar
            if not os.path.isdir('fotos/'):
                os.mkdir('fotos/')
            # Salvar a imagem redimensionada na pasta fotos
            cv2.imwrite(f'fotos/pessoa.{id}.{amostra}.jpg', imagemFace)
            print(f'Foto {amostra} capturada com sucesso')
            amostra += 1

        # Escrevendo na tela
        cv2.putText(imagem, 'HOMEM', (x, y + l),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    print('Faces capturadas com sucesso')
    # exibir a imagem capturada pela webcam
    cv2.imshow('Face', imagem)
    cv2.waitKey(1)

# Liberar a memória do uso da camera
camera.release()
# destruir todas janelas criada a cima
cv2.destroyAllWindows()
