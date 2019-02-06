# trabalhar com o opencv
import cv2
# recursos do sistema operacional
import os
# trabalhar com recursos cientificos como calculos, por exemplo calculo de media
import numpy as np

# Classificador de face, o arquivo é um treinamento para detecção de face em imagens
classificador = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
# Classificador de olhos, o arquivo é um treinamento para detecção de olhos em imagens
classificadorOlho = cv2.CascadeClassifier('haarcascade-eye.xml')

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

        # Capturar somente a região que contem a detecção de face para analisar se existe o olho
        regiao = imagem[y:y + a, x:x + l]
        # Convertendo a imagem para cinza
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        # Não é necessario configurar o scale e nem o tamanho minimo porque a imagem que vamos trabalhar ja esta focada na face
        olhosDetectados = classificadorOlho.detectMultiScale(regiao)
        for ox, oy, ol, oa in olhosDetectados:
            # desenhar um retangulo em volta do olho, primeiro parametro é a imagem, segundo é a posição de inicio da imagem,
            # o terceiro é a regição onde vai ser desenhado o retangulo, o quarto é a cor e o quinto é o tamanho da borda
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)

        # Se as amostrar chegar na quantidade de numeroAmostras, vamos parar o for
        if amostra >= numeroAmostras + 1:
            break

        # Se o programa estiver esperando uma tecla E(&) a tecla Q for teclada(comparar com hexadecimal) vamos capturar uma imagem e salva-la
        # o ord converte caracteres para hexadecimal, 0xFF = tecla "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(np.average(imagemCinza))
            # average vai calcular a média de pixel da imagem, se o valor for muito baixa, quer dizer que a imagem é mais escura
            # cada pixel tem o seu rgb e por isso quanto mais perto do 0 for o resultado, mais escura vai ser a imagem, 0 = preto, 255 = branco
            if np.average(imagemCinza) > 110:
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
