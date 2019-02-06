import cv2
import os
import numpy as np

# classificadores de imagens, esses objetos treinam e criam arquivos treinados com as imagens dadas a eles
# o num_components é a configuração de quantos eigenfaces vai ser treinado
# threshold é o limite da distancia de confiabilidade para o algoritimo KNN, quanto menor for o trashold mais precisa e qualidade sera a detecção
eigenface = cv2.face.EigenFaceRecognizer_create(
    num_components=50)
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

# def cria um metodo


def getImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    # print(caminhos)
    # Armazenar as faces de cada id
    faces = []
    # Armazenar todos ids
    ids = []
    for caminhoImagem in caminhos:
        # leitura de uma imagem pelo diretorio e converter para a cor cinza
        imagemFace = cv2.cvtColor(cv2.imread(
            caminhoImagem), cv2.COLOR_BGR2GRAY)
        # extrair o id da imagem, o os.path.split quebra uma url pela / e retorna um array e depois quebramos por . e pegamos a posição que é o id
        id = int(os.path.split(caminhoImagem)[1].split('.')[1])
        # adicionando o id extraido na lista de ids
        ids.append(id)
        # adicionando a face na lista de faces
        faces.append(imagemFace)
    # criar um array com dois array de diferentes, ou seja, criar um array que contem o array de ids e o aray de faces
    return np.array(ids), faces


# Ao criar duas variavel que recebe o retorno do metodo getImagemComId que retorna um array que contem dois valores, esses valores vão ser adicionar nas duas variavel
# por sequencia, ids = [0], faces = [1]
ids, faces = getImagemComId()

print('Treinando...')

# treinar cada imagem com o seu id
eigenface.train(faces, ids)
# Arquivo treinado para identificar pessoa 1 e pessoa 2
eigenface.write('classificadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLbph.yml')

print('Treinamento realizado')
