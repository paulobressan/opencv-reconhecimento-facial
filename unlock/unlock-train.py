import cv2
import os
import numpy as np

lbph = cv2.face.LBPHFaceRecognizer_create()

def GetImagesUser():
    paths = [os.path.join('unlock/user', u) for u in os.listdir('unlock/user')]
    faces = []
    ids = []

    for path in paths:
        face = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        faces.append(face)
        id = int(os.path.split(path)[1].split('.')[0])
        ids.append(id)
    return np.array(ids), faces


ids, faces = GetImagesUser()

lbph.train(faces, ids)
lbph.write('unlock/identifyLbph.yml')
print('Algoritmo treinado')
