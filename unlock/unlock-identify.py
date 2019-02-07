import cv2

detectFace = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
identify = cv2.face.LBPHFaceRecognizer_create()
identify.read('unlock/identifyLbph.yml')
width, heigth = 220, 220
webcam = cv2.VideoCapture(0)

while True:
    conect, image = webcam.read()
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detectedFaces = detectFace.detectMultiScale(
        imageGray, scaleFactor=1.5, minSize=(100, 100))

    for x, y, l, a in detectedFaces:
        imageFace = cv2.resize(imageGray[y:y+a, x:x+l], (width, heigth))
        id, confiance = identify.predict(imageFace)
        if id == 1:
            print('Computador liberado')
        
    cv2.waitKey(1)

webcam.release()
cv2.destroyAllWindows()
