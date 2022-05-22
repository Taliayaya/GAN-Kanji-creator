import cv2
import numpy as np
from keras.models import load_model


generateur = load_model("./GEN.h5")
for i in range(20):
    bruit = np.random.normal(0, 1, size=[1, 100])
    print("Generating image...")
    print()
    image = generateur.predict(bruit)
    image = (image*127.5)+127.5
    image = image.astype("uint8")
    image = image.reshape((128, 128, 3))
    imname = "newkanji_"+str(i)+".jpg"
    cv2.imwrite("doss/" + imname, image)
