from glob import glob
from keras.optimizers import Adam
from tqdm import tqdm

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, UpSampling2D, Reshape

import os

# ========
# Settings
# ========
ITERATIONS = 60
# Number of images to take from the dataset / that will be used to train
DEMI_BATCH = 500

# =====================================
# LOAD & PREPARE TRUE IMAGES (DATASETS)
# =====================================

os.environ['KAGGLE_CONFIG_DIR'] = "/content/"

optimizer = Adam(lr=0.0002, beta_1=0.5)

images_vraies = []

noms_image = glob("kanji/*")

i = 0

for nom in tqdm(noms_image):
    i = i + 1
    if i == 20000:
        break
    image = cv2.imread(nom, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (128, 128))
    image = image.astype("float32")
    image = (image-127.5)/127.5
    images_vraies.append(image)

images_vraies = np.array(images_vraies)

# ====================
# CREATE DISCRIMINATOR
# ====================

discriminateur = Sequential()
discriminateur.add(Conv2D(32, kernel_size=(
    3, 3), activation='relu', input_shape=(128, 128, 3)))

discriminateur.add(MaxPooling2D(pool_size=(2, 2)))

discriminateur.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
discriminateur.add(MaxPooling2D(pool_size=(2, 2)))


discriminateur.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
discriminateur.add(MaxPooling2D(pool_size=(2, 2)))

discriminateur.add(Flatten())

discriminateur.add(Dense(32, activation="relu"))
discriminateur.add(Dense(32, activation="relu"))

discriminateur.add(Dense(1, activation="sigmoid"))

print("DISCRIMINATEUR :")
discriminateur.summary()

discriminateur.compile(loss='binary_crossentropy',
                       optimizer=optimizer, metrics=['accuracy'])

# ================
# CREATE GENERATOR
# ================

generateur = Sequential()
generateur.add(Dense(16*16*256, activation='relu', input_shape=(100,)))

generateur.add(Reshape((16, 16, 256)))  # 16x16
generateur.add(UpSampling2D(size=(2, 2)))  # 16x16 -> 32x32

generateur.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))

generateur.add(UpSampling2D(size=(2, 2)))  # 32x32 -> 64x64

generateur.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))

generateur.add(UpSampling2D(size=(2, 2)))  # 64x64 -> 128x128
generateur.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))

generateur.add(Conv2D(3, kernel_size=(2, 2),
               padding='same', activation='tanh'))

print("GENERATEUR :")
generateur.summary()

# WE DON'T COMPILE IT. He's never training alone

# ============
# CREATE COMBO
# ============

combo = Sequential()
combo.add(generateur)
combo.add(discriminateur)

discriminateur.trainable = False
combo.compile(loss='binary_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])

print("COMBO :")
combo.summary()


# =====
# START
# =====

for iteration in range(ITERATIONS):
    print()
    print("################")
    print("  Boucle n°"+str(iteration))
    print("################")
    # ======================================
    # create a datapack for the discriminator
    # =======================================

    # step 1 : take good images
    # step 2 : generate labels (1 for true) of good images
    # step 3 : generate wrong images
    # step 4 : generate labels (0 for false) of wrong images

    # He needs to train on good image of the dataset -> label 1
    # And wrong images generated -> label 0

    x = []
    y = []

    # step 1 : take good images
    images_bonnes = images_vraies[np.random.randint(
        0, images_vraies.shape[0], size=DEMI_BATCH)]
    # step 2 : generate labels (1 for true) of good images
    labels_bonnes = np.ones(DEMI_BATCH)  # un tableau avec 1000 fois le label 1
    # step 3 :  generate wrong images
    # 1000 tableaux de 100 nombres aléatoires
    bruit = np.random.normal(0, 1, size=[DEMI_BATCH, 100])
    images_mauvaises = generateur.predict(bruit)  # milles images générées
    # step 4 :  generate labels (0 for false) of wrong images
    labels_mauvaises = np.zeros(DEMI_BATCH)

    x = np.concatenate([images_bonnes, images_mauvaises])
    y = np.concatenate([labels_bonnes, labels_mauvaises])

    # ===================
    # TRAIN DISCRIMINATOR
    # ===================

    discriminateur.trainable = True
    print()
    print("Training Discriminator :")
    print()
    discriminateur.fit(x, y, epochs=1, batch_size=16)

    # ============================
    # Create a data pack for combo
    # ============================

    # generate noises
    # 1000 tables of 100 random numbers
    bruit = np.random.normal(0, 1, size=[1000, 100])
    # generate labels 1
    labels_combo = np.ones(1000)

    # ===========
    # train combo
    # ===========
    print()
    print("Training Generator :")
    print()
    discriminateur.trainable = False
    combo.fit(bruit, labels_combo, epochs=1, batch_size=32)

    # =================================
    # Every 10 iteration, saves 1 image
    # =================================

    if iteration % 10 == 0:
        bruit = np.random.normal(0, 1, size=[1, 100])
        print("Generating image...")
        print()
        image = generateur.predict(bruit)
        image = (image*127.5)+127.5
        image = image.astype("uint8")
        image = image.reshape((128, 128, 3))
        imname = "1genim_"+str(iteration)+".jpg"
        cv2.imwrite("doss/" + imname, image)

        #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
        #                      Saving Models                      #
        #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
    if iteration % 10 == 0 and iteration != 0:
        print()
        print("Saving models...")
        print()
        discriminateur.trainable = True


discriminateur.save("DISC.h5")
generateur.save("GEN.h5")
