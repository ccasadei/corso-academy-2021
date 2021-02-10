from math import ceil

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(1234)
tf.random.set_seed(1234)

# Definisco un modello 'backbone' utilizzando il classificatore 'MobileNet'
# pre-addestrato con dataset 'ImageNet'
# Indico che non voglio i top layer del classificatore, perchè userò i miei
# Indico anche che l'ultimo layer sarà un GlobalAveragePool, che consente di
# limitare ulteriormente i parametri da gestire
backbone = MobileNetV2(weights='imagenet', include_top=False, pooling="avg")

# indico i layer del backbone come "non addestrabili", in modo da non modificarli
for layer in backbone.layers:
    layer.trainable = False

# definisco gli ultimi layer di classificazione, usando come ingresso le uscite del backbone
x = backbone.output
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)

# il mio modello complessivo avrà gli stessi ingressi del backbone e l'uscita che ho definito
# nei mei top-layers
model = Model(inputs=backbone.input, outputs=preds)

# visualizzo un sommario del modello complessivo
model.summary()

# compilo il modello con una loss di classificazione, l'ottimizzatore Adam ed aggiungendo l'accuracy come metrica
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# creo un generatore di immagini che utilizzi la funzione di preprocessing necessaria al modello MobileNetV2
train_datagen = ImageDataGenerator(preprocessing_function=mobilenet_v2.preprocess_input)

# indico al generatore di immagini dove si trovano le immagini, le dimensioni da usare, il formato colore da usare,
# il batch_size con cui costruire i vari batch, il tipo di classificazione, e se deve mischiare il dataset
train_generator = train_datagen.flow_from_directory('./train',
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)

# addestro il modello usando il generatore di immagini definito in precedenza, indicando
# quanti cicli eseguire per ogni epoca (lo calcolo dividendo l'ampiezza del dataset per il batch_size)
# ed utilizzando 10 epoche in tutto
model.fit_generator(generator=train_generator,
                    steps_per_epoch=ceil(train_generator.n / train_generator.batch_size),
                    epochs=5,
                    verbose=1)

def load_image(img_path):
    # carico l'immagine dal file
    img = image.load_img(img_path, target_size=(224, 224))
    plt.imshow(img)
    plt.show()
    # trasformo l'immagine in un array Numpy
    # lo shape dell'array sono (altezza, larghezza, canali colore)
    # quindi in questo caso (224, 224, 3)
    img_array = image.img_to_array(img)
    # aggiungo una dimensione all'inizio
    # lo shape diventa (1, 224, 224, 3)
    # dove "1" indica quante immagini sono presenti nel batch
    img_array_batch = np.expand_dims(img_array, axis=0)
    # normalizzo i valori da 0..255 a 0..1
    img_array_batch /= 255.

    return img_array_batch


img_di_test = load_image("./test/cinciarella.jpg")
predizione = model.predict(img_di_test)
print("Predizione cinciarella: {:.2%}".format(predizione[0][0]))

img_di_test = load_image("./test/corvo.jpg")
predizione = model.predict(img_di_test)
print("Predizione corvo: {:.2%}".format(predizione[0][1]))
