from random import randint

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.python.keras.utils import np_utils

# ottengo il dataset MNIST già suddiviso in dataset X e Y, di addestramento e di test
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# visualizzo 4 cifre random
for i in range(4):
    plt.subplot(2, 2, (i + 1))
    # in 'shape[0]' è contenuto il numero di esempi del dataset (in questo caso di addestramento)
    plt.imshow(X_train[randint(0, X_train.shape[0])], cmap=plt.get_cmap('gray'))
plt.show()

# imposto un seed random in modo da ottenere risultati replicabili, d'ora in avanti
np.random.seed(1234)
tf.random.set_seed(1234)

# gli array di MNIST fornite da Keras hanno queste dimensioni
#   (n_esempi, larghezza, altezza)
#
# la nostra rete invece lavora con array di queste dimensioni
#   (n_esempi, larghezza, altezza, n_canali_colore)
#
# quindi mi basta aggiungere una nuova dimensione agli array
X_train = np.expand_dims(X_train, axis=-1).astype('float32')
X_test = np.expand_dims(X_test, axis=-1).astype('float32')

# normalizzo i valori dei pixel portandoli dal range intero 0-255 al range in virgola mobile 0.0-1.0
# visto che sono array numpy, è sufficiente eseguire l'operazione direttamente sull'array
X_train = X_train / 255
X_test = X_test / 255

# modifico gli array dei risultati ("ground truth") in modo siano in formato 'one hot encode'
# quindi i valori interi corrispondenti alla classe della cifra (0, 1, 2, ..., 9) vengono
# codificati in stringhe posizionali di 0 ed 1
# esempi:
#   0 --> 1,0,0,0,0,0,0,0,0,0
#   1 --> 0,1,0,0,0,0,0,0,0,0
#   2 --> 0,0,1,0,0,0,0,0,0,0
#  ....
#   9 --> 0,0,0,0,0,0,0,0,0,1
# in questo modo è più semplice ottenere un risultato significativo dalla rete neurale, in quanto
# ogni cifra posizionale corrisponderà ad un neurone dello strato di output che si attiverà o meno
# a seconda del risultato della classificazione della rete neurale
# in 'shape[0]' continuerà ad essere contenuto il numero di cifre del dataset
# in 'shape[1]' ci sarà invece il numero di cifre posizionali, corrispondente al numero di classi possibili
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# definisco un modello di rete neurale convoluzionale
# come dimensione del layer di ingresso uso quelle del train set (esclusa la dimensione iniziale del numero di cifre del dataset)
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), name="input_layer")
conv_1 = Conv2D(16, (5, 5), activation="relu", name="conv_1")(input_layer)
maxpool_1 = MaxPooling2D(pool_size=(2, 2), name="maxpool_1")(conv_1)
conv_2 = Conv2D(24, (5, 5), activation="relu", name="conv_2")(maxpool_1)
maxpool_2 = MaxPooling2D(pool_size=(2, 2), name="maxpool_2")(conv_2)
flattened = Flatten(name="flattened")(maxpool_2)
fc_3 = Dense(32, activation="relu", name="fc_3")(flattened)
fc_4 = Dense(10, activation="softmax", name="output_layer")(fc_3)

model = Model(inputs=[input_layer], outputs=[fc_4])
model.summary()

# compilo il modello indicando che tipo di loss_function devo utilizzare,
# il tipo di ottimizzatore e le metriche che voglio vengano calcolate
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# addestro il modello
errors_list = model.fit(X_train, y_train, epochs=10, verbose=1)

# visualizzo graficamente l'andamento dell'errore e dell'accuratezza durante la fase di addestramento
plt.plot(errors_list.history["loss"], label="Errore")
plt.plot(errors_list.history["accuracy"], label="Accuratezza")
plt.legend()
plt.title("Andamento errore")
plt.xlabel("Epoche")
plt.ylabel("Errore/Accuratezza")
plt.show()

# valuto il modello
valutazioni = model.evaluate(X_test, y_test, verbose=0)
print("Valuto %d immagini di test" % len(X_test))
print("Accuratezza del modello: {:.2f}%".format(valutazioni[1] * 100))