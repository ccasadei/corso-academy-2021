"""
@Author: Cristiano Casadei

Esempio di rete neurale (con un neurone) che approssimi una funzione con questi valori ingresso - uscita

(le righe indicate con * sono quelle prese come esempi di addestramento)

        IN      OUT
  *  0 0 0 0 -> 0
  *  0 0 0 1 -> 0
     0 0 1 0 -> 0                                       \b2 b3
     0 0 1 1 -> 0                                  b1 b0 \
  *  0 1 0 0 -> 1                                         \00 01 10 11
     0 1 0 1 -> 0                                       00| 0| 1| 1| 1
  *  0 1 1 0 -> 0                                       01| 0| 0| 1| 1
     0 1 1 1 -> 0   <- Esempio per il test              10| 0| 0| 0| 1
     1 0 0 0 -> 1                                       11| 0| 0| 0| 0
  *  1 0 0 1 -> 1
     1 0 1 0 -> 0
  *  1 0 1 1 -> 0
  *  1 1 0 0 -> 1
     1 1 0 1 -> 1   <- Esempio per il test
  *  1 1 1 0 -> 1
     1 1 1 1 -> 1
"""

import numpy as np
from matplotlib import pyplot as plt


class MyNumpyNN:
    """
    Classe di gestione della mia rete neurale in Numpy (un neurone)
    """

    def __init__(self, nr_ingressi):
        """
        Inizializzazione dell'istanza
        
        :param nr_ingressi: numero di ingressi della rete neurale
        """

        # rendo l'esperimento ripetibile
        np.random.seed(1234)

        # inizializzo i pesi in modo casuale
        # ottengo un array "pesi" di dimensioni pari al numero di ingressi e con valori da -1 a + 1
        self.pesi = 2 * np.random.random((nr_ingressi, 1)) - 1

    def fn_sigmoidale(self, x):
        """
        Definizione della funzione sigmoidale

        :param x: valore dell'ascissa
        :return: valore dell'ordinata
        """
        return 1 / (1 + np.exp(-x))

    def derivata_fn_sigmoidale(self, x):
        """
        Definizione della derivata della funzione sigmoidale

        :param x: valore dell'ascissa
        :return: valore dell'ordinata
        """
        return x * (1 - x)

    def inferenza(self, ingressi):
        """
        Calcolo il valore dell'uscita in base agli ingressi applicati alla rete

        :param ingressi: valori di ingresso applicati alla rete
        :return: valori di uscita calcolati
        """

        # mi assicuro che gli ingressi siano forniti in virgola mobile
        # per non perdere precisione nei calcoli
        ingressi = ingressi.astype(float)

        # applico i pesi e sommo gli ingressi per ogni combinazioni
        # DOT è la moltiplicazione "riga per colonna" tra il vettore ingressi e il vettore pesi
        somme_ingressi_pesati = np.dot(ingressi, self.pesi)

        # applico la funzione sigmoidale alle somme degli ingressi pesati per ogni combinazione
        # ottengo il vettore delle uscite (una uscita per ogni combinazione in ingresso)
        uscita = self.fn_sigmoidale(somme_ingressi_pesati)

        return uscita

    def addestramento(self, ingressi_esempi, uscite_desiderate, iterazioni):
        """
        Funzione di addestramento della rete neurale

        :param ingressi_esempi: valori di ingresso degli esempi da utilizzare
        :param uscite_desiderate: valori delle uscite attese per ogni esempio
        :param iterazioni: numero di cicli per l'addestramento
        :return lista degli errori quadratici medi
        """

        lista_errori_quadratici_medi = []
        for _ in range(iterazioni):
            # applico i valori di ingresso alla rete neurale e ne calcolo il valore di uscita
            uscite = self.inferenza(ingressi_esempi)

            # calcolo l'errore come la differenza tra valori attesi e valori ottenuti
            # ci servirà per l'aggiustamento dei pesi
            errori = uscite_desiderate - uscite

            # poi calcolo l'errore quadratico medio su tutti gli esempi e lo aggiungo nella lista degli errori
            # ci servirà per una valutazione della fase di addestramento
            errore_quadratico_medio = np.sqrt(np.mean(errori ** 2))
            lista_errori_quadratici_medi.append(errore_quadratico_medio)

            # in base all'errore, agli ingressi e alle uscite desiderate, calcolo gli aggiustamenti dei pesi
            # utilizzando l'algoritmo di backpropagation
            # gli aggiustamenti corrispondono a quanto devo scostarmi rispetto al valore attuale dei pesi
            aggiustamenti = np.dot(ingressi_esempi.T, errori * self.derivata_fn_sigmoidale(uscite))

            # applico l'aggiustamento ai pesi
            self.pesi += aggiustamenti

        return lista_errori_quadratici_medi


if __name__ == "__main__":
    # preparo delle combinazioni di ingresso di prova
    ingressi_addestramento = np.array([[0, 0, 0, 0],
                                       [0, 0, 0, 1],
                                       [0, 1, 0, 0],
                                       [0, 1, 1, 0],
                                       [1, 0, 0, 1],
                                       [1, 0, 1, 1],
                                       [1, 1, 0, 0],
                                       [1, 1, 1, 0]])

    # preparo le relative uscite desiderate
    uscite_desiderate = np.array([[0],
                                  [0],
                                  [1],
                                  [0],
                                  [1],
                                  [0],
                                  [1],
                                  [1]])

    # inizializzo la rete neurale
    rete_neurale = MyNumpyNN(nr_ingressi=ingressi_addestramento.shape[1])

    # addestro la rete
    lista_mse = rete_neurale.addestramento(ingressi_addestramento, uscite_desiderate, 1000)

    # visualizzo solo i primi 500 valori, tanto per dare una visione all'andamento
    plt.plot(lista_mse[:500])
    plt.xlabel("Iterazioni")
    plt.ylabel("Errore quadratico medio")
    plt.title("Valutazione addestramento")
    plt.show()

    # provo ad usare una nuova combinazione di input mai vista prima
    ingressi_test = np.array([[1, 1, 0, 1],
                              [0, 1, 1, 1]])
    uscite_test = np.array([[1],
                            [0]])
    for indice in range(len(ingressi_test)):
        print()
        print("Input di test: %s" % ingressi_test[indice])
        print("Uscita calcolata (mi aspetto '%d'): %.5f" % (uscite_test[indice][0], rete_neurale.inferenza(ingressi_test[indice])[0]))
