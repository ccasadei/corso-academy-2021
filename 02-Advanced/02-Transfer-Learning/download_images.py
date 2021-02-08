import os
import urllib

from simple_image_download import simple_image_download as simp
from tqdm import tqdm


def download_image(url, directory, filename):
    os.makedirs(directory, exist_ok=True)
    try:
        # eseguo una GET sulla url passata come parametro
        with urllib.request.urlopen(url) as url_get:
            # apro il file su cui scriverò l'immagine
            with open(os.path.join(directory,filename), "wb") as f:
                # leggo dalla GET e scrivo sul file
                f.write(url_get.read())
    except:
        pass


# uso la libreria per scaricare automaticamente un certo numero di immagini di cinciarelle e corvi
# NOTA: il dataset scaricato può contenere immagini non adatte all'addestramento, quindi va verificato
response = simp.simple_image_download

for img_n, img in enumerate(tqdm(response().urls("cinciarella", limit=100), desc="Cinciarella")):
    download_image(img, "./train/cinciarella/", "cinciarella_%02d.jpg" % img_n)

for img_n, img in enumerate(tqdm(response().urls("corvo", limit=100), desc="Corvo")):
    download_image(img, "./train/corvo/", "corvo_%02d.jpg" % img_n)


# verifico la predizione su un disegno di ciciarella (immagine non utilizzato in training)
download_image("https://www.ebnitalia.it/easyUp/gallery/square/IMG_7801%20Cinciarella%20-_1_m.jpg",
               "./test/",
               "cinciarella.jpg")

# analogamente per una immagine di corvo
download_image("https://st.depositphotos.com/1047246/3493/v/600/depositphotos_34937775-stock-illustration-black-raven-bird-hand-drawing.jpg",
               "./test/",
               "corvo.jpg")