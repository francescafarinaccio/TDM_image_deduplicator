
#Estrattore di feature per immagini naturali basato su metodi nativi HOG (scikit-image) e Istogramma cromatico 3D HSV (OpenCV).

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import cv2
from skimage.feature import hog


#dataclass per rappresentare i descrittori estratti da un'immagine fotografica
@dataclass
class PhotoDescriptor:
    filepath: str
    hsv_hist: np.ndarray   # Istogramma 3D appiattito (8x8x8 = 512 bin)
    hog_vec: np.ndarray    # Vettore dei gradienti orientati (HOG)


#estrattore di feature per immagini naturali basato su HOG e Istogramma HSV
class PhotoFeatureExtractor:

    def __init__(self, hsv_bins: Tuple[int, int, int] = (8, 8, 8), w_hsv: float = 0.5, w_hog: float = 0.5):
        """
        :param hsv_bins: Quantizzazione per i canali Hue, Saturation e Value.
        :param w_hsv: Peso assegnato alla similarità cromatica HSV.
        :param w_hog: Peso assegnato alla similarità di struttura HOG.
        """
        self.hsv_bins = hsv_bins
        self.w_hsv = w_hsv
        self.w_hog = w_hog

    #funzione principale per estrarre i descrittori da un'immagine
    def extract(self, filepath: str) -> PhotoDescriptor:
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            total_bins = self.hsv_bins[0] * self.hsv_bins[1] * self.hsv_bins[2]
            return PhotoDescriptor(
                filepath=filepath,
                hsv_hist=np.zeros(total_bins, dtype=np.float32),
                hog_vec=np.zeros(0, dtype=np.float32)
            )


        #a differenza di RGB, lo spazio colore HSV separa la cromaticità dalla luminosità, rendendo l'istogramma più robusto a variazioni di illuminazione
        #lo spazio colore viene discretizzato in un num fisso di bin per ogni canale (Hue, Saturation, Value), creando un istogramma 3D che rappresenta la distribuzione dei colori nell'immagine.
        #il confronto tra istogrammi 3D di immagini diverse può essere effettuato tramite la distanza di Bhattacharyya, che misura la similarità tra due distribuzioni di probabilità.
    
        # 1. Istogramma HSV 3D (OpenCV)
        hsv_hist = self._extract_hsv_hist(img_bgr)


        #l'img viene divisa in una griglia di celle, e per ogni cella viene calcolato un istogramma dei gradienti orientati (HOG), che cattura la struttura locale dell'immagine.
        #per ciascun pixel si calcola il gradiente 
        # in ogni cella si costruisce un istogramma che raggruppa le direzioni dei gradienti 
        # tutti gli istogrammi delle celle vengono normalizzati e concatenati per formare un vettore HOG globale che rappresenta la struttura dell'immagine.        

        # 2. HOG (Scikit-Image)
        hog_vec = self._extract_hog(img_bgr)

        return PhotoDescriptor(
            filepath=filepath,
            hsv_hist=hsv_hist,
            hog_vec=hog_vec
        )

    #calcola l'istogramma 3D HSV e lo normalizza
    def _extract_hsv_hist(self, img_bgr: np.ndarray) -> np.ndarray:
   
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # H in [0, 180], S in [0, 256], V in [0, 256]
        hist = cv2.calcHist(
            [img_hsv], 
            channels=[0, 1, 2], 
            mask=None, 
            histSize=list(self.hsv_bins), 
            ranges=[0, 180, 0, 256, 0, 256]
        )
        
        # Normalizzazione L1 per rendercelo indipendente dalle dimensioni dell'immagine
        cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
        return hist.astype(np.float32).flatten()

    #calcola il vettore HOG dell'immagine ridimensionata e in scala di grigi
    def _extract_hog(self, img_bgr: np.ndarray) -> np.ndarray:
        # Ridimensionamento standard per uniformare la dimensione del vettore HOG finale
        resized = cv2.resize(img_bgr, (128, 128), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        hog_vec = hog(
            gray, 
            orientations=8, 
            pixels_per_cell=(16, 16), 
            cells_per_block=(2, 2), 
            block_norm='L2-Hys', 
            visualize=False
        )
        return hog_vec.astype(np.float32)

    #calcola la similarità composita tra due descrittori di immagini naturali, combinando la similarità cromatica HSV e la similarità strutturale HOG in un unico punteggio.
    def compute_similarity(self, desc1: PhotoDescriptor, desc2: PhotoDescriptor) -> float:
       
        if desc1.hsv_hist.size == 0 or desc2.hsv_hist.size == 0:
            return 0.0

        # 1. Similarità CROMATICA: Distanza di Bhattacharyya via cv2.compareHist
        hist1_3d = desc1.hsv_hist.reshape(self.hsv_bins)
        hist2_3d = desc2.hsv_hist.reshape(self.hsv_bins)
        
        # Distanza di Bhattacharyya d \in [0, 1] dove 0 significa identico
        bhattacharyya_dist = cv2.compareHist(hist1_3d, hist2_3d, cv2.HISTCMP_BHATTACHARYYA)
        sim_hsv = max(0.0, 1.0 - bhattacharyya_dist)

        # 2. Similarità STRUTTURALE: Cosine Similarity sui vettori HOG
        norm1 = np.linalg.norm(desc1.hog_vec)
        norm2 = np.linalg.norm(desc2.hog_vec)
        
        if norm1 == 0 or norm2 == 0:
            sim_hog = 0.0
        else:
            sim_hog = float(np.dot(desc1.hog_vec, desc2.hog_vec) / (norm1 * norm2))
            sim_hog = max(0.0, sim_hog)

        # 3. Combinazione pesata
        similarity = self.w_hsv * sim_hsv + self.w_hog * sim_hog
        return round(float(similarity), 4)