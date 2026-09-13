"""
Modulo: photo_extractor.py
Descrizione: Estrattore di feature per immagini naturali basato su metodi nativi 
             HOG (scikit-image) e Istogramma cromatico 3D HSV (OpenCV).
Corso: Trattamento Dati Multimediali (TDM)
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import cv2
from skimage.feature import hog


@dataclass
class PhotoDescriptor:
    """Rappresentazione dei vettori HOG e dell'istogramma HSV 3D."""
    filepath: str
    hsv_hist: np.ndarray   # Istogramma 3D appiattito (8x8x8 = 512 bin)
    hog_vec: np.ndarray    # Vettore dei gradienti orientati (HOG)


class PhotoFeatureExtractor:
    """
    Estrattore nativo basato su HOG e Istogrammi HSV 3D.
    """

    def __init__(self, hsv_bins: Tuple[int, int, int] = (8, 8, 8), w_hsv: float = 0.5, w_hog: float = 0.5):
        """
        :param hsv_bins: Quantizzazione per i canali Hue, Saturation e Value.
        :param w_hsv: Peso assegnato alla similarità cromatica HSV.
        :param w_hog: Peso assegnato alla similarità di struttura HOG.
        """
        self.hsv_bins = hsv_bins
        self.w_hsv = w_hsv
        self.w_hog = w_hog

    def extract(self, filepath: str) -> PhotoDescriptor:
        """Estrae i descrittori HSV 3D ed HOG dal file immagine."""
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            total_bins = self.hsv_bins[0] * self.hsv_bins[1] * self.hsv_bins[2]
            return PhotoDescriptor(
                filepath=filepath,
                hsv_hist=np.zeros(total_bins, dtype=np.float32),
                hog_vec=np.zeros(0, dtype=np.float32)
            )

        # 1. Istogramma HSV 3D (OpenCV)
        hsv_hist = self._extract_hsv_hist(img_bgr)

        # 2. HOG (Scikit-Image)
        hog_vec = self._extract_hog(img_bgr)

        return PhotoDescriptor(
            filepath=filepath,
            hsv_hist=hsv_hist,
            hog_vec=hog_vec
        )

    def _extract_hsv_hist(self, img_bgr: np.ndarray) -> np.ndarray:
        """Calcola l'istogramma 3D nello spazio colore HSV e lo normalizza L1."""
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

    def _extract_hog(self, img_bgr: np.ndarray) -> np.ndarray:
        """Ridimensiona l'immagine e calcola il vettore HOG nativo."""
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

    def compute_similarity(self, desc1: PhotoDescriptor, desc2: PhotoDescriptor) -> float:
        """
        Calcola la similarità composita tra due descrittori.
        
        :return: Punteggio tra 0.0 (completamente diverse) e 1.0 (duplicato visivo).
        """
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