"""
Modulo: doc_extractor.py
Descrizione: Estrattore di feature nativo per documenti (DOC) basato su
             ORB Keypoint Matching e Profili di Proiezione Orizzontale/Verticale.
Corso: Trattamento Dati Multimediali (TDM)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2


@dataclass
class DocDescriptor:
    """Rappresentazione dei descrittori locali ORB e del profilo di layout del documento."""
    filepath: str
    descriptors: Optional[np.ndarray]   # Descrittori binari ORB (N x 32 uint8)
    proj_v: np.ndarray                  # Profilo di proiezione verticale (layout righe)
    proj_h: np.ndarray                  # Profilo di proiezione orizzontale (layout colonne)


class DocFeatureExtractor:
    """
    Estrattore nativo per documenti basato su ORB (OpenCV) e Profili di Proiezione.
    """

    def __init__(self, max_features: int = 500, match_threshold: float = 0.75):
        """
        :param max_features: Numero massimo di keypoint ORB da estrarre per pagina.
        :param match_threshold: Tolleranza per il Ratio Test di Lowe sui descrittori.
        """
        self.max_features = max_features
        self.match_threshold = match_threshold
        self.orb = cv2.ORB_create(nfeatures=self.max_features)
        # Matcher brute-force con metrica di Hamming per descrittori binari
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def extract(self, filepath: str) -> DocDescriptor:
        """Estrae i descrittori ORB e i profili di proiezione dal documento."""
        img_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return DocDescriptor(filepath, None, np.zeros(0), np.zeros(0))

        # 1. Binarizzazione temporanea per la pulizia del testo
        _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 2. Estrazione Keypoints e Descrittori ORB (Nativi)
        _, descriptors = self.orb.detectAndCompute(img_gray, None)

        # 3. Profili di Proiezione (Somma dei pixel lungo gli assi per catturare la struttura del testo)
        # Ridimensioniamo a una griglia fissa per rendere i profili confrontabili
        resized_bin = cv2.resize(binary, (256, 256), interpolation=cv2.INTER_AREA)
        proj_v = np.sum(resized_bin, axis=1, dtype=np.float32)  # Somma per riga (righe di testo)
        proj_h = np.sum(resized_bin, axis=0, dtype=np.float32)  # Somma per colonna (margini e colonne)

        # Normalizzazione L2 dei profili
        norm_v = np.linalg.norm(proj_v)
        norm_h = np.linalg.norm(proj_h)
        proj_v = proj_v / norm_v if norm_v > 0 else proj_v
        proj_h = proj_h / norm_h if norm_h > 0 else proj_h

        return DocDescriptor(
            filepath=filepath,
            descriptors=descriptors,
            proj_v=proj_v,
            proj_h=proj_h
        )

    def compute_similarity(self, desc1: DocDescriptor, desc2: DocDescriptor) -> float:
        """
        Calcola la similarità composita tra due documenti.
        
        :return: Punteggio tra 0.0 e 1.0.
        """
        # 1. Similarità Strutturale dei Profili di Proiezione (Cosine Similarity)
        sim_layout = 0.0
        if desc1.proj_v.size > 0 and desc2.proj_v.size > 0:
            sim_v = float(np.dot(desc1.proj_v, desc2.proj_v))
            sim_h = float(np.dot(desc1.proj_h, desc2.proj_h))
            sim_layout = (sim_v + sim_h) / 2.0

        # 2. Keypoint Matching con ORB (Lowe's Ratio Test)
        sim_orb = 0.0
        if desc1.descriptors is not None and desc2.descriptors is not None:
            if len(desc1.descriptors) >= 2 and len(desc2.descriptors) >= 2:
                # Trova i 2 migliori match per ciascun descrittore
                matches = self.bf_matcher.knnMatch(desc1.descriptors, desc2.descriptors, k=2)
                
                good_matches = 0
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        # Ratio Test di Lowe per eliminare i falsi positivi
                        if m.distance < self.match_threshold * n.distance:
                            good_matches += 1

                min_keypoints = min(len(desc1.descriptors), len(desc2.descriptors))
                sim_orb = good_matches / float(min_keypoints) if min_keypoints > 0 else 0.0

        # Fusione pesata: 60% corrispondenza simboli ORB, 40% layout del testo
        similarity = 0.6 * sim_orb + 0.4 * sim_layout
        return round(float(min(1.0, max(0.0, similarity))), 4)