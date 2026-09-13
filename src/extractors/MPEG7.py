"""
Modulo: mpeg7.py
Descrizione: Estrattore di feature visive conforme allo standard MPEG-7
             (Color Layout Descriptor ed Edge Histogram Descriptor).
Corso: Trattamento Dati Multimediali (TDM)
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2


@dataclass
class MPEG7Descriptor:
    """Rappresentazione compatta dei vettori MPEG-7 per un'immagine naturale."""
    filepath: str
    cld_y: np.ndarray    # Coefficienti DCT Y (lunghezza 6)
    cld_cb: np.ndarray   # Coefficienti DCT Cb (lunghezza 3)
    cld_cr: np.ndarray   # Coefficienti DCT Cr (lunghezza 3)
    ehd_hist: np.ndarray # Istogramma dei bordi a 80 bin


class MPEG7FeatureExtractor:
    """
    Estrattore di descrittori visivi MPEG-7 (CLD ed EHD) per la deduplicazione
    di immagini a toni continui e fotografie.
    """

    # Indici della scansione Zig-Zag 8x8 per estrarre le basse frequenze DCT
    ZIGZAG_INDEXES = [
        (0,0), (0,1), (1,0), (2,0), (1,1), (0,2),
        (0,3), (1,2), (2,1), (3,0)
    ]

    # Maschere dei 5 filtri di bordo per l'Edge Histogram Descriptor
    EDGE_KERNELS = [
        np.array([[1, -1], [1, -1]], dtype=np.float32),        # Verticale
        np.array([[1, 1], [-1, -1]], dtype=np.float32),        # Orizzontale
        np.array([[np.sqrt(2), 0], [0, -np.sqrt(2)]], dtype=np.float32),  # 45 gradi
        np.array([[0, np.sqrt(2)], [-np.sqrt(2), 0]], dtype=np.float32),  # 135 gradi
        np.array([[2, -2], [-2, 2]], dtype=np.float32)         # Non-direzionale
    ]

    def __init__(self, w_cld: float = 0.5, w_ehd: float = 0.5):
        """
        :param w_cld: Peso assegnato al Color Layout Descriptor nella distanza globale.
        :param w_ehd: Peso assegnato all'Edge Histogram Descriptor nella distanza globale.
        """
        self.w_cld = w_cld
        self.w_ehd = w_ehd

    def extract(self, filepath: str) -> MPEG7Descriptor:
        """
        Estrae i descrittori CLD ed EHD da un file immagine.
        """
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            return MPEG7Descriptor(
                filepath, np.zeros(6), np.zeros(3), np.zeros(3), np.zeros(80)
            )

        # 1. Estrazione Color Layout Descriptor (CLD)
        cld_y, cld_cb, cld_cr = self._extract_cld(img_bgr)

        # 2. Estrazione Edge Histogram Descriptor (EHD)
        ehd_hist = self._extract_ehd(img_bgr)

        return MPEG7Descriptor(
            filepath=filepath,
            cld_y=cld_y,
            cld_cb=cld_cb,
            cld_cr=cld_cr,
            ehd_hist=ehd_hist
        )

    def _extract_cld(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcola la DCT 2D sulla griglia 8x8 trasformata in YCbCr."""
        # Ridimensionamento a griglia 8x8
        resized_8x8 = cv2.resize(img_bgr, (8, 8), interpolation=cv2.INTER_AREA)
        
        # Conversione in YCbCr
        img_ycbcr = cv2.cvtColor(resized_8x8, cv2.COLOR_BGR2YCrCb)
        y_channel = img_ycbcr[:, :, 0].astype(np.float32)
        cr_channel = img_ycbcr[:, :, 1].astype(np.float32)
        cb_channel = img_ycbcr[:, :, 2].astype(np.float32)

        # Calcolo DCT 2D
        dct_y = cv2.dct(y_channel)
        dct_cb = cv2.dct(cb_channel)
        dct_cr = cv2.dct(cr_channel)

        # Scansione Zig-Zag
        y_coeff = np.array([dct_y[r, c] for r, c in self.ZIGZAG_INDEXES[:6]])
        cb_coeff = np.array([dct_cb[r, c] for r, c in self.ZIGZAG_INDEXES[:3]])
        cr_coeff = np.array([dct_cr[r, c] for r, c in self.ZIGZAG_INDEXES[:3]])

        return y_coeff, cb_coeff, cr_coeff

    def _extract_ehd(self, img_bgr: np.ndarray) -> np.ndarray:
        """Divide l'immagine in 16 sotto-regioni e calcola l'istogramma dei bordi a 80 bin."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Manteniamo le dimensioni divisibili per 4 e per i blocchi 2x2
        grid_h, grid_w = h // 4, w // 4
        ehd_histogram = np.zeros(80, dtype=np.float32)

        bin_idx = 0
        for row in range(4):
            for col in range(4):
                # Estraggo la sotto-regione locale (1/16 dell'immagine)
                sub_img = gray[row*grid_h:(row+1)*grid_h, col*grid_w:(col+1)*grid_w]
                sub_h, sub_w = sub_img.shape

                counts = np.zeros(5, dtype=np.float32)
                block_count = 0

                # Analizziamo la sotto-regione in micro-blocchi 2x2
                for y in range(0, sub_h - 1, 2):
                    for x in range(0, sub_w - 1, 2):
                        block = sub_img[y:y+2, x:x+2].astype(np.float32)
                        
                        # Calcolo delle 5 risposte dei filtri
                        responses = [np.abs(np.sum(block * k)) for k in self.EDGE_KERNELS]
                        max_resp_idx = int(np.argmax(responses))
                        max_val = responses[max_resp_idx]

                        # Soglia per determinare se il bordo è significativo
                        if max_val > 11.0:
                            counts[max_resp_idx] += 1.0
                        block_count += 1

                # Normalizzazione dell'istogramma locale della sotto-regione
                if block_count > 0:
                    counts /= float(block_count)

                ehd_histogram[bin_idx:bin_idx+5] = counts
                bin_idx += 5

        return ehd_histogram

    def compute_similarity(self, desc1: MPEG7Descriptor, desc2: MPEG7Descriptor) -> float:
        """
        Calcola l'indice di similarità composito tra due descrittori MPEG-7.
        
        :return: Valore di similarità compreso tra 0.0 e 1.0 (1.0 = immagini identiche).
        """
        # 1. Distanza Euclidea Pesata per CLD
        w_y = np.array([2.0, 1.5, 1.5, 1.0, 1.0, 1.0])
        w_chroma = np.array([2.0, 1.0, 1.0])

        d_y = np.sqrt(np.sum(w_y * ((desc1.cld_y - desc2.cld_y) ** 2)))
        d_cb = np.sqrt(np.sum(w_chroma * ((desc1.cld_cb - desc2.cld_cb) ** 2)))
        d_cr = np.sqrt(np.sum(w_chroma * ((desc1.cld_cr - desc2.cld_cr) ** 2)))
        
        d_cld = d_y + d_cb + d_cr

        # 2. Distanza L1 (Manhattan) per EHD
        d_ehd = np.sum(np.abs(desc1.ehd_hist - desc2.ehd_hist))

        # Normalizzazione empirica delle distanze in range [0, 1]
        norm_d_cld = min(d_cld / 150.0, 1.0)
        norm_d_ehd = min(d_ehd / 10.0, 1.0)

        d_total = self.w_cld * norm_d_cld + self.w_ehd * norm_d_ehd
        similarity = 1.0 - d_total

        return round(float(max(0.0, similarity)), 4)