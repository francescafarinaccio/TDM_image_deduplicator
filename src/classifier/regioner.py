 #Classificatore di contenuto basato sui principi di segmentazione di regione dello standard JBIG2 e MPEG-7. Distinzione tra Documenti/Grafica e Fotografie/Scene naturali.


from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import cv2

from .enums import ImageType


class ImageClassifier:

    def __init__(
        self,
        max_unique_colors_ratio: float = 0.05,
        otsu_separability_threshold: float = 0.60,
        symbol_density_range: Tuple[float, float] = (0.0005, 0.08)
    ):
        """
        :param max_unique_colors_ratio: Soglia massima del rapporto tra colori unici e totale pixel.
        :param otsu_separability_threshold: Soglia di separabilità delle classi dell'istogramma (Otsu).
        :param symbol_density_range: Range di densità accettabile per componenti connesse di tipo 'testo'.
        """
        self.max_unique_colors_ratio = max_unique_colors_ratio
        self.otsu_separability_threshold = otsu_separability_threshold
        self.min_symbol_density, self.max_symbol_density = symbol_density_range

    #funzione principale di classificazione, richiamata da scanner.py
    def classify(self, filepath: str) -> Tuple[ImageType, Dict[str, Any]]:
        """
        :param filepath: Percorso completo dell'immagine
        :return: Tupla (ImageType.DOCUMENT | ImageType.PHOTO, dizionario_metriche)
        """
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            # Fallback di sicurezza in caso di errore di caricamento
            return ImageType.PHOTO, {"error": "Failed to read image"}

        h, w, c = img_bgr.shape
        total_pixels = h * w

        # 1. Analisi della Tavolozza Cromatico-Saturazione
        unique_colors_ratio, avg_saturation = self._analyze_color_diversity(img_bgr, total_pixels)

        # 2. Converti in scala di grigi per l'analisi strutturale JBIG2
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 3. Binarizzazione di Otsu e Misura della Separabilità dell'Istogramma
        binary_img, otsu_score = self._compute_otsu_separability(img_gray)

        # 4. Analisi dei Componenti Connessi (JBIG2 Symbol Extracting)
        #JBIG2 deriva dallo standard di compressione per immagini binarie e decompone una pagine
        #in regioni omogenee in base al contenuto
        symbol_density, avg_aspect_ratio = self._analyze_connected_components(binary_img, total_pixels)

        # 5. Albero Decisionale per la Classificazione
        is_document = (
            (unique_colors_ratio <= self.max_unique_colors_ratio or avg_saturation < 25.0) and
            (otsu_score >= self.otsu_separability_threshold) and
            (self.min_symbol_density <= symbol_density <= self.max_symbol_density)
        )

        classified_type = ImageType.DOCUMENT if is_document else ImageType.PHOTO

        metrics = {
            "unique_colors_ratio": round(unique_colors_ratio, 5),
            "avg_saturation": round(avg_saturation, 2),
            "otsu_separability_score": round(otsu_score, 4),
            "symbol_density": round(symbol_density, 5),
            "avg_symbol_aspect_ratio": round(avg_aspect_ratio, 2)
        }

        return classified_type, metrics

    def _analyze_color_diversity(self, img_bgr: np.ndarray, total_pixels: int) -> Tuple[float, float]:
        # Convertiamo in HSV per misurare la saturazione media
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        avg_saturation = float(np.mean(img_hsv[:, :, 1]))

        # Ridimensioniamo per velocizzare il conteggio dei colori unici
        small_img = cv2.resize(img_bgr, (150, 150), interpolation=cv2.INTER_NEAREST)
        pixels = small_img.reshape(-1, 3)
        unique_colors = len(np.unique(pixels, axis=0))
        unique_ratio = unique_colors / float(pixels.shape[0])

        return unique_ratio, avg_saturation

    #funzione che calcola la soglia ottima di Otsu e la misura di separabilità delle classi dell'istogramma
    def _compute_otsu_separability(self, img_gray: np.ndarray) -> Tuple[np.ndarray, float]:
       
        # Calcolo istogramma
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).ravel()
        hist_norm = hist / float(hist.sum())

        # Varianza totale della grana dell'immagine
        pixel_values = np.arange(256)
        mean_total = np.sum(pixel_values * hist_norm)
        var_total = np.sum(((pixel_values - mean_total) ** 2) * hist_norm)

        # Binarizzazione con Otsu
        otsu_thresh, binary_img = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Calcolo varianza tra le classi (Inter-Class Variance)
        w0 = np.sum(hist_norm[:int(otsu_thresh)])
        w1 = 1.0 - w0

        if w0 == 0 or w1 == 0 or var_total == 0:
            return binary_img, 0.0

        mean0 = np.sum(pixel_values[:int(otsu_thresh)] * hist_norm[:int(otsu_thresh)]) / w0
        mean1 = np.sum(pixel_values[int(otsu_thresh):] * hist_norm[int(otsu_thresh):]) / w1

        var_between = w0 * w1 * ((mean0 - mean1) ** 2)
        
        # Metrica di separabilità dell'istogramma \eta \in [0, 1]
        separability_score = float(var_between / var_total)

        return binary_img, separability_score

    def _analyze_connected_components(self, binary_img: np.ndarray, total_pixels: int) -> Tuple[float, float]:
      
        #L'algoritmo analizza l'immagine binarizzata isolando le componenti connesse, ovvero i singoli gruppi di pixel neri adiacenti
        #densità: nm di oggetti distinti / area totale dell'immagine
        #avg aspect ratio: rapporto medio tra larghezza e altezza dei bounding box dei componenti connessi (i caratteri hanno tanti piccoli BB)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img)

        valid_symbols = 0
        aspect_ratios = []

        for i in range(1, num_labels):  # Escludiamo lo sfondo (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            w_box = stats[i, cv2.CC_STAT_WIDTH]
            h_box = stats[i, cv2.CC_STAT_HEIGHT]

            # Filtriamo rumore puntiforme ed elementi giganti (sfondi/riquadri)
            if 10 <= area <= (total_pixels * 0.05):
                valid_symbols += 1
                aspect_ratios.append(w_box / float(h_box) if h_box > 0 else 1.0)

        symbol_density = valid_symbols / float(total_pixels)
        avg_aspect_ratio = float(np.mean(aspect_ratios)) if aspect_ratios else 0.0

        return symbol_density, avg_aspect_ratio