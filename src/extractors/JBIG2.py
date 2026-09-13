"""
Modulo: JBIG2.py
Descrizione: Estrattore di feature e matcher per documenti basato sul
             Dizionario di Simboli e Soft Pattern Matching dello standard JBIG2.
Corso: Trattamento Dati Multimediali (TDM)
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
import cv2 


@dataclass
class JBIG2Descriptor:
    """Rappresentazione compatta del dizionario dei simboli di un documento."""
    filepath: str
    symbol_count: int
    symbol_dictionary: np.ndarray  # Tensor (N, grid_size, grid_size) di booleani
    aspect_ratios: List[float]


class JBIG2FeatureExtractor:
    """
    Estrattore basato sul Pattern Matching di JBIG2 per la deduplicazione
    di documenti scansionati, file di testo e grafica al tratto.
    """

    def __init__(self, grid_size: int = 16, max_symbols: int = 300, spm_tolerance: float = 0.15):
        """
        :param grid_size: Dimensione KxK della griglia di normalizzazione per ciascun simbolo.
        :param max_symbols: Numero massimo di simboli significativi da estrarre.
        :param spm_tolerance: Tolleranza errore XOR (epsilon) per considerare due simboli identici.
        """
        self.grid_size = grid_size
        self.max_symbols = max_symbols
        self.spm_tolerance = spm_tolerance

    def extract(self, filepath: str) -> JBIG2Descriptor:
        """
        Estrae il dizionario di simboli normalizzati dal documento.
        """
        img_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return JBIG2Descriptor(filepath, 0, np.zeros((0, self.grid_size, self.grid_size), dtype=bool), [])

        # 1. Binarizzazione con Otsu
        _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        total_pixels = binary.shape[0] * binary.shape[1]

        # 2. Estrazione componenti connesse
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        symbols = []
        aspect_ratios = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Filtriamo componenti fuori scala (rumore puntiforme o bordi pagina)
            if 12 <= area <= (total_pixels * 0.02) and w > 2 and h > 2:
                # Ritaglio della ROI del simbolo
                glyph = binary[y:y+h, x:x+w]

                # Normalizzazione della ROI a griglia fissa (grid_size x grid_size)
                resized_glyph = cv2.resize(glyph, (self.grid_size, self.grid_size), interpolation=cv2.INTER_AREA)
                bool_glyph = resized_glyph > 127

                symbols.append(bool_glyph)
                aspect_ratios.append(w / float(h))

                if len(symbols) >= self.max_symbols:
                    break

        symbol_tensor = np.array(symbols, dtype=bool) if symbols else np.zeros((0, self.grid_size, self.grid_size), dtype=bool)

        return JBIG2Descriptor(
            filepath=filepath,
            symbol_count=len(symbols),
            symbol_dictionary=symbol_tensor,
            aspect_ratios=aspect_ratios
        )

    def compute_similarity(self, desc1: JBIG2Descriptor, desc2: JBIG2Descriptor) -> float:
        """
        Calcola l'indice di similarità Soft Pattern Matching (SPM) tra due descrittori.
        
        :return: Valore di similarità compreso tra 0.0 (zero sovrapposizione) e 1.0 (duplicato perfetto).
        """
        dict1 = desc1.symbol_dictionary
        dict2 = desc2.symbol_dictionary

        if len(dict1) == 0 or len(dict2) == 0:
            return 0.0

        matches = 0
        pixels_per_symbol = float(self.grid_size * self.grid_size)

        # Per ciascun simbolo nel primo documento, cerchiamo un match morbido nel secondo
        for sym1 in dict1:
            # Calcolo vettorizzato dell'errore XOR contro tutti i simboli del secondo dizionario
            xor_diffs = np.logical_xor(sym1, dict2)
            error_rates = np.sum(xor_diffs, axis=(1, 2)) / pixels_per_symbol

            min_error = np.min(error_rates)
            if min_error <= self.spm_tolerance:
                matches += 1

        # Indice di similarità normalizzato rispetto alla dimensione del dizionario più piccolo
        min_dict_size = min(len(dict1), len(dict2))
        similarity = matches / float(min_dict_size)

        return round(float(similarity), 4)