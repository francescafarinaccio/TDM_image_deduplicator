"""
Modulo: scanner.py
Descrizione: Servizio di scansione ricorsiva del filesystem,
             estrazione EXIF e calcolo dello Quality Score.
"""

import os
import math
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image, ExifTags

from .metadata import ImageMetadata


class ImageScanner:
    """
    Classe di servizio responsabile dell'esplorazione del file system,
    estrazione metadati e valutazione della qualità delle immagini.
    """
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    def __init__(self, w_res: float = 0.35, w_size: float = 0.20, w_blur: float = 0.45):
        self.w_res = w_res
        self.w_size = w_size
        self.w_blur = w_blur

    def scan_directory(self, root_dir: str) -> List[ImageMetadata]:
        """Scansiona ricorsivamente la directory target e restituisce la lista di ImageMetadata."""
        path_obj = Path(root_dir)
        if not path_obj.exists() or not path_obj.is_dir():
            raise ValueError(f"Directory non valida: {root_dir}")

        results: List[ImageMetadata] = []
        for file_path in path_obj.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    meta = self.process_image(str(file_path))
                    if meta:
                        results.append(meta)
                except Exception as e:
                    print(f"[WARN] Errore durante la scansione di {file_path}: {e}")
                    
        return results

    def process_image(self, filepath: str) -> Optional[ImageMetadata]:
        """Elabora un singolo file e istanzia un oggetto ImageMetadata."""
        p = Path(filepath)
        file_size = p.stat().st_size
        
        exif_data = {}
        try:
            with Image.open(filepath) as img:
                width, height = img.size
                channels = len(img.getbands())
                raw_exif = img._getexif()
                if raw_exif:
                    for tag_id, value in raw_exif.items():
                        tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                        exif_data[tag_name] = value
        except Exception:
            return None

        # Parsing EXIF
        datetime_orig = str(exif_data.get('DateTimeOriginal', '')) or None
        make = str(exif_data.get('Make', '')).strip() or None
        model = str(exif_data.get('Model', '')).strip() or None
        iso = exif_data.get('ISOSpeedRatings', None)
        focal = exif_data.get('FocalLength', None)
        if focal and isinstance(focal, tuple):
            focal = float(focal[0]) / float(focal[1]) if focal[1] != 0 else None

        aspect_ratio = round(width / float(height), 4) if height > 0 else 0.0
        megapixels = round((width * height) / 1e6, 2)

        blur_score, sobel_score = self._compute_sharpness_scores(filepath)
        quality_score = self._compute_quality_score(megapixels, file_size, blur_score)

        return ImageMetadata(
            filepath=str(p.resolve()),
            filename=p.name,
            extension=p.suffix.lower(),
            file_size_bytes=file_size,
            width=width,
            height=height,
            channels=channels,
            aspect_ratio=aspect_ratio,
            megapixels=megapixels,
            datetime_original=datetime_orig,
            camera_make=make,
            camera_model=model,
            iso=iso if isinstance(iso, int) else None,
            focal_length=float(focal) if focal else None,
            blur_score=blur_score,
            sobel_score=sobel_score,
            quality_score=quality_score
        )

    #funzione per calcorlare la varianza del Laplaciano e del gradiente di Sobel che indicano la nitidezza dell'immagine
    def _compute_sharpness_scores(self, filepath: str) -> Tuple[float, float]: 
        """Calcola la varianza del Laplaciano e del gradiente di Sobel."""
        img_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return 0.0, 0.0

        laplacian_var = float(cv2.Laplacian(img_gray, cv2.CV_64F).var())

        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_var = float(np.var(np.abs(sobel_x) + np.abs(sobel_y)))

        return round(laplacian_var, 2), round(sobel_var, 2)

    #funzione per calcolare lo score composito logaritmico basato su megapixel, dimensione del file e nitidezza
    def _compute_quality_score(self, megapixels: float, file_size_bytes: int, blur_score: float) -> float:
        """Calcola lo score logaritmico composito."""
        norm_res = math.log10(megapixels * 1e6 + 1.0)
        norm_size = math.log10(file_size_bytes + 1.0)
        norm_blur = math.log10(blur_score + 1.0)

        composite = (
            self.w_res * norm_res +
            self.w_size * norm_size +
            self.w_blur * norm_blur
        )
        return round(composite, 4)