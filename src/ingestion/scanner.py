
#Servizio di scansione ricorsiva del filesystem,estrazione EXIF e calcolo dello Quality Score.

import os
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from PIL import Image, ExifTags
from .metadata import ImageMetadata

DEFAULT_DATASET_DIR = "./dataset"

class ImageScanner:
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    def __init__(
        self, 
        target_dir: str = DEFAULT_DATASET_DIR, 
        w_res: float = 0.35, 
        w_size: float = 0.20, 
        w_blur: float = 0.45
    ):
        """
        :param target_dir: Directory statica di default per la scansione.
        :param w_res: Peso risoluzione/megapixel nel Quality Score.
        :param w_size: Peso dimensione file nel Quality Score.
        :param w_blur: Peso nitidezza/blur nel Quality Score.
        """
        self.target_dir = Path(target_dir)
        self.w_res = w_res
        self.w_size = w_size
        self.w_blur = w_blur

    #funzione principale di scansione, richiamata da main.py. Restituisce una lista di dizionari con filepath, quality_score e metadati completi.
    def scan(self, root_dir: Optional[str] = None) -> List[Dict[str, Any]]:
     
        path_to_scan = root_dir if root_dir else str(self.target_dir)
        metadata_objects = self.scan_directory(path_to_scan)

        # Convertiamo gli oggetti ImageMetadata in dizionari per compatibilità con la pipeline
        results: List[Dict[str, Any]] = []
        for meta in metadata_objects:
            results.append({
                "filepath": meta.filepath,
                "quality_score": meta.quality_score,
                "filename": meta.filename,
                "megapixels": meta.megapixels,
                "file_size_bytes": meta.file_size_bytes,
                "blur_score": meta.blur_score,
                "metadata_obj": meta
            })

        return results

    #funzione che scansiona la directory e restituisce una lista di oggetti ImageMetadata
    def scan_directory(self, root_dir: Optional[str] = None) -> List[ImageMetadata]:

        target_path = Path(root_dir) if root_dir else self.target_dir

        if not target_path.exists():
            print(f"[!] La directory '{target_path}' non esiste. Creazione automatica in corso...")
            target_path.mkdir(parents=True, exist_ok=True)
            return []

        results: List[ImageMetadata] = []
        for file_path in target_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    meta = self.process_image(str(file_path))
                    if meta:
                        results.append(meta)
                except Exception as e:
                    print(f"[WARN] Errore durante la scansione di {file_path}: {e}")
                    
        return results

    #funzione che elabora un singolo file e restituisce un oggetto ImageMetadata
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

    #funzione che calcola i punteggi di nitidezza utilizzando Laplaciano e Sobel
    #filtro di Sobel per riconoscere i contorni e il filtro Laplaciano per la nitidezza
    def _compute_sharpness_scores(self, filepath: str) -> Tuple[float, float]: 

        img_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return 0.0, 0.0

        laplacian_var = float(cv2.Laplacian(img_gray, cv2.CV_64F).var())

        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_var = float(np.var(np.abs(sobel_x) + np.abs(sobel_y)))

        return round(laplacian_var, 2), round(sobel_var, 2)

    #funzione che calcola lo score di qualità composito basato su risoluzione, dimensione file e nitidezza
    #applico una normalizzazione logaritmica per evitare che valori estremi influenzino troppo il punteggio finale
    def _compute_quality_score(self, megapixels: float, file_size_bytes: int, blur_score: float) -> float:
        norm_res = math.log10(megapixels * 1e6 + 1.0)
        norm_size = math.log10(file_size_bytes + 1.0)
        norm_blur = math.log10(blur_score + 1.0)

        composite = (
            self.w_res * norm_res +
            self.w_size * norm_size +
            self.w_blur * norm_blur
        )
        return round(composite, 4)