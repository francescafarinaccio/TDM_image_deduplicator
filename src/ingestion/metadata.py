
#Modello dati (Data Transfer Object) che incapsula le informazioni e gli indici di qualità di un'immagine

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass #decoratore per indicare che la classe è una dataclass
class ImageMetadata:
    filepath: str
    filename: str
    extension: str
    file_size_bytes: int
    width: int
    height: int
    channels: int
    aspect_ratio: float
    megapixels: float
    
    # Metadati EXIF
    #metadati opzionali, possono essere None se non presenti
    datetime_original: Optional[str] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    iso: Optional[int] = None
    focal_length: Optional[float] = None
    
    # Indicatori di Qualità Visiva
    blur_score: float = 0.0          # Varianza della risposta del filtro di Laplace
    sobel_score: float = 0.0         # Varianza del gradiente di Sobel
    quality_score: float = 0.0       # Score composito pesato

    #converte l'oggetto in un dizionario 
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)