"""
Modulo: enums.py
Descrizione: Enumerazione delle categorie di immagini per la segmentazione ad albero.
"""

from enum import Enum


class ImageType(Enum):
    """Macro-categorie di immagini identificate secondo la segmentazione JBIG2."""
    DOCUMENT = "DOC"
    PHOTO = "PHOTO"