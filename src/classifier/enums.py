
#Enumerazione delle categorie di immagini per la segmentazione ad albero.

from enum import Enum

#macro categorie di immagini per la classificazione
class ImageType(Enum):
    DOCUMENT = "DOC"
    PHOTO = "PHOTO"