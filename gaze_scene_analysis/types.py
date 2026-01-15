# Nouveau type pour l'objet regardé
from typing import Optional
from dataclasses import dataclass
import numpy as np
from typing import Tuple

@dataclass
class FrameData:
    frame_id: int # Indice de la frame dans la séquence
    image: object
    gaze_point: tuple[int,int] # Coordonnées du point de regard (x, y) dans l'image
    gyro: Tuple[float, float, float] # Données gyroscopiques (x, y, z) en deg/s
    accel: Tuple[float, float, float] # Données d'accéléromètre (x, y, z) en g
    timestamp: np.int64 # Timestamp en ns

@dataclass
class LookedObject:
    class_name: str
    confidence: float