from gaze_scene_analysis.segmentation import SegmentationInterface
from gaze_scene_analysis.types import FrameData, LookedObject

OBJECTS : list[str] = [ # en français, et en commentaire le nom du label dans le Cityscapes dataset
    "humain", # person + rider
    "véhicule motorisé", # car + truck + bus + motorcycle
    "vélo", # bicycle
    "panneau", # traffic sign
    "feu de signalisation", # traffic light
    "poteau", # pole
    "batiment", # building
    "route" # road
]

class ElisaSegmentation(SegmentationInterface):
    """Implémentation factice de l'interface de segmentation."""

    def segment(self, frame: FrameData) -> LookedObject | None:
        """Retourne un objet regardé factice ou None."""
        import random

        if random.random() < 0.2:
            # 20% de chances de ne rien regarder
            return None

        class_name = random.choice(OBJECTS)
        confidence = random.uniform(0.5, 1.0)

        return LookedObject(class_name=class_name, confidence=confidence)