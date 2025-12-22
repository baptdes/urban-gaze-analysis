from abc import ABC, abstractmethod
from gaze_scene_analysis.types import FrameData, LookedObject

class SegmentationInterface(ABC):
    """Interface pour segmenter une image."""

    @abstractmethod
    def segment(self, frame: FrameData) -> LookedObject | None:
        """
        Retourne l'objet regardé dans la frame, ou None si aucun objet n'est regardé.
        """
        pass