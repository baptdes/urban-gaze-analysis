from abc import ABC, abstractmethod
from gaze_scene_analysis.types import FrameData

class PreprocessorInterface(ABC):
    """Interface pour tout module de prétraitement."""

    @abstractmethod
    def process(self, frame: FrameData) -> FrameData | None:
        """
        Traite une frame.
        Retourne la frame modifiée ou None si elle doit être ignorée.
        """
        pass