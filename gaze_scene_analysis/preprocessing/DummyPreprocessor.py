import numpy as np
from gaze_scene_analysis.preprocessing import PreprocessorInterface
from gaze_scene_analysis.types import FrameData

class DummyPreprocessor(PreprocessorInterface):
    """Préprocesseur factice qui ne modifie pas les frames."""
    
    def __init__(self):
        pass
        
    def process(self, frame: FrameData) -> FrameData | None:
        return frame
