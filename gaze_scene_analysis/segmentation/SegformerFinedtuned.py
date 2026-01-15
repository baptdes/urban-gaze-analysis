from gaze_scene_analysis.segmentation import SegmentationInterface
from gaze_scene_analysis.loaders.VideoLoader import VideoLoader
from gaze_scene_analysis.types import FrameData, LookedObject
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy import ndimage

CITYSCAPES_MAP = {
    0: "route",
    1: "route",      # sidewalk -> route
    2: "batiment",   # building
    3: "batiment",   # wall
    4: "batiment",   # fence
    5: "poteau",     # pole
    6: "feu de signalisation",
    7: "panneau",
    8: "vegetation",   # vegetation
    9: None,   # terrain
    10: None,   # sky
    11: "humain",    # person
    12: "humain",    # rider
    13: "vehicule motorise", # car
    14: "vehicule motorise", # truck
    15: "vehicule motorise", # bus
    17: "vehicule motorise", # motorcycle
    18: "velo"       # bicycle
}

NEUTRAL_COLOR = (128, 128, 128)  # Gris pour les classes non mappées

class SegformerFinedtuned(SegmentationInterface):
    """Implementation de l'interface de segmentation avec Segformer."""
    def __init__(self, verbose: bool = True):
        model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if verbose:
            print(f"Utilisation de l'appareil : {self.device}")
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            use_safetensors=True
        )
        self.model.to(self.device)
        if verbose:
            print("Id to class name mapping:")
            id2label = self.model.config.id2label
            for id, name in id2label.items():
                print(f"  {id}: {name}")

    def get_segmentation(self, image: np.ndarray, 
                        confidence_threshold: float = 0.75,
                        min_region_size: int = 150) -> tuple[np.ndarray, torch.Tensor]:
        """Version simplifiée et optimisée."""
        h, w = image.shape[:2]
        
        # Prédiction
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.feature_extractor(images=image_pil, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        logits = F.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )
        
        probs = torch.softmax(logits, dim=1)
        max_probs, segmentation = probs.max(dim=1)
        
        segmentation = segmentation.squeeze(0).cpu().detach().numpy()
        max_probs = max_probs.squeeze(0).cpu().detach().numpy()
        
        confidence_mask = (max_probs >= confidence_threshold)
        segmentation = ndimage.median_filter(segmentation, size=3)
        
        return segmentation, probs
    
    def segment(self, frame: FrameData) -> LookedObject | None:
        """Retourne l'objet regarde selon le point de gaze, mappe à nos classes."""
        segmentation, probs = self.get_segmentation(frame.image)
        gaze_x, gaze_y = frame.gaze_point
        gaze_x = int(round(gaze_x))
        gaze_y = int(round(gaze_y))
        label_id = int(segmentation[gaze_y, gaze_x])
        mapped_class_name = CITYSCAPES_MAP.get(label_id)
        confidence = float(probs[0, label_id, gaze_y, gaze_x].item())
        if mapped_class_name is None:
            return None
        return LookedObject(
            class_name=mapped_class_name,
            confidence=confidence
        )

if __name__ == "__main__":
    from matplotlib.patches import Patch

    # --- Initialisation ---
    video_folder = r"data\Cyclistes\2025-11-20_15-30-11-a3a383b4"
    loader = VideoLoader(video_folder)
    segmenter = SegformerFinedtuned()

    # Preparer la fenêtre OpenCV
    window_name = "Segmentation en temps reel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # Palette de couleurs pour la segmentation
    cmap = plt.cm.tab20

    skip_frames = 100

    for frame_data in loader:
        if frame_data.frame_id < skip_frames:
            continue

        # --- Segmentation ---
        segmentation, probs = segmenter.get_segmentation(frame_data.image)
        unique_labels = np.unique(segmentation)

        # --- Creer une image couleur pour l'affichage ---
        seg_color = np.zeros_like(frame_data.image, dtype=np.uint8)
        for label in unique_labels:
            if CITYSCAPES_MAP.get(label) is None:
                seg_color[mask] = NEUTRAL_COLOR
                continue
            mask = segmentation == label
            rgb_color = np.array(cmap(label / 19.0)[:3]) * 255
            bgr_color = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))
            seg_color[mask] = bgr_color

        # Melanger l'image originale et la segmentation
        alpha = 0.6
        overlay = cv2.addWeighted(frame_data.image, 1 - alpha, seg_color, alpha, 0)

        # Dessiner le point de regard
        gaze_x, gaze_y = frame_data.gaze_point
        cv2.circle(overlay, (int(gaze_x), int(gaze_y)), 10, (0, 255, 0), 2)
        cv2.circle(overlay, (int(gaze_x), int(gaze_y)), 3, (0, 0, 255), -1)

        # --- Fusion des labels mappes pour l'affichage ---
        mapped_label_to_info = {}
        for label in unique_labels:
            mapped_class_name = CITYSCAPES_MAP.get(label)
            if mapped_class_name is None:
                continue
            if mapped_class_name not in mapped_label_to_info:
                # Couleur basee sur le premier label rencontre
                rgb_color = np.array(cmap(label / 19.0)[:3]) * 255
                bgr_color = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))
                mapped_label_to_info[mapped_class_name] = {
                    "bgr_color": bgr_color
                }

        # Ajouter les labels fusionnes comme texte
        y0 = 20
        dy = 20
        for i, (mapped_class_name, info) in enumerate(mapped_label_to_info.items()):
            text = f"{mapped_class_name}"
            bgr_color = info["bgr_color"]
            cv2.putText(overlay, text, (10, y0 + i * dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 1, cv2.LINE_AA)

        # Affichage
        cv2.imshow(window_name, overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

