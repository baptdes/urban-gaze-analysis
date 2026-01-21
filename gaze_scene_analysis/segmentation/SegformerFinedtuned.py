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
from tqdm import tqdm

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

            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            print(f"Nombre total de paramètres : {total_params}")
            print(f"Nombre de paramètres entraînables : {trainable_params}")

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
    
    def segment(self, frame: FrameData, gaze_radius: int = 20) -> LookedObject | None:
        """
        Retourne l'objet regardé selon le point de gaze, en considérant une petite zone autour.
        
        Args:
            frame: FrameData contenant l'image et le point de gaze.
            gaze_radius: rayon (en pixels) autour du point de gaze pour prendre en compte les pixels voisins.
        """
        segmentation, probs = self.get_segmentation(frame.image)
        h, w = segmentation.shape

        gaze_x, gaze_y = frame.gaze_point
        gaze_x = int(round(gaze_x))
        gaze_y = int(round(gaze_y))

        # Définir la zone autour du gaze point
        x_min = max(0, gaze_x - gaze_radius)
        x_max = min(w, gaze_x + gaze_radius + 1)
        y_min = max(0, gaze_y - gaze_radius)
        y_max = min(h, gaze_y + gaze_radius + 1)

        # Extraire la zone autour du regard
        region_seg = segmentation[y_min:y_max, x_min:x_max]
        region_probs = probs[0, :, y_min:y_max, x_min:x_max]  # shape: (num_classes, H, W)

        # Calculer la probabilité moyenne par classe dans la région
        avg_probs_per_class = region_probs.mean(dim=(1, 2))  # shape: (num_classes,)
        label_id = int(avg_probs_per_class.argmax().item())
        confidence = float(avg_probs_per_class[label_id].item())

        mapped_class_name = CITYSCAPES_MAP.get(label_id)
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

    # Chronique temporelle pour le graphique final
    timeline = []

    for frame_data in tqdm(loader, desc="Traitement des frames"):
        if frame_data.frame_id < skip_frames or frame_data.frame_id % 10 != 0:
            continue
        
        # --- Segmentation ---
        segmentation, probs = segmenter.get_segmentation(frame_data.image)
        unique_labels = np.unique(segmentation)
        
        # --- Creer une image couleur pour l'affichage ---
        seg_color = np.zeros_like(frame_data.image, dtype=np.uint8)
        for label in unique_labels:
            if CITYSCAPES_MAP.get(label) is None:
                mask = segmentation == label
                seg_color[mask] = NEUTRAL_COLOR
                continue
            mask = segmentation == label
            rgb_color = np.array(cmap(label / 19.0)[:3]) * 255
            bgr_color = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))
            seg_color[mask] = bgr_color
        
        # Melanger l'image originale et la segmentation
        alpha = 0.6
        overlay = cv2.addWeighted(frame_data.image, 1 - alpha, seg_color, alpha, 0)
        
        # --- Objet regardé ---
        looked_object = segmenter.segment(frame_data)
        
        # Enregistrer pour la chronique temporelle
        if looked_object is not None:
            timeline.append((frame_data.frame_id, frame_data.timestamp, 
                            looked_object.class_name, looked_object.confidence))
            text_gaze = f"Regard: {looked_object.class_name} ({looked_object.confidence:.2f})"
            cv2.putText(overlay, text_gaze, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            timeline.append((frame_data.frame_id, frame_data.timestamp, None, None))
        
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

    # --- Affichage du graphique final ---
    print("\nGénération du graphique de chronique temporelle...")

    # Extraire les classes détectées
    classes = [cl for _, _, cl, _ in timeline if cl is not None]
    unique_classes = list(sorted(set(classes)))

    if not unique_classes:
        print("Aucune classe détectée, pas de graphique à afficher.")
    else:
        # Créer un mapping classe -> position Y
        class_to_y = {cl: i for i, cl in enumerate(unique_classes)}
        colors = plt.get_cmap('tab20', len(unique_classes))
        
        plt.figure(figsize=(14, 6))
        
        # Créer des barres pour chaque classe
        for i, cl in enumerate(unique_classes):
            xs = [fid for fid, _, c, _ in timeline if c == cl]
            if xs:
                plt.bar(xs, [1]*len(xs), bottom=[i]*len(xs), 
                    color=colors(i), edgecolor='k', linewidth=0.5, 
                    width=1.0, label=cl)
        
        plt.yticks(np.arange(len(unique_classes)) + 0.5, unique_classes)
        plt.xlabel("Frame Number")
        plt.ylabel("Classes détectées")
        plt.title("Chronique temporelle de l'analyse oculométrique")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Statistiques supplémentaires
        print("\n=== Statistiques ===")
        print(f"Nombre total de frames analysées: {len(timeline)}")
        print(f"Frames avec détection: {len(classes)}")
        print(f"Classes uniques détectées: {len(unique_classes)}")
        print("\nRépartition par classe:")
        for cl in unique_classes:
            count = classes.count(cl)
            percentage = (count / len(timeline)) * 100
            print(f"  {cl}: {count} frames ({percentage:.1f}%)")