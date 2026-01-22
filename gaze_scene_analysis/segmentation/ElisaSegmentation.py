from gaze_scene_analysis.segmentation import SegmentationInterface
from gaze_scene_analysis.types import FrameData, LookedObject
from PIL import Image
import requests
import torch
from torch import nn
import logging
import os
import warnings
import random
import cv2

# Suppress TensorFlow messages (already in your code, but for completeness)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Suppress Transformers specific warnings
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress general Python UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du device: {device}")

if not torch.cuda.is_available():
    print("ATTENTION: GPU non disponible, utilisation du CPU")

processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-semantic", use_fast=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-cityscapes-semantic")

# Déplacer le modèle sur le GPU
model = model.to(device)
model.eval()

# Fixation d'une palette
np.random.seed(42) 
FIXED_PALETTE = np.array([list(np.random.choice(range(256), size=3)) for _ in range(model.config.num_queries + 1)])

# Basé sur le mapping officiel : 0: road, 1: sidewalk, 2: building, etc.
CITYSCAPES_MAP = {
    0: "route",
    1: "route",      # sidewalk
    2: "batiment",   # building
    3: "batiment",   # wall
    4: "batiment",   # fence
    5: "poteau",     # pole
    6: "feu de signalisation",
    7: "panneau",
    8: "batiment",   # vegetation
    11: "humain",    # person
    12: "humain",    # rider
    13: "vehicule motorise", # car
    14: "vehicule motorise", # truck
    15: "vehicule motorise", # bus
    17: "vehicule motorise", # motorcycle
    18: "velo"       # bicycle
}

class ElisaSegmentation(SegmentationInterface):
    def segment(self, frame: FrameData) -> LookedObject | None:
        
        affichage = False # Mettre à True pour voir affichage du résultat
        
        image = Image.fromarray(frame.image) if isinstance(frame.image, np.ndarray) else frame.image
        
        inputs = processor(images=frame.image, return_tensors="pt").to(device)   
        with torch.no_grad():
            outputs = model(**inputs)
            
            
        target_size = image.size[::-1]
        # Cette fonction interne reconstruit la carte sémantique à partir des requêtes
        semantic_segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[target_size])[0]
        
        # On utilise une méthode plus robuste pour obtenir la carte de probabilités
        # On combine les prédictions de classes et les masques binaires
        mask_cls_probs = outputs.class_queries_logits.softmax(-1)[0]
        mask_pred_probs = outputs.masks_queries_logits.sigmoid()[0]
        
        # On calcule la probabilité sémantique globale par pixel
        # Probabilité = Somme sur les requêtes de (P(classe) * P(masque))
        semantic_scores = torch.einsum("qc,qhw->chw", mask_cls_probs, mask_pred_probs)
        semantic_probs = semantic_scores / (semantic_scores.sum(dim=0, keepdim=True) + 1e-6)
        
        # Redimensionnement à la taille de l'image d'origine
        semantic_probs = torch.nn.functional.interpolate(
            semantic_probs.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        )[0]

        # 3. Extraction de la classe et de la confiance au point du regard
        try:
            gaze_x, gaze_y = int(frame.gaze_point[0]), int(frame.gaze_point[1])
            h, w = semantic_segmentation.shape
            
            if 0 <= gaze_x < w and 0 <= gaze_y < h:
                label_id = int(semantic_segmentation[gaze_y, gaze_x])
                class_name = CITYSCAPES_MAP.get(label_id, "inconnu")
                
                # La confiance est la probabilité de la classe prédite à cet endroit précis
                confidence = float(semantic_probs[label_id, gaze_y, gaze_x].item())
            else:
                class_name = "hors-champ"
                confidence = 0.0
        except (TypeError, IndexError):
            class_name = "inconnu"
            confidence = 0.0
        
        
        # Code affichage
        if affichage :
            seg = semantic_segmentation.cpu().numpy()
            color_seg = FIXED_PALETTE[seg].astype(np.uint8)
            
            # Cercle là où l'utilisateur regarde
            img = np.array(image) * 0.5 + (color_seg[..., ::-1]) * 0.5
            img = img.astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            text = f"({confidence:.2%})"
                
            if class_name != "hors-champ":
                cv2.circle(img_bgr, (gaze_x, gaze_y), 10, (0, 255, 0), 2) # Cercle vert au point de regard
                cv2.putText(img_bgr, class_name, (gaze_x + 15, gaze_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img_bgr, text, (gaze_x + 15, gaze_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Gaze Segmentation", img_bgr)
            cv2.waitKey(1)
        

        return LookedObject(class_name=class_name, confidence=confidence) 