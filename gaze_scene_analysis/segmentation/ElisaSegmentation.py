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

# 1. Suppress TensorFlow messages (already in your code, but for completeness)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# 2. Suppress Transformers specific warnings
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

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

processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-semantic", use_fast=True)
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-cityscapes-semantic")


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
        image = Image.fromarray(frame.image) if isinstance(frame.image, np.ndarray) else frame.image
        
        inputs = processor(images=frame.image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Carte de segmentation
        predicted_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        
        # Coordonnées du regard
        try:
            gaze_x, gaze_y = int(frame.gaze_point[0]), int(frame.gaze_point[1])
            
            # Vérifier si le regard est bien dans les limites de l'image
            h, w = predicted_map.shape
            if 0 <= gaze_x < w and 0 <= gaze_y < h:
                label_id = int(predicted_map[gaze_y, gaze_x])
                class_name = CITYSCAPES_MAP.get(label_id, "inconnu")
            else:
                class_name = "hors-champ"
        except (TypeError, IndexError):
            class_name = "inconnu"

        # Code affichage
        """
        color_palette = [list(np.random.choice(range(256), size=3)) for _ in range(len(model.config.id2label))]
        seg = predicted_map.cpu().numpy()
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        palette = np.array(color_palette)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        
        # Cercle là où l'utilisateur regarde
        img = np.array(image) * 0.5 + (color_seg[..., ::-1]) * 0.5
        img = img.astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if class_name != "hors-champ":
            cv2.circle(img_bgr, (gaze_x, gaze_y), 10, (0, 255, 0), 2) # Cercle vert au point de regard
            cv2.putText(img_bgr, class_name, (gaze_x + 15, gaze_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Gaze Segmentation", img_bgr)
        cv2.waitKey(1)  
        """
        return LookedObject(class_name=class_name, confidence=0.5) #Mask2Former ne prédit une classe que si la probabilité est supérieure à 0.5