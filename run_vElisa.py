# Pipeline complet pour Urban Gaze Analysis

import sys
from gaze_scene_analysis.loaders.VideoLoader import VideoLoader
from gaze_scene_analysis.preprocessing.DummyPreprocessor import DummyPreprocessor
from gaze_scene_analysis.segmentation.DummySegmentation import DummySegmentation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def main():
	# Chemin du dossier de données
	data_folder = "data/2025-11-20_15-30-11-a3a383b4"
    #data_folder = "data/2025-11-20_15-40-17-10b70589"

	# 1. Chargement des données vidéo et capteurs
	loader = VideoLoader(data_folder)

	# 2. Prétraitement (dummy)
	preprocessor = DummyPreprocessor()

	# 3. Segmentation (dummy)
	segmenter = DummySegmentation()

	# Chronique temporelle : liste des résultats (frame_id, timestamp, classe)
	timeline = []

	for frame in tqdm(loader, total=len(loader), desc="Traitement des frames"):
		# Prétraitement
		processed = preprocessor.process(frame)
		if processed is None:
			timeline.append((frame.frame_id, frame.timestamp, None))
			continue

		# Segmentation : renvoie directement l'objet regardé (ou None)
		looked_object = segmenter.segment(processed)
		# On stocke la classe et la confiance de l'objet regardé (ou None)
		if looked_object is not None:
			timeline.append((frame.frame_id, frame.timestamp, looked_object.class_name, looked_object.confidence))
		else:
			timeline.append((frame.frame_id, frame.timestamp, None, None))

	# Affichage graphique
	classes = [cl for _, _, cl, _ in timeline if cl is not None]
	unique_classes = list(sorted(set(classes)))
	if not unique_classes:
		print("Aucune classe détectée, rien à afficher.")
		return

	class_to_y = {cl: i for i, cl in enumerate(unique_classes)}
	colors = plt.get_cmap('tab20', len(unique_classes))

	plt.figure(figsize=(14, 4))
	for i, cl in enumerate(unique_classes):
		xs = [fid for fid, _, c, _ in timeline if c == cl]
		plt.bar(xs, [1]*len(xs), bottom=[i]*len(xs), color=colors(i), edgecolor='k', linewidth=0.5, width=1.0, label=cl)

	plt.yticks(np.arange(len(unique_classes)) + 0.5, unique_classes)
	plt.xlabel("Frame Number")
	plt.title("Analyse oculométrique sur frames labellisées")
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	main()
