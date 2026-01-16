# Pipeline complet pour Urban Gaze Analysis

import sys
from gaze_scene_analysis.loaders.VideoLoader import VideoLoader
from gaze_scene_analysis.preprocessing.ElisaPreprocessor import ElisaPreprocessor
from gaze_scene_analysis.segmentation.ElisaSegmentation import ElisaSegmentation
from gaze_scene_analysis.types import LookedObject
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
from collections import Counter

def most_frequent(class_list):
    """
    Identifie la classe la plus fréquente et renvoie un LookedObject correspondant.
    """
    # Filtrer les None
    clean_list = [c for c in class_list if c is not None]
    if not clean_list:
        return None

    occurence_count = Counter(clean_list)
    
    # Récupère la classe la plus fréquente
    most_common_class, count = occurence_count.most_common(1)[0]
    
    # Calcul d'un score de confiance basé sur la fréquence
    confidence_score = count / len(class_list)
    
    return LookedObject(class_name=most_common_class, confidence=confidence_score)



def main():
	# Chemin du dossier de données
	#data_folder = "data/2025-11-20_15-30-11-a3a383b4"
	data_folder = "data/2025-11-20_15-40-17-10b70589"

	# 1. Chargement des données vidéo et capteurs
	loader = VideoLoader(data_folder)
    
    # 1bis. Afficher le point du regard sur l'image et créer tableau 2D contenant
    # l'emplacement du regard pour chaque frame
	"""
	matrice_regard = np.empty(shape=(len(loader),2), dtype='float')
	for frame_data in loader:
		print(f"Frame {frame_data.frame_id}: gaze={frame_data.gaze_point}")
		
		# Visualiser le point de regard sur l'image
		img_display = frame_data.image.copy()
		gaze_x, gaze_y = frame_data.gaze_point
		cv2.circle(img_display, (int(gaze_x), int(gaze_y)), 10, (0, 255, 0), 2)
		cv2.circle(img_display, (int(gaze_x), int(gaze_y)), 3, (0, 0, 255), -1)
  
		matrice_regard[frame_data.frame_id][0] = gaze_x
		matrice_regard[frame_data.frame_id][1] = gaze_y
		
		cv2.imshow("Frame", img_display)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q') or key == 27:
			break
	cv2.destroyAllWindows()
 	"""

	# 2. Prétraitement (dummy)
	preprocessor = ElisaPreprocessor()

	# 3. Segmentation (dummy)
	segmenter = ElisaSegmentation()

	# Chronique temporelle : liste des résultats (frame_id, timestamp, classe)
	start_frame = 67 # numéro de la frame à partir de laquelle on a l'image
	intervalle_segmentation = 1000 # Toutes les combien de frames on récupère une classe
	cpt_max = 10 # Nb de frames consécutives sur laquelle on effectue la segmentation pour récupérer la classe majoritaire
	timeline = []
	# Initialisation des tableaux où on sauvegarde les infos nécessaires pour la chroniques temporelle
	previous_10_frame_ids = np.empty(10, dtype=object)
	previous_10_frame_ts = np.empty(10, dtype=object)
	previous_10_classes = np.full(10, None, dtype=object)
	# Drapeau pour savoir si on est sur une phase de segmentation consécutives
	save = False
	# Compteur pour itérer sur les frames consécutives qu'on traite
	cpt = 0
	looked_object = None
 
	for frame in tqdm(loader, total=len(loader), desc="Traitement des frames"):
		# Prétraitement
		processed = preprocessor.process(frame)
		if processed is None:
			timeline.append((frame.frame_id, frame.timestamp, None))
			continue

		# Segmentation : renvoie directement l'objet regardé (ou None)
		if processed.frame_id>=start_frame :
			previous_looked_object = looked_object
			if  (processed.frame_id % intervalle_segmentation == 0 or processed.frame_id==start_frame) or save:
				looked_object = segmenter.segment(processed)
				save = True
				previous_10_classes[cpt] = looked_object.class_name
				previous_10_frame_ids[cpt] = frame.frame_id
				previous_10_frame_ts[cpt] = frame.timestamp
				cpt += 1
			else :
				looked_object = previous_looked_object
			
			# On enregistre le résultats pour les frames consécutives une fois qu'on les a parcourues
			if cpt == cpt_max :
				looked_object = most_frequent(previous_10_classes.tolist())
				print(f"\n{previous_10_classes} \nMOST FREQUENT = {looked_object.class_name} avec confiance = {looked_object.confidence}\n")
				# On stocke la classe et la confiance de l'objet regardé (ou None)
				if looked_object is not None:
					for id, ts in zip(previous_10_frame_ids, previous_10_frame_ts):
						timeline.append((id, ts, looked_object.class_name, looked_object.confidence))
				else:
					timeline.append((frame.frame_id, frame.timestamp, None, None))	
     
				# Remise à 0 du compteur et du drapeau
				cpt = 0
				save = False
    
			# Lorsqu'on segemente pas, on met le nom de la dernière classe qui a été vue
			elif cpt == 0:
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
