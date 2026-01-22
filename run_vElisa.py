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



def most_frequent(class_list, confidence_list):
    """
    Retourne l'objet le plus fréquent avec un score combiné :
    Mix entre la fréquence d'apparition sur la fenetre de frames, et la confiance moyenne du modèle.
    """
    # 1. Associer classes et confiances, en filtrant les entrées invalides (None)
    valid_entries = [
        (cls, conf) 
        for cls, conf in zip(class_list, confidence_list) 
        if cls is not None and conf is not None
    ]
    
    # Si aucune détection valide dans la fenêtre, on retourne None
    if not valid_entries:
        return None

    # 2. Identifier la classe majoritaire
    all_classes = [entry[0] for entry in valid_entries]
    occurence_count = Counter(all_classes)
    most_common_class, count = occurence_count.most_common(1)[0]
    
    # 3. Calcul du Score de Fréquence (Stabilité)
    # count : nombre de fois où la classe gagnante est apparue
    # len(class_list) : taille totale de la fenêtre (y compris les None/Ratés)
    freq_score = count / len(class_list)
    
    # 4. Calcul du Score Moyen du Modèle pour cette classe
    # On ne prend la moyenne QUE des confiances associées à la classe gagnante
    relevant_confidences = [
        conf for cls, conf in valid_entries 
        if cls == most_common_class
    ]
    avg_model_score = sum(relevant_confidences) / len(relevant_confidences)
    
    # 5. Calcul du Score Final (Mix)
    # Ici : 50% importance pour la stabilité temporelle, 50% pour la confiance du réseau
    weight_freq = 0.5
    weight_model = 0.5
    
    mixed_confidence = (weight_freq * freq_score) + (weight_model * avg_model_score)
    
    return LookedObject(class_name=most_common_class, confidence=mixed_confidence)


def main():
	# Chemin du dossier de données
	depart = "n7"
	#depart = "alsace_lorraine"
	if (depart=="n7"):
		data_folder = "data/2025-11-20_15-30-11-a3a383b4"
	else:
		data_folder = "data/2025-11-20_15-40-17-10b70589"

	# 1. Chargement des données vidéo et capteurs
	loader = VideoLoader(data_folder)

	# 2. Prétraitement (dummy)
	preprocessor = ElisaPreprocessor()

	# 3. Segmentation
	segmenter = ElisaSegmentation()

	# Chronique temporelle : liste des résultats (frame_id, timestamp, classe)
	start_frame = 67 # numéro de la frame à partir de laquelle on a l'image
	intervalle_segmentation = 1000 # Toutes les combien de frames on récupère une classe
	cpt_max = 5 # Nb de frames consécutives sur laquelle on effectue la segmentation pour récupérer la classe majoritaire
	timeline = []
	# Initialisation des tableaux où on sauvegarde les infos nécessaires pour la chroniques temporelle
	previous_n_frame_ids = np.empty(cpt_max, dtype=object)
	previous_n_frame_ts = np.empty(cpt_max, dtype=object)
	previous_n_classes = np.full(cpt_max, None, dtype=object)
	previous_n_confidences = np.empty(cpt_max, dtype=object)
	# Drapeau pour savoir si on est sur une phase de segmentation consécutives
	save = False
	# Compteur pour itérer sur les frames consécutives qu'on traite
	cpt = 0
	looked_object = None
 
	for frame in tqdm(loader, total=len(loader), desc="Traitement des frames"):
		# Prétraitement
		processed = preprocessor.process(frame)
		if processed is None:
			timeline.append((frame.frame_id, frame.timestamp, None, None))
			continue

		# Segmentation : renvoie directement l'objet regardé (ou None)
		if processed.frame_id>=start_frame :
			previous_looked_object = looked_object
			if  (processed.frame_id % intervalle_segmentation == 0 or processed.frame_id==start_frame) or save:
				looked_object = segmenter.segment(processed)
				save = True
				previous_n_classes[cpt] = looked_object.class_name
				previous_n_confidences[cpt] =looked_object.confidence
				previous_n_frame_ids[cpt] = frame.frame_id
				previous_n_frame_ts[cpt] = frame.timestamp
				cpt += 1
			else :
				looked_object = previous_looked_object
			
			# On enregistre le résultats pour les frames consécutives une fois qu'on les a parcourues
			if cpt == cpt_max :
				looked_object = most_frequent(previous_n_classes.tolist(), previous_n_confidences.tolist())
				# On stocke la classe et la confiance de l'objet regardé (ou None)
				if looked_object is not None:
					for id, ts in zip(previous_n_frame_ids, previous_n_frame_ts):
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
			

	#---------------------- Analyse oculométrique -----------------------
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
	plt.xlabel("Numéro de Frame")
	plt.title("Analyse oculométrique sur frames labellisées")
	plt.tight_layout()
	filename = f"chronique_temp_{depart}_cptmax{cpt_max}_intervalle{intervalle_segmentation}.png"
	plt.savefig(filename, dpi=300) 
	print(f"Graphique sauvegardé sous : {filename}")
	plt.show()
 
	#---------------------- Plot score de confiance -----------------------
	plt.figure(figsize=(14, 5)) 
    # Extraction des données : on ne garde que les frames où une confiance a été calculée
    # Structure de timeline : (frame_id, timestamp, class_name, confidence)
	data_confiance = [(fid, conf) for fid, _, _, conf in timeline if conf is not None]
    # Séparation X (frames) et Y (confiance)
	xs_conf = [d[0] for d in data_confiance]
	ys_conf = [d[1] for d in data_confiance]
	# Tracé de la courbe
	#plt.plot(xs_conf, ys_conf, color='royalblue', linestyle='-', linewidth=1, label='Confiance Mixte')
	plt.scatter(xs_conf, ys_conf, s=2, alpha=0.6)
	plt.title(f"Évolution du score de confiance")
	plt.xlabel("Numéro de Frame")
	plt.ylabel("Confiance")
	plt.ylim(0, 1.05) # Fixe l'axe Y entre 0 et 1.05
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.tight_layout()
	# Sauvegarde
	filename_conf = f"confiance_{depart}_cptmax{cpt_max}_int{intervalle_segmentation}.png"
	plt.savefig(filename_conf, dpi=300)
	print(f"Graphique de confiance sauvegardé sous : {filename_conf}")
	plt.show()
    
if __name__ == "__main__":
	main()
