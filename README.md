
# Urban Gaze Analysis Pipeline

Pipeline d'analyse de données issues de lunettes Pupil Labs en environnement urbain.

## Structure générale du projet

```
urban-gaze-analysis/
├── run.py                  # Script principal pour lancer la pipeline
├── requirements.txt        # Dépendances Python
├── data/                   # Données
├── gaze_scene_analysis/    # Code source du pipeline
│   ├── loaders/            # Chargement et synchronisation des données vidéo/capteurs
│   ├── preprocessing/      # Modules de prétraitement (interfaces et exemples)
│   └── segmentation/       # Modules de segmentation (interfaces et exemples)
└── tests/                  # Scripts de test
```

## Installation

Créez un environnement (recommandé avec conda) et installez les dépendances :
	```sh
	conda create -n APP python=3.10
	conda activate APP
	pip install -r requirements.txt
	```

## Lancement de la pipeline

Pour exécuter le pipeline principal :

```sh
python run.py
```

Ou pour lancer un module spécifique (ex : tests ou fichiers du module) :

```sh
python -m gaze_scene_analysis.loaders.VideoLoader
```

ou encore 

```sh
python -m tests.VideoLoader_speedtest
```

## Fonctionnement général

Le pipeline suit les étapes suivantes :
1. Chargement des données synchronisées (vidéo, regard, IMU) via `VideoLoader`.
2. Prétraitement des frames (filtres, nettoyage, etc.) via un module de preprocessing.
3. Segmentation de la scène pour détecter l'objet regardé via un module de segmentation.
4. Génération d'une chronique temporelle des objets regardés.

## Ajouter une méthode de preprocessing

1. Créez un fichier dans `gaze_scene_analysis/preprocessing/` (ex : `MyPreprocessor.py`).
2. Implémentez la classe en héritant de `PreprocessorInterface` :

```python
from gaze_scene_analysis.preprocessing import PreprocessorInterface

class MyPreprocessor(PreprocessorInterface):
	 def process(self, frame):
		  # ... traitement ...
		  return frame
```
3. Utilisez votre préprocesseur dans `run.py` :
	```python
	from gaze_scene_analysis.preprocessing.MyPreprocessor import MyPreprocessor
	preprocessor = MyPreprocessor()
	```

## Ajouter une méthode de segmentation

1. Créez un fichier dans `gaze_scene_analysis/segmentation/` (ex : `MySegmentation.py`).
2. Implémentez la classe en héritant de `SegmentationInterface` :

```python
from gaze_scene_analysis.segmentation import SegmentationInterface

class MySegmentation(SegmentationInterface):
	 def segment(self, frame):
		  # ... détection d'objet ...
		  return LookedObject(class_name="car", confidence=0.9)
```
3. Utilisez votre segmentateur dans `run.py` :
	```python
	from gaze_scene_analysis.segmentation.MySegmentation import MySegmentation
	segmenter = MySegmentation()
	```