import time
import cv2
from pathlib import Path
from gaze_scene_analysis.loaders.VideoLoader import VideoLoader

def main():
    # Chemin vers le dossier contenant les données
    folder_path = r"data\Cyclistes\2025-11-20_15-30-11-a3a383b4"

    # Test avec VideoLoader
    loader = VideoLoader(folder_path)

    video_path = loader.get_video_path()

    start_time = time.time()
    frame_count_loader = 0
    for frame_data in loader:
        frame_count_loader += 1
    end_time = time.time()
    avg_time_loader = (end_time - start_time) / frame_count_loader if frame_count_loader > 0 else 0

    print(f"Temps moyen par frame avec VideoLoader : {avg_time_loader:.4f} secondes")

    # Test avec cv2.VideoCapture
    cap = cv2.VideoCapture(video_path)

    start_time = time.time()
    frame_count_cap = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count_cap += 1
    end_time = time.time()
    avg_time_cap = (end_time - start_time) / frame_count_cap if frame_count_cap > 0 else 0

    cap.release()
    print(f"Temps moyen par frame avec cv2.VideoCapture : {avg_time_cap:.4f} secondes")

if __name__ == "__main__":
    main()
