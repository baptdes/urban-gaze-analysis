import cv2
import json
import os
import numpy as np
import pandas as pd
from typing import Tuple, Iterator
from pathlib import Path
from gaze_scene_analysis.types import FrameData


def load_camera_params(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Charge les paramètres de calibration de la caméra."""
    with open(json_path, "r") as f:
        data = json.load(f)
    K = np.array(data["camera_matrix"], dtype=np.float32)
    D = np.array(data["distortion_coefficients"], dtype=np.float32)
    return K, D


def load_gaze_offset(json_path: str) -> np.ndarray:
    """Charge l'offset de calibration du regard."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return np.array(data.get("gaze_offset", [0.0, 0.0]), dtype=np.float32)


class VideoLoader:
    """
    Classe pour charger et itérer sur les frames vidéo avec données associées.
    Pour chaque frame, on fournit:
    - L'image (avec correction de distorsion)
    - Le point de regard interpolé ET corrigé de la distorsion
    - Les données IMU interpolées (gyro et accel)
    - Le timestamp de la frame
    """
    
    def __init__(self, folder_path: str):
        """
        Args:
            folder_path: Chemin du dossier contenant les données
        """
        self.folder_path = Path(folder_path)
        
        # Trouver le fichier vidéo
        self.video_path = self._find_video()
        
        # Chemins des fichiers de données
        self.camera_params_path = self.folder_path / "scene_camera.json"
        self.info_path = self.folder_path / "info.json"
        self.gaze_path = self.folder_path / "gaze.csv"
        self.fixations_path = self.folder_path / "fixations.csv"
        self.world_timestamps_path = self.folder_path / "world_timestamps.csv"
        self.imu_path = self.folder_path / "imu.csv"
        
        # Pré-charger et aligner toutes les données
        self._load_all_data()
        
    def _find_video(self) -> Path:
        """Trouve le fichier vidéo dans le dossier."""
        video_files = list(self.folder_path.glob("*.mp4"))
        if not video_files:
            raise RuntimeError(f"Aucun fichier .mp4 trouvé dans {self.folder_path}")
        return video_files[0]
    
    def _load_all_data(self):
        """Charge et pré-traite toutes les données."""
        # Charger les timestamps des frames vidéo
        world_ts_df = pd.read_csv(self.world_timestamps_path)
        self.frame_timestamps = world_ts_df["timestamp [ns]"].to_numpy(dtype=np.int64)
        self.n_frames = len(self.frame_timestamps)
        
        # Charger et aligner les données de regard
        self._load_gaze_data()
        
        # Charger et aligner les données IMU
        self._load_imu_data()
        
        # Préparer l'undistortion
        self._prepare_undistortion()
        
        # Undistort les points de regard (vectorisé pour toutes les frames)
        self._undistort_gaze_points()
        
        # Charger et aligner les données de fixations
        self._load_fixations_data()
        
        # Undistort les points de fixation (vectorisé pour toutes les frames)
        self._undistort_fixation_points()
    
    def _load_gaze_data(self):
        """Charge et interpole les données de regard."""
        gaze_df = pd.read_csv(self.gaze_path)
        gaze_ts = gaze_df["timestamp [ns]"].to_numpy(dtype=np.int64)
        gaze_x = gaze_df["gaze x [px]"].to_numpy(dtype=np.float32)
        gaze_y = gaze_df["gaze y [px]"].to_numpy(dtype=np.float32)
        
        # Appliquer l'offset de calibration
        offset = load_gaze_offset(self.info_path)
        gaze_x -= offset[0]
        gaze_y -= offset[1]
        
        # Interpoler aux timestamps des frames (vectorisé)
        self.gaze_x = np.interp(self.frame_timestamps, gaze_ts, gaze_x).astype(np.float32)
        self.gaze_y = np.interp(self.frame_timestamps, gaze_ts, gaze_y).astype(np.float32)
        
  
    def _load_fixations_data(self):
        fixations_df = pd.read_csv(self.fixations_path)
        fixations_ts_start = fixations_df["start timestamp [ns]"].to_numpy(dtype=np.int64)
        fixations_ts_end = fixations_df["end timestamp [ns]"].to_numpy(dtype=np.int64)
        moyenne_fixations_ts = (fixations_ts_start + fixations_ts_end) / 2
        print(f"\nmoyenne_ts_fixations : {moyenne_fixations_ts}\n")
        fixations_x = fixations_df["fixation x [px]"].to_numpy(dtype=np.float32)
        fixations_y = fixations_df["fixation y [px]"].to_numpy(dtype=np.float32)
        
        # Pas d'offset de calibration?
        
        # Interpoler aux timestamps des frames (vectorisé)
        self.fixations_x = np.interp(self.frame_timestamps, moyenne_fixations_ts, fixations_x).astype(np.float32)
        self.fixations_y = np.interp(self.frame_timestamps, moyenne_fixations_ts, fixations_y).astype(np.float32)

    
    def _load_imu_data(self):
        """Charge et interpole les données IMU."""
        imu_df = pd.read_csv(self.imu_path)
        
        imu_ts = imu_df["timestamp [ns]"].to_numpy(dtype=np.int64)
        
        # Données gyroscopiques
        gyro_data = imu_df[["gyro x [deg/s]", "gyro y [deg/s]", "gyro z [deg/s]"]].to_numpy(dtype=np.float32)
        
        # Données accéléromètre
        accel_data = imu_df[["acceleration x [g]", "acceleration y [g]", "acceleration z [g]"]].to_numpy(dtype=np.float32)
        
        # Interpoler aux timestamps des frames
        self.gyro_data = np.zeros((self.n_frames, 3), dtype=np.float32)
        self.accel_data = np.zeros((self.n_frames, 3), dtype=np.float32)
        
        for i in range(3):
            self.gyro_data[:, i] = np.interp(self.frame_timestamps, imu_ts, gyro_data[:, i])
            self.accel_data[:, i] = np.interp(self.frame_timestamps, imu_ts, accel_data[:, i])
    
    def _prepare_undistortion(self):
        """Prépare les maps pour l'undistortion (calcul une seule fois)."""
        K, D = load_camera_params(self.camera_params_path)
        
        # Obtenir les dimensions de la vidéo
        cap = cv2.VideoCapture(str(self.video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Sauvegarder les paramètres de la caméra
        self.K = K
        self.D = D
        
        # Calculer les maps d'undistortion (une seule fois)
        self.new_K, roi = cv2.getOptimalNewCameraMatrix(
            K, D, (width, height), alpha=0, newImgSize=(width, height)
        )
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            K, D, None, self.new_K, (width, height), cv2.CV_32FC1
        )
    
    def _undistort_gaze_points(self):
        """
        Undistort tous les points de regard en une seule fois (vectorisé).
        Les coordonnées du regard sont dans l'image distordue, on doit les transformer
        pour correspondre à l'image undistorted.
        """
        # Préparer les points (n_frames, 1, 2) pour cv2.undistortPoints
        gaze_points_distorted = np.stack([self.gaze_x, self.gaze_y], axis=1)
        gaze_points_distorted = gaze_points_distorted.reshape(-1, 1, 2).astype(np.float32)
        
        # Undistort tous les points en une seule opération
        gaze_points_undistorted = cv2.undistortPoints(
            gaze_points_distorted,
            self.K,
            self.D,
            P=self.new_K
        )
        
        # Récupérer les coordonnées undistorted
        gaze_points_undistorted = gaze_points_undistorted.reshape(-1, 2)
        self.gaze_x_undistorted = gaze_points_undistorted[:, 0].astype(np.float32)
        self.gaze_y_undistorted = gaze_points_undistorted[:, 1].astype(np.float32)
        
        
    def _undistort_fixation_points(self):
        """
        Undistort tous les points de regard en une seule fois (vectorisé).
        Les coordonnées du regard sont dans l'image distordue, on doit les transformer
        pour correspondre à l'image undistorted.
        """
        # Préparer les points (n_frames, 1, 2) pour cv2.undistortPoints
        fixation_points_distorted = np.stack([self.fixations_x, self.fixations_y], axis=1)
        fixation_points_distorted = fixation_points_distorted.reshape(-1, 1, 2).astype(np.float32)
        
        # Undistort tous les points en une seule opération
        fixation_points_undistorted = cv2.undistortPoints(
            fixation_points_distorted,
            self.K,
            self.D,
            P=self.new_K
        )
        
        # Récupérer les coordonnées undistorted
        fixation_points_undistorted = fixation_points_undistorted.reshape(-1, 2)
        self.fixation_x_undistorted = fixation_points_undistorted[:, 0].astype(np.float32)
        self.fixation_y_undistorted = fixation_points_undistorted[:, 1].astype(np.float32)
    
    
    def __len__(self) -> int:
        """Retourne le nombre de frames."""
        return self.n_frames
    
    def __iter__(self) -> Iterator[FrameData]:
        """Itère sur toutes les frames de manière optimisée."""
        # Mode économe en mémoire: streaming vidéo
        cap = cv2.VideoCapture(str(self.video_path))
        frame_id = 0
        
        while frame_id < self.n_frames:
            ret, image = cap.read()
            if not ret:
                break
            
            # Appliquer l'undistortion à l'image
            image = cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)
            
            yield FrameData(
                frame_id=frame_id,
                image=image,
                gaze_point=(float(self.gaze_x_undistorted[frame_id]), 
                           float(self.gaze_y_undistorted[frame_id])),
                fixation_point=(float(self.fixation_x_undistorted[frame_id]), 
                           float(self.fixation_y_undistorted[frame_id])),
                gyro=tuple(self.gyro_data[frame_id]),
                accel=tuple(self.accel_data[frame_id]),
                timestamp=self.frame_timestamps[frame_id]
            )
            
            frame_id += 1
        
        cap.release()

    def get_video_path(self) -> str:
        """Retourne le chemin du fichier vidéo."""
        return str(self.video_path)


# Exemple d'utilisation
if __name__ == "__main__":
    loader = VideoLoader(r"data\Cyclistes\2025-11-20_15-30-11-a3a383b4")
    
    for frame_data in loader:
        print(f"Frame {frame_data.frame_id}: gaze={frame_data.gaze_point}")
        
        # Visualiser le point de regard sur l'image
        img_display = frame_data.image.copy()
        gaze_x, gaze_y = frame_data.gaze_point
        cv2.circle(img_display, (int(gaze_x), int(gaze_y)), 10, (0, 255, 0), 2)
        cv2.circle(img_display, (int(gaze_x), int(gaze_y)), 3, (0, 0, 255), -1)
        
        cv2.imshow("Frame", img_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
    cv2.destroyAllWindows()