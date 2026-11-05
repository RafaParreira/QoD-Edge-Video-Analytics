
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import requests


# Configurações
SERVER_URL = os.getenv(
    "SERVER_URL",
    "http://127.0.0.1:8000/detect"
)
VIDEO_PATH = Path("videos/teste.mp4")
RESULTS_DIR = Path("results/raw")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Verifica se o vídeo existe
if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Vídeo não encontrado: {VIDEO_PATH}"
    )

# Abre o vídeo
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Não foi possível abrir o vídeo.")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Vídeo: {VIDEO_PATH}")
print(f"FPS original: {fps:.2f}")
print(f"Total de frames: {total_frames}")

# Arquivo CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = RESULTS_DIR / f"baseline_{timestamp}.csv"

session = requests.Session()

frame_id = 0
successful_requests = 0

with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "frame_id",
        "request_time_ms",
        "inference_time_ms",
        "num_detections",
        "status"
    ])

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Codifica o frame como JPEG
        success, encoded_frame = cv2.imencode(
            ".jpg",
            frame
        )

        if not success:
            print(f"Erro ao codificar frame {frame_id}")
            frame_id += 1
            continue

        image_bytes = encoded_frame.tobytes()

        # Mede o tempo total da requisição HTTP
        start_time = time.perf_counter()

        try:
            response = session.post(
                SERVER_URL,
                files={
                    "file": (
                        "frame.jpg",
                        image_bytes,
                        "image/jpeg"
                    )
                },
                timeout=30
            )

            request_time_ms = (
                time.perf_counter() - start_time
            ) * 1000

            response.raise_for_status()

            result = response.json()

            inference_time_ms = result["inference_time_ms"]
            num_detections = result["num_detections"]

            status = "success"
            successful_requests += 1

        except requests.RequestException as error:

            request_time_ms = (
                time.perf_counter() - start_time
            ) * 1000

            inference_time_ms = ""
            num_detections = ""
            status = f"error: {error}"

        writer.writerow([
            frame_id,
            round(request_time_ms, 3),
            inference_time_ms,
            num_detections,
            status
        ])

        if frame_id % 10 == 0:
            print(
                f"Frame {frame_id}/{total_frames} | "
                f"Requisições bem-sucedidas: {successful_requests}"
            )

        frame_id += 1

cap.release()
session.close()

print("\nExperimento finalizado!")
print(f"Frames processados: {frame_id}")
print(f"Requisições bem-sucedidas: {successful_requests}")
print(f"Resultados salvos em: {csv_path}")
