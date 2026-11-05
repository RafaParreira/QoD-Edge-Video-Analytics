
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from ultralytics import YOLO


app = FastAPI(
    title="QoD Edge Video Analytics",
    description="Servidor Edge para análise de vídeos com YOLO11n",
    version="0.1.0",
)

# Carrega o modelo uma única vez na inicialização.
model = YOLO("yolo11n.pt")


@app.get("/health")
def health():
    return {
        "status": "online",
        "model": "YOLO11n",
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="O arquivo enviado deve ser uma imagem.",
        )

    image_bytes = await file.read()

    frame_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Não foi possível decodificar a imagem.",
        )

    start_time = time.perf_counter()

    results = model.predict(
        source=frame,
        verbose=False,
    )

    inference_time_ms = (
        time.perf_counter() - start_time
    ) * 1000

    detections = []

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])

            detections.append({
                "class_id": class_id,
                "class_name": model.names[class_id],
                "confidence": float(box.conf[0]),
                "bbox": [
                    float(value)
                    for value in box.xyxy[0].tolist()
                ],
            })

    return {
        "status": "success",
        "inference_time_ms": round(inference_time_ms, 3),
        "num_detections": len(detections),
        "detections": detections,
    }
