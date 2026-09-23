"""
Microservicio de reconocimiento óptico de marcas (OMR) determinístico.

No usa IA, LLMs ni OCR con visión. Ver omr.py para el pipeline de detección
(contornos + K-means geométrico) y lectura de tinta por círculo.
"""

import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from omr import OmrError, read_answers
from pdf_key import PdfKeyError, extract_key_from_bytes

app = FastAPI(title="OMR Service", version="0.4.0")


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")
    return img


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/procesar-hoja")
async def procesar_hoja(file: UploadFile = File(...)):
    data = await file.read()
    img = decode_image(data)

    try:
        respuestas = read_answers(img)
    except OmrError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    alumno = os.path.splitext(file.filename or "pendiente")[0]

    return JSONResponse({"alumno": alumno, "respuestas": respuestas})


@app.post("/extraer-clave")
async def extraer_clave(file: UploadFile = File(...)):
    data = await file.read()

    try:
        clave = extract_key_from_bytes(data)
    except PdfKeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return JSONResponse(clave)
