"""
Motor de reconocimiento óptico de marcas (OMR) determinístico.

No usa IA, LLMs ni OCR con visión. Toda la lógica es procesamiento clásico de
imagen con OpenCV:
  1. Detectar los círculos impresos (contornos + filtro de circularidad).
  2. Agruparlos jerárquicamente: sección (materia) -> fila (pregunta) ->
     columna (A/B/C/D), usando K-means 1D sobre X e Y. Esto es inmune a que
     la foto venga con una leve rotación/perspectiva, ya que no depende de
     coordenadas fijas.
  3. Para cada círculo, contar la proporción de píxeles oscuros (tinta)
     dentro de su región. El círculo con mayor proporción por pregunta es la
     respuesta marcada.

Parámetros de la hoja (fijos, definidos por el formato del examen):
  N_SECTIONS materias, cada una con N_QUESTIONS preguntas de 4 opciones A-D.
"""

import numpy as np
import cv2

DARK_THRESHOLD = 150

N_SECTIONS = 4
LETTERS = ["A", "B", "C", "D"]
N_QUESTIONS = 25
SECTION_NAMES = ["Lenguajes", "Saberes_y_PC", "Etica_N_y_S", "Humanidades"]

EXPECTED_TOTAL = N_SECTIONS * N_QUESTIONS * len(LETTERS)


class OmrError(Exception):
    """Error determinístico y explicable del pipeline OMR (no una excepción genérica)."""


def _detect_raw_circles(gray: np.ndarray, y_min: int) -> list[tuple[float, float, float]]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    raw = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 150 or area > 4000:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.6:
            continue
        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius < 8 or radius > 30 or y < y_min:
            continue
        raw.append((x, y, radius))

    return raw


def _merge_close_circles(circles, median_r, merge_dist_factor=0.6):
    """Cada círculo impreso genera 2 contornos (borde ext./int. del anillo)
    muy cercanos entre sí; se fusionan quedándose con el de mayor radio."""
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    merged = []
    used = [False] * len(circles)
    threshold = median_r * merge_dist_factor

    for i, (x, y, r) in enumerate(circles):
        if used[i]:
            continue
        used[i] = True
        merged.append((x, y, r))
        for j in range(i + 1, len(circles)):
            if used[j]:
                continue
            x2, y2, _ = circles[j]
            if np.hypot(x - x2, y - y2) < threshold:
                used[j] = True

    return merged


def _detect_circles(gray: np.ndarray, y_min: int):
    raw = _detect_raw_circles(gray, y_min)
    if not raw:
        return []
    radii = np.array([r for _, _, r in raw])
    median_r = float(np.median(radii))
    filtered = [(x, y, r) for (x, y, r) in raw if abs(r - median_r) <= 0.25 * median_r]
    return _merge_close_circles(filtered, median_r)


def _kmeans_1d(values, k) -> np.ndarray:
    data = np.array(values, dtype=np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    order = np.argsort(centers.flatten())
    rank = {old: new for new, old in enumerate(order)}
    return np.array([rank[label] for label in labels])


def build_bubble_grid(img: np.ndarray) -> dict:
    """Detecta y agrupa los 400 círculos de la hoja. Devuelve
    {seccion: {pregunta: {letra: (x, y, w, h)}}}. Lanza OmrError si el
    conteo no coincide con el formato esperado de la hoja."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]

    circles = _detect_circles(gray, y_min=int(h * 0.12))
    if len(circles) != EXPECTED_TOTAL:
        raise OmrError(
            f"Se detectaron {len(circles)} círculos, se esperaban {EXPECTED_TOTAL}. "
            "Verifica que la foto muestre la hoja completa, bien iluminada y sin recortes."
        )

    xs = np.array([c[0] for c in circles])
    ys = np.array([c[1] for c in circles])
    radii = np.array([c[2] for c in circles])

    section_labels = _kmeans_1d(xs, N_SECTIONS)
    grid: dict = {name: {} for name in SECTION_NAMES}

    for section_idx, sec_name in enumerate(SECTION_NAMES):
        sec_mask = np.where(section_labels == section_idx)[0]
        expected_section_total = N_QUESTIONS * len(LETTERS)
        if len(sec_mask) != expected_section_total:
            raise OmrError(
                f"Sección {sec_name}: se detectaron {len(sec_mask)} círculos, "
                f"se esperaban {expected_section_total}."
            )

        row_labels_local = _kmeans_1d(ys[sec_mask], N_QUESTIONS)

        for row in range(N_QUESTIONS):
            row_idx = sec_mask[np.where(row_labels_local == row)[0]]
            if len(row_idx) != len(LETTERS):
                raise OmrError(
                    f"Sección {sec_name}, fila {row + 1}: se detectaron "
                    f"{len(row_idx)} círculos, se esperaban {len(LETTERS)}."
                )
            row_idx = row_idx[np.argsort(xs[row_idx])]
            q_num = str(row + 1)

            options = {}
            for letter, gi in zip(LETTERS, row_idx):
                cx, cy, r = xs[gi], ys[gi], radii[gi]
                pad = r * 1.3
                x0, y0 = int(cx - pad), int(cy - pad)
                bw, bh = int(pad * 2), int(pad * 2)
                options[letter] = (x0, y0, bw, bh)

            grid[sec_name][q_num] = options

    return grid


def _darkness_score(gray: np.ndarray, roi) -> float:
    x, y, w, h = roi
    region = gray[max(y, 0) : y + h, max(x, 0) : x + w]
    if region.size == 0:
        return 0.0
    return float(np.count_nonzero(region < DARK_THRESHOLD)) / region.size


def read_answers(img: np.ndarray) -> dict:
    """Punto de entrada principal: detecta la cuadrícula y lee la respuesta
    marcada (mayor proporción de tinta) para cada pregunta de cada sección."""
    grid = build_bubble_grid(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    respuestas: dict = {}
    for section, questions in grid.items():
        respuestas[section] = {}
        for q_num, options in questions.items():
            scores = {letter: _darkness_score(gray, roi) for letter, roi in options.items()}
            respuestas[section][q_num] = max(scores, key=scores.get)

    return respuestas
