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


def _remove_isolated_circles(circles, factor: float = 2.5):
    """Descarta círculos "sueltos" que no tienen ningún vecino cercano (p. ej.
    una palabra que el alumno circuló a mano en el nombre): en la cuadrícula
    real, cada círculo de respuesta siempre tiene otros círculos muy cerca
    (misma fila/columna), mientras que una marca aislada no."""
    if len(circles) < 2:
        return circles

    xs = np.array([c[0] for c in circles])
    ys = np.array([c[1] for c in circles])
    nearest = []
    for i in range(len(circles)):
        dists = np.hypot(xs - xs[i], ys - ys[i])
        dists[i] = np.inf
        nearest.append(dists.min())
    nearest = np.array(nearest)
    median_nearest = float(np.median(nearest))

    return [c for c, d in zip(circles, nearest) if d <= median_nearest * factor]


def _detect_circles(gray: np.ndarray, y_min: int):
    raw = _detect_raw_circles(gray, y_min)
    if not raw:
        return []
    radii = np.array([r for _, _, r in raw])
    median_r = float(np.median(radii))
    filtered = [(x, y, r) for (x, y, r) in raw if abs(r - median_r) <= 0.25 * median_r]
    merged = _merge_close_circles(filtered, median_r)
    return _remove_isolated_circles(merged)


def _kmeans_1d(values, k) -> np.ndarray:
    data = np.array(values, dtype=np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    order = np.argsort(centers.flatten())
    rank = {old: new for new, old in enumerate(order)}
    return np.array([rank[label] for label in labels])


MIN_SECTION_FRACTION = 0.80  # mínimo de círculos esperados por sección para intentar procesarla
MIN_GOOD_ROWS_FRACTION = 0.5  # mínimo de filas "completas" (4 círculos) para calibrar columnas


def build_bubble_grid(img: np.ndarray) -> dict:
    """Detecta y agrupa los círculos de la hoja. Devuelve
    {seccion: {pregunta: {letra: (cx, cy, r)}}} (centro y radio del círculo,
    no un rectángulo) para poder medir solo el interior del círculo y evitar
    contar el propio borde impreso.

    Es tolerante a que falten o sobren uno o dos círculos sueltos (ruido de
    la foto, marcas del alumno fuera de lugar, etc.): en vez de exigir un
    conteo exacto, calibra la posición de las 4 columnas (A/B/C/D) usando las
    filas que sí se detectaron completas, y para las filas incompletas asigna
    cada círculo encontrado a la columna más cercana. Una opción sin círculo
    detectado en su fila queda simplemente "sin dato" (se trata como no
    marcada al leer la respuesta) en vez de invalidar toda la foto.
    Solo lanza OmrError si una sección tiene demasiados pocos círculos como
    para ser confiable."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]

    circles = _detect_circles(gray, y_min=int(h * 0.12))

    xs = np.array([c[0] for c in circles])
    ys = np.array([c[1] for c in circles])
    radii = np.array([c[2] for c in circles])

    section_labels = _kmeans_1d(xs, N_SECTIONS)
    grid: dict = {name: {} for name in SECTION_NAMES}

    expected_section_total = N_QUESTIONS * len(LETTERS)

    for section_idx, sec_name in enumerate(SECTION_NAMES):
        sec_mask = np.where(section_labels == section_idx)[0]
        if len(sec_mask) < expected_section_total * MIN_SECTION_FRACTION:
            raise OmrError(
                f"Sección {sec_name}: se detectaron {len(sec_mask)} círculos, "
                f"se esperaban {expected_section_total}. Verifica que la foto "
                "muestre la hoja completa, bien iluminada y sin recortes."
            )

        row_labels_local = _kmeans_1d(ys[sec_mask], N_QUESTIONS)
        rows_idx = []
        for row in range(N_QUESTIONS):
            row_idx = sec_mask[np.where(row_labels_local == row)[0]]
            row_idx = row_idx[np.argsort(xs[row_idx])]
            rows_idx.append(row_idx)

        good_rows = [r for r in rows_idx if len(r) == len(LETTERS)]
        if len(good_rows) < N_QUESTIONS * MIN_GOOD_ROWS_FRACTION:
            raise OmrError(
                f"Sección {sec_name}: solo {len(good_rows)} de {N_QUESTIONS} "
                "filas se detectaron completas, no es suficiente para "
                "calibrar las columnas. Verifica la foto."
            )

        col_centers = [
            float(np.median([xs[r[k]] for r in good_rows])) for k in range(len(LETTERS))
        ]

        for row, row_idx in enumerate(rows_idx):
            q_num = str(row + 1)
            options: dict = {}
            used_cols: set = set()
            for gi in row_idx:
                order = sorted(range(len(LETTERS)), key=lambda k: abs(xs[gi] - col_centers[k]))
                for k in order:
                    if k not in used_cols:
                        used_cols.add(k)
                        options[LETTERS[k]] = (float(xs[gi]), float(ys[gi]), float(radii[gi]))
                        break
            grid[sec_name][q_num] = options

    return grid


def _fill_intensity(gray: np.ndarray, circle, inner_ratio: float = 0.7) -> float:
    """Intensidad promedio de gris dentro del círculo (excluyendo el propio
    anillo impreso, midiendo solo el interior). Menor valor = más tinta."""
    cx, cy, r = circle
    inner_r = max(1, int(r * inner_ratio))
    x0, y0 = int(cx - inner_r), int(cy - inner_r)
    x1, y1 = int(cx + inner_r), int(cy + inner_r)

    h, w = gray.shape
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x1, w), min(y1, h)
    patch = gray[y0c:y1c, x0c:x1c]
    if patch.size == 0:
        return 255.0

    mask = np.zeros(patch.shape, dtype=np.uint8)
    center_local = (int(cx) - x0c, int(cy) - y0c)
    cv2.circle(mask, center_local, inner_r, 255, thickness=-1)

    mean_val = cv2.mean(patch, mask=mask)[0]
    return float(mean_val)


RANGE_MIN = 12.0  # contraste mínimo (más oscuro vs más claro) para asumir que hay tinta
REL_GAP = 0.25  # fracción del contraste total que separa "marcado" de "en blanco"


def _classify_question(scores: dict) -> str:
    """Decide la respuesta de una pregunta a partir de las 4 intensidades
    (menor intensidad = más tinta).

    En vez de un umbral fijo en píxeles (que no se adapta bien a fotos con
    distinta iluminación/contraste), se usa una fracción relativa del
    contraste total de esa pregunta (más oscura vs más clara):
      1. Si el contraste total es muy bajo, no hay tinta real -> "" (blanco).
      2. Si el salto entre la más oscura y la 2da más oscura ya es una
         fracción grande del contraste total -> una sola marcada.
      3. Si no, pero el salto entre la 2da y la 3ra sí lo es -> dos marcadas
         ("MULTIPLE:X,Y", inválida, requiere revisión del maestro).
      4. Si ningún salto destaca claramente (caso ambiguo/ruidoso), se
         devuelve la más oscura como mejor estimación, igual que antes.
    """
    items = sorted(scores.items(), key=lambda kv: kv[1])
    values = [v for _, v in items]
    total_range = values[-1] - values[0]

    if total_range < RANGE_MIN:
        return ""

    gap01 = (values[1] - values[0]) / total_range
    if gap01 >= REL_GAP:
        return items[0][0]

    gap12 = (values[2] - values[1]) / total_range
    if gap12 >= REL_GAP:
        return "MULTIPLE:" + ",".join(sorted([items[0][0], items[1][0]]))

    return items[0][0]


def read_answers(img: np.ndarray) -> dict:
    """Punto de entrada principal: detecta la cuadrícula y clasifica la
    respuesta de cada pregunta de cada sección (ver _classify_question)."""
    grid = build_bubble_grid(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    respuestas: dict = {}
    for section, questions in grid.items():
        respuestas[section] = {}
        for q_num, options in questions.items():
            scores = {}
            for letter in LETTERS:
                if letter in options:
                    scores[letter] = _fill_intensity(gray, options[letter])
                else:
                    scores[letter] = 255.0  # no se detectó el círculo; se asume sin tinta
            respuestas[section][q_num] = _classify_question(scores)

    return respuestas
