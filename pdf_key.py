"""
Extractor determinístico de la clave de respuestas correctas desde el PDF
del examen (el "solucionario" con las respuestas resaltadas en amarillo).

No usa IA/OCR/modelos de lenguaje: el resaltado amarillo en este PDF es un
rectángulo vectorial de relleno puro (1.0, 1.0, 0.0) dibujado sobre el texto,
no una anotación. El proceso es:
  1. Ubicar todos los rectángulos amarillos de cada página y fusionar los que
     pertenecen a la misma opción (una respuesta larga puede resaltarse en
     2+ líneas/rectángulos consecutivos).
  2. Extraer el texto contenido en cada rectángulo fusionado y tomar la letra
     de opción (a/b/c/d) que contiene.
  3. Ubicar los números de pregunta ("1.", "2.", ...) como palabras sueltas
     en la página.
  4. Recorrer preguntas y resaltados de TODO el documento en orden real de
     lectura (columna izquierda de arriba a abajo, luego columna derecha,
     luego la siguiente página) llevando un contador de "pregunta actual",
     de modo que una respuesta resaltada que queda partida entre columnas o
     entre páginas se asigne correctamente a su pregunta.

Asume el formato fijo de este examen (mismo generador/plantilla): los
encabezados de materia son siempre "Lenguajes", "Saberes y P...",
"Ética, N...", "De lo Hum...".
"""

import re

import fitz

SECTION_MARKERS = [
    ("Lenguajes", "Lenguajes"),
    ("Saberes y P", "Saberes_y_PC"),
    ("Ética, N", "Etica_N_y_S"),
    ("De lo Hum", "Humanidades"),
]

QNUM_RE = re.compile(r"^(\d{1,2})\.$")
OPT_RE = re.compile(r"([a-dA-D])\)")


class PdfKeyError(Exception):
    """Error determinístico y explicable al extraer la clave del PDF."""


def _detect_section(text: str):
    for marker, name in SECTION_MARKERS:
        if marker in text:
            return name
    return None


def _column_of(x0: float, page_width: float) -> int:
    return 0 if x0 < page_width / 2 else 1


def _merge_yellow_rects(rects, y_gap: float = 22):
    rects = sorted(rects, key=lambda r: r.x0)
    buckets = []
    for r in rects:
        placed = False
        for b in buckets:
            if abs(b[0].x0 - r.x0) < 40:
                b.append(r)
                placed = True
                break
        if not placed:
            buckets.append([r])

    groups = []
    for b in buckets:
        b = sorted(b, key=lambda r: r.y0)
        current = [b[0]]
        for r in b[1:]:
            if r.y0 - current[-1].y1 < y_gap:
                current.append(r)
            else:
                groups.append(current)
                current = [r]
        groups.append(current)
    return groups


def extract_key_from_bytes(pdf_bytes: bytes) -> dict:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    result: dict = {}
    current_section = None
    current_question = None

    for page in doc:
        page_text = page.get_text("text")
        sec = _detect_section(page_text)
        if sec:
            current_section = sec
            current_question = None
            result.setdefault(current_section, {})

        if current_section is None:
            continue

        page_width = page.rect.width
        events = []  # (column, y0, kind, data)

        for w in page.get_text("words"):
            x0, y0, word = w[0], w[1], w[4]
            m = QNUM_RE.match(word)
            if m:
                events.append((_column_of(x0, page_width), y0, "q", int(m.group(1))))

        drawings = page.get_drawings()
        yellow = [d["rect"] for d in drawings if d.get("fill") == (1.0, 1.0, 0.0)]
        for g in _merge_yellow_rects(yellow):
            combined = fitz.Rect(g[0])
            for r in g[1:]:
                combined |= r
            text = page.get_text("text", clip=combined).strip()
            m = OPT_RE.search(text)
            if not m:
                continue
            events.append((_column_of(combined.x0, page_width), combined.y0, "opt", m.group(1).upper()))

        events.sort(key=lambda e: (e[0], e[1]))

        for _column, _y0, kind, data in events:
            if kind == "q":
                current_question = data
            elif current_question is not None:
                result[current_section][str(current_question)] = data

    if not result:
        raise PdfKeyError(
            "No se detectó ninguna materia conocida en el PDF. Verifica que "
            "sea el examen con el formato esperado."
        )

    # La hoja de respuestas física tiene capacidad fija para 25 preguntas por
    # materia; el examen puede usar menos (filas de más quedan en blanco),
    # pero nunca más de 25 sin cambiar también la hoja impresa.
    for materia, preguntas in result.items():
        if len(preguntas) == 0:
            raise PdfKeyError(
                f"Materia '{materia}': no se detectó ninguna respuesta "
                "resaltada. Verifica que las respuestas correctas estén "
                "resaltadas en amarillo."
            )
        if len(preguntas) > 25:
            raise PdfKeyError(
                f"Materia '{materia}': se detectaron {len(preguntas)} "
                "respuestas resaltadas, y la hoja de respuestas solo tiene "
                "25 preguntas por materia. Revisa el PDF."
            )

    return result
