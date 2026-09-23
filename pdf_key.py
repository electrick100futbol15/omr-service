"""
Extractor determinístico de la clave de respuestas correctas desde el PDF
del examen (el "solucionario" con las respuestas resaltadas en amarillo).

No usa IA/OCR/modelos de lenguaje. El proceso es:
  1. Ubicar todos los rectángulos amarillos de cada página (relleno vectorial
     puro (1.0, 1.0, 0.0), no una anotación) y fusionar los que pertenecen a
     la misma opción (una respuesta larga puede resaltarse en 2+ líneas).
  2. Extraer el texto contenido en cada rectángulo fusionado y tomar la letra
     de opción (a/b/c/d) que contiene.
  3. Ubicar los números de pregunta ("1.", "2.", ...) como palabras sueltas
     en la página.
  4. Detectar el inicio de una materia nueva: la página donde aparece la
     pregunta "1." SIEMPRE contiene también el título de esa materia escrito
     en una fuente mucho más grande que el resto del texto (tamaño >= 30,
     frente a 12-18 del cuerpo del examen). Esto identifica materias por
     ORDEN de aparición y por su título real impreso, sin depender de que el
     nombre sea uno de un catálogo fijo — así cualquier título nuevo que la
     maestra use en un examen futuro se reconoce igual.
  5. Recorrer preguntas y resaltados de TODO el documento en orden real de
     lectura (columna izquierda de arriba a abajo, luego columna derecha,
     luego la siguiente página) llevando un contador de "pregunta actual",
     de modo que una respuesta resaltada que queda partida entre columnas o
     entre páginas se asigne correctamente a su pregunta.

Asume el formato fijo de la hoja física de respuestas: siempre hay 4
materias (secciones) con hasta 25 preguntas cada una. Los nombres de las
materias se identifican por posición (Seccion_1, Seccion_2, ...) más el
título real detectado, no por texto fijo esperado.
"""

import re

import fitz

QNUM_RE = re.compile(r"^(\d{1,2})\.$")
OPT_RE = re.compile(r"([a-dA-D])\)")
TITLE_MIN_SIZE = 30  # los títulos de materia usan una fuente mucho más grande que el cuerpo del examen


class PdfKeyError(Exception):
    """Error determinístico y explicable al extraer la clave del PDF."""


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


def _page_starts_question_one(page) -> bool:
    for w in page.get_text("words"):
        m = QNUM_RE.match(w[4])
        if m and m.group(1) == "1":
            return True
    return False


def _page_title(page) -> str:
    spans = []
    for block in page.get_text("dict")["blocks"]:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if text and span["size"] >= TITLE_MIN_SIZE:
                    spans.append((span["bbox"][1], text))
    spans.sort(key=lambda s: s[0])
    return " ".join(text for _, text in spans).strip()


def extract_key_from_bytes(pdf_bytes: bytes) -> dict:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    result: dict = {}
    titulos: dict = {}
    current_section = None
    current_question = None
    section_count = 0

    for page in doc:
        if _page_starts_question_one(page):
            section_count += 1
            current_section = f"Seccion_{section_count}"
            current_question = None
            titulo = _page_title(page)
            result[current_section] = {}
            titulos[current_section] = titulo or current_section

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
            "No se detectó ninguna materia en el PDF. Verifica que sea el "
            "examen con el formato esperado (título grande de la materia en "
            "la misma página donde inicia la pregunta 1)."
        )

    # La hoja de respuestas física tiene capacidad fija para 25 preguntas por
    # materia; el examen puede usar menos (filas de más quedan en blanco),
    # pero nunca más de 25 sin cambiar también la hoja impresa.
    for materia, preguntas in result.items():
        nombre = titulos.get(materia, materia)
        if len(preguntas) == 0:
            raise PdfKeyError(
                f"Materia '{nombre}': no se detectó ninguna respuesta "
                "resaltada. Verifica que las respuestas correctas estén "
                "resaltadas en amarillo."
            )
        if len(preguntas) > 25:
            raise PdfKeyError(
                f"Materia '{nombre}': se detectaron {len(preguntas)} "
                "respuestas resaltadas, y la hoja de respuestas solo tiene "
                "25 preguntas por materia. Revisa el PDF."
            )

    return {"respuestas_correctas": result, "titulos": titulos}
