"""Parse the lab's catalog Markdown into the record shape
indexer.py's chunk-building logic expects (see _records_to_chunks there).

Why this exists: the human-authored .md is git-diffable and easy for a
non-technical admin to edit, but generic text/table splitters detach
price-table headers from rows and corrupt price/code associations. This
module is the one place that understands the .md's structure (category
price tables + prose sections), so both the admin/UI upload path
(indexer.py) and the manual CLI (scripts/md_catalog_to_jsonl.py) share
the exact same parsing -- no duplicated logic to drift out of sync.
"""

import re
import unicodedata

_STOPWORDS = {
    "de", "la", "el", "los", "las", "y", "o", "u", "e", "del", "con", "en",
    "por", "sin", "para", "no", "a", "un", "una", "c/u", "cada", "total",
    "parcial", "simple", "radical", "biopsia", "estudio", "estudios",
}

_TABLE_ROW_RE = re.compile(r"^\|\s*([A-Z]{2,4}\d{2,4})\s*\|\s*(.+?)\s*\|\s*\$?([\d,]+\.\d{2})\s*\|")

_CATEGORY_TYPE_OVERRIDE = {
    "ESTUDIOS CITOLÓGICOS ESPECÍFICOS": "cytology",
    "ESTUDIOS ESPECIALES": "protocol",
}


def _slug(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _keywords_from(*texts: str, limit: int = 6) -> list[str]:
    words: list[str] = []
    for text in texts:
        for word in re.findall(r"[a-záéíóúñ]+", _slug(text)):
            if len(word) > 3 and word not in _STOPWORDS and word not in words:
                words.append(word)
    return words[:limit]


def _item_type(category: str, name: str) -> str:
    if "protocolo" in name.lower():
        return "protocol"
    return _CATEGORY_TYPE_OVERRIDE.get(category, "biopsy")


def _parse_category_table(category: str, body: str) -> list[dict]:
    items = []
    for line in body.splitlines():
        m = _TABLE_ROW_RE.match(line.strip())
        if not m:
            continue
        code, name, price = m.group(1), m.group(2).strip(), float(m.group(3).replace(",", ""))
        item_type = _item_type(category, name)
        items.append({
            "id": f"{item_type}-{code.lower()}",
            "name": name,
            "price": price,
            "type": item_type,
            "category": category,
            "keywords": _keywords_from(category, name),
        })
    return items


def _first_paragraph(body: str) -> str:
    for para in body.strip().split("\n\n"):
        para = para.strip()
        if para and not para.startswith(("#", ">", "|", "-", "---")):
            return re.sub(r"\s+", " ", para)
    return ""


def _parse_policies(body: str) -> list[dict]:
    items = []
    for i, m in enumerate(
        re.finditer(r"^\d+\.\s+\*\*(.+?):\*\*\s+(.+?)(?=\n\d+\.\s+\*\*|\Z)", body, re.MULTILINE | re.DOTALL), 1
    ):
        title, text = m.group(1).strip(), re.sub(r"\s+", " ", m.group(2)).strip()
        items.append({
            "id": f"policy-{i:03d}",
            "name": f"Política de {title}",
            "price": None,
            "type": "policy",
            "category": "políticas",
            "keywords": _keywords_from(title),
            "text": text,
        })
    return items


def _parse_faqs(body: str) -> list[dict]:
    items = []
    for i, m in enumerate(
        re.finditer(r"^###\s+(.+?)\n(.+?)(?=\n###\s|\n---|\Z)", body, re.MULTILINE | re.DOTALL), 1
    ):
        question, answer = m.group(1).strip(), re.sub(r"\s+", " ", m.group(2)).strip()
        answer = re.sub(r"\*\*(.+?)\*\*", r"\1", answer)
        if not answer:
            continue
        items.append({
            "id": f"faq-{i:03d}",
            "name": question,
            "price": None,
            "type": "faq",
            "category": "faq",
            "keywords": _keywords_from(question),
            "text": answer,
        })
    return items


def _parse_org_metadata(contact_body: str, org_name: str, last_updated: str) -> dict:
    fields = dict(re.findall(r"\*\*(.+?):\*\*\s*(.+)", contact_body))
    return {
        "id": "org_metadata",
        "type": "org_metadata",
        "organization": org_name,
        "location": fields.get("Dirección", ""),
        "last_updated": last_updated,
        "phone": fields.get("Teléfono", ""),
        "email": fields.get("Email", ""),
        "whatsapp": fields.get("WhatsApp", ""),
        "social_media": fields.get("Redes sociales", ""),
        "address": fields.get("Dirección", ""),
    }


def convert(md_text: str) -> list[dict]:
    """Parse a catalog Markdown document into JSONL-shaped records.

    Returns an empty list if the document doesn't look like this catalog
    format (no price tables, no known section headers) -- callers should
    fall back to generic text chunking in that case, not treat this as
    an error. A tenant's free-text info doc is a valid, different use case.
    """
    sections = re.split(r"\n##\s+", md_text)
    org_name = sections[0].splitlines()[0].lstrip("# ").strip()
    body_by_header = {}
    for chunk in sections[1:]:
        header, _, body = chunk.partition("\n")
        body_by_header[header.strip()] = body

    records: list[dict] = []

    if "¿Quiénes somos?" in body_by_header:
        text = _first_paragraph(body_by_header["¿Quiénes somos?"])
        records.append({
            "id": "service-001", "name": "¿Quiénes somos?", "price": None, "type": "service",
            "category": "general", "keywords": _keywords_from("laboratorio", "histopatología", "citología"),
            "text": text,
        })

    if "Servicios o Productos" in body_by_header:
        text = _first_paragraph(body_by_header["Servicios o Productos"])
        records.append({
            "id": "service-002", "name": "Servicios y Productos", "price": None, "type": "service",
            "category": "general", "keywords": _keywords_from("biopsias", "citología", "especialidades"),
            "text": text,
        })

    # Category price tables: any "## " section whose body contains a price-table header.
    for header, body in body_by_header.items():
        if "| Código |" in body or "|--------|" in body:
            records.extend(_parse_category_table(header, body))

    if "Horarios de atención" in body_by_header:
        rows = re.findall(r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$", body_by_header["Horarios de atención"], re.MULTILINE)
        rows = [(d, h) for d, h in rows if d not in ("Día", "---")]
        text = ". ".join(f"{d}: {h}" for d, h in rows) + "."
        records.append({
            "id": "service-003", "name": "Horarios de Atención", "price": None, "type": "service",
            "category": "general", "keywords": _keywords_from("horario", "atencion"),
            "text": text,
        })

    last_updated_m = re.search(r"actualizada?:\s*([A-Za-zñÑ]+\s+\d{4})", md_text)
    last_updated = last_updated_m.group(1) if last_updated_m else ""

    if "Ubicación y Contacto" in body_by_header:
        contact_body = body_by_header["Ubicación y Contacto"]
        records.append(_parse_org_metadata(contact_body, org_name, last_updated))
        fields = dict(re.findall(r"\*\*(.+?):\*\*\s*(.+)", contact_body))
        text = " ".join(f"{k}: {v}." for k, v in fields.items())
        records.append({
            "id": "service-004", "name": "Ubicación y Contacto", "price": None, "type": "service",
            "category": "general", "keywords": _keywords_from("direccion", "telefono", "contacto"),
            "text": text,
        })

    if "Políticas y Condiciones" in body_by_header:
        records.extend(_parse_policies(body_by_header["Políticas y Condiciones"]))

    if "Preguntas Frecuentes" in body_by_header:
        records.extend(_parse_faqs(body_by_header["Preguntas Frecuentes"]))

    if "Mensaje de bienvenida" in body_by_header:
        lines = [
            l.lstrip(">").strip()
            for l in body_by_header["Mensaje de bienvenida"].splitlines()
            if l.strip().startswith(">")
        ]
        text = " ".join(l for l in lines if l)
        records.append({
            "id": "service-005", "name": "Mensaje de Bienvenida", "price": None, "type": "service",
            "category": "general", "keywords": _keywords_from("bienvenida", "saludo"),
            "text": text,
        })

    disclaimer_m = re.search(r">\s*\*\*Nota:\*\*\s*(Esta lista no incluye.+?)\n", md_text)
    if disclaimer_m:
        records.append({
            "id": "disclaimer-001", "name": "Nota sobre Alcance de Servicios", "price": None, "type": "policy",
            "category": "políticas", "keywords": _keywords_from("alcance", "nota"),
            "text": re.sub(r"\s+", " ", disclaimer_m.group(1)).strip(),
        })

    return records
