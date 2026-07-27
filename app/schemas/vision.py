from pydantic import BaseModel


class VisionExtraction(BaseModel):
    is_legible: bool
    procedure_name: str | None = None  # bare literal term as written, e.g. "IGRA", "zapatilla talla 42" — verified against this
    price_question: str | None = None  # customer-facing formatted question, e.g. "¿Cuánto cuesta un examen de IGRA?"
    # Set ONLY when the document lists 2+ DISTINCT samples/items under one order
    # (e.g. a biopsy request with "Muestra #1", "#2", "#3" — real /qa 2026-07-24
    # finding: a 6-sample request was silently collapsed to 1 question). Each
    # entry is the bare literal item name, same convention as procedure_name —
    # each gets verified individually before being included in the final answer.
    # None/empty for the single-item case (the overwhelming majority of photos).
    samples: list[str] | None = None


class VisionVerification(BaseModel):
    text_visible: bool
