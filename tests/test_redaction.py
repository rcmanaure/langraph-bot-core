"""Unit tests for app.services.redaction.redact_document_numbers."""
from app.services.redaction import redact_document_numbers


def test_ordinary_message_is_unaffected():
    msg = "¿Cuál es el precio del examen de sangre?"
    assert redact_document_numbers(msg) == msg


def test_short_item_code_is_unaffected():
    msg = "El código es BX045"
    assert redact_document_numbers(msg) == msg


def test_cups_procedure_code_is_unaffected():
    """A bare digit run with no document-type keyword nearby must survive —
    CUPS procedure codes are exactly this shape (found in /code-review)."""
    msg = "cuánto cuesta el examen con código CUPS 890201"
    assert redact_document_numbers(msg) == msg


def test_phone_number_is_unaffected():
    msg = "mi celular es 3001234567"
    assert redact_document_numbers(msg) == msg


def test_unprefixed_price_is_unaffected():
    """Found in /code-review: a price typed without "$" must not be mistaken
    for a document number just because it's 6+ digits."""
    msg = "cuesta 150000 pesos"
    assert redact_document_numbers(msg) == msg


def test_cedula_with_keyword_is_redacted():
    msg = "mi cédula es 12345678"
    assert redact_document_numbers(msg) == "mi cédula [documento]"


def test_dotted_cedula_with_keyword_is_redacted():
    msg = "mi documento es 12.345.678"
    assert redact_document_numbers(msg) == "mi documento [documento]"


def test_dni_with_keyword_is_redacted():
    msg = "mi DNI: 87654321"
    assert redact_document_numbers(msg) == "mi DNI [documento]"


def test_dotted_cc_abbreviation_with_keyword_is_redacted():
    """Found in /code-review: "C.C." ends in a period, and a \\b right after
    it can't match between two non-word characters (the period, then the
    space that follows) -- the match silently failed to fire at all."""
    msg = "paciente C.C. 87654321, agenda para el jueves"
    assert redact_document_numbers(msg) == "paciente C.C. [documento], agenda para el jueves"


def test_passport_with_keyword_is_redacted():
    msg = "mi pasaporte es AB1234567"
    assert redact_document_numbers(msg) == "mi pasaporte [documento]"


def test_price_prefixed_with_dollar_sign_is_not_redacted():
    """This app's own prompts always prefix a price with "$" (see
    generate.py's format hint) — a figure written that way must never be
    mistaken for a document number, even right after a keyword-free amount."""
    msg = "el examen cuesta $150000"
    assert redact_document_numbers(msg) == msg


def test_multiple_document_numbers_are_all_redacted():
    msg = "cédula 12345678 y pasaporte CD9876543"
    assert redact_document_numbers(msg) == "cédula [documento] y pasaporte [documento]"


def test_short_digit_run_is_unaffected():
    """Fewer than 6 digits doesn't match any document shape in use."""
    msg = "tengo 5 años esperando"
    assert redact_document_numbers(msg) == msg
