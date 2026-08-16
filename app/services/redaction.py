import re

# A bare 6-10 digit run collides with far too much ordinary content in this
# domain — CUPS procedure codes, phone numbers, a price typed without a "$"
# (found in /code-review) — to redact on shape alone. Instead the shape must
# follow an explicit document-type word, which is how a person actually
# writes one out ("mi cédula es 12345678", "DNI: 87654321").
_KEYWORD = r"c[ée]dula|c\.?\s?c\.?|dni|pasaporte|documento|identificaci[oó]n"

_ID_RE = re.compile(
    rf"(?i)(\b(?:{_KEYWORD})\b)[\s:]*(?:es|no\.?|n[uú]mero)?[\s:]*"
    r"(?:(?<!\$)\d{1,3}(?:[.\s]\d{3}){1,3}\b"           # grouped: 12.345.678
    r"|(?<!\$)\d{6,10}\b"                                 # bare: 12345678
    r"|[A-Za-z]{1,2}\d{6,9}\b)"                           # passport: AB1234567
)

_PLACEHOLDER = "[documento]"


def redact_document_numbers(text: str) -> str:
    """Masks a national identity document, cédula or passport number that
    follows an explicit document-type word in free-form user text. Applied
    at the inbound turn — see turn.py — before the text becomes part of
    persisted graph state or a conversation audit record, so using the
    system never accumulates a searchable list of patient identifiers.
    """
    return _ID_RE.sub(lambda m: f"{m.group(1)} {_PLACEHOLDER}", text)
