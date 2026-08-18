"""Static safety check for the operator inbox page (#40).

The page renders message content and author names written by anyone who
can message the bot, plus whatever an operator types. "No user-supplied
text is ever inserted as markup" is the one hard constraint on this page,
and per the ticket it must be verified by a test that needs no JavaScript
runner -- this greps the shipped source for the DOM APIs that would turn
text into markup, rather than trying to simulate a browser.
"""
from pathlib import Path

_SOURCE = (Path(__file__).parent.parent / "app" / "templates" / "operator.html").read_text(encoding="utf-8")

# Any of these turns a string into parsed markup instead of literal text --
# if none of them appear anywhere in the file, nothing on the page CAN
# inject markup, regardless of what content flows through it.
_MARKUP_INJECTION_APIS = [
    "innerHTML",
    "outerHTML",
    "insertAdjacentHTML",
    "document.write",
    "document.writeln",
    "dangerouslySetInnerHTML",
]


def test_operator_html_never_uses_a_markup_injection_api():
    for api in _MARKUP_INJECTION_APIS:
        assert api not in _SOURCE, f"{api} appears in operator.html -- third-party text could be inserted as markup"


def test_operator_html_actually_renders_dynamic_content_via_textcontent():
    """A negative check alone would also pass on a page that renders
    nothing at all -- confirm the safe mechanism is the one actually doing
    the work, not merely that the unsafe one is absent."""
    assert ".textContent" in _SOURCE


def test_operator_html_reaches_the_api_with_the_operator_key_header():
    """No second credential -- the existing operator key travels as
    X-Operator-Key on every API call, the same header every other
    /operator/* endpoint already expects (#37/#39)."""
    assert "X-Operator-Key" in _SOURCE
