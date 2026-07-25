#!/usr/bin/env python
"""Manual test: OCR stack with real medical order images."""
import asyncio
import io
from pathlib import Path

from PIL import Image

import app.services.vision as vision_module
from app.services.vision import VISION_UNCERTAIN, extract_procedure_query

# Real images from WhatsApp
IMAGES = {
    "biopsy_1": Path("C:/Users/rcman/Downloads/WhatsApp Image 2026-07-21 at 9.48.45 AM.jpeg"),
    "biopsy_2": Path("C:/Users/rcman/Downloads/WhatsApp Image 2026-07-21 at 11.42.00 AM.jpeg"),
}

# End-to-end validation set: real biopsy request forms (handwritten, varying
# legibility) used to check the generalized vision prompt against actual
# hard cases via a real LLM call, not a mocked response.
_REPO_ROOT = Path(__file__).parent.parent  # moved from repo root into tests/
E2E_IMAGES = {
    "real_1": _REPO_ROOT / "test_images" / "real_1.jpeg",
    "real_2": _REPO_ROOT / "test_images" / "real_2.jpeg",
    "real_3": _REPO_ROOT / "test_images" / "real_3.jpeg",
    "real_4": _REPO_ROOT / "test_images" / "real_4.jpeg",
    "whatsapp_1": _REPO_ROOT / "test_images" / "WhatsApp Image 2026-07-22 at 1.53.52 PM.jpeg",
    "whatsapp_2": _REPO_ROOT / "test_images" / "WhatsApp Image 2026-07-22 at 11.39.57 AM (1).jpeg",
    "whatsapp_3": _REPO_ROOT / "test_images" / "WhatsApp Image 2026-07-23 at 2.01.23 PM.jpeg",
    "whatsapp_4": _REPO_ROOT / "test_images" / "WhatsApp Image 2026-07-24 at 10.39.23 AM.jpeg",
}

# "demo" is the actual lab tenant configured in dev (expertise_area:
# "Patológia- Biopsias-Citologías" — pathology/biopsies/cytology), matching
# the SP Unidad de Diagnóstico Histológico catalog. Exercises the real
# per-tenant cache-key and specialization-context path instead of "".
_LAB_TENANT_SLUG = "demo"


def test_preprocess_real_images():
    """Test preprocessing on real medical order images."""
    print("\n=== Testing preprocessing on real medical images ===\n")

    for name, path in IMAGES.items():
        if not path.exists():
            print(f"SKIP: {name} not found at {path}")
            continue

        print(f"Testing {name} ({path.name})...")
        img_bytes = path.read_bytes()

        # Original
        orig_img = Image.open(io.BytesIO(img_bytes))
        orig_size = len(img_bytes)
        print(f"  Original: {orig_img.size} {orig_size/1024:.1f}KB")

        # Preprocess
        processed = vision_module._preprocess_image(img_bytes)
        proc_img = Image.open(io.BytesIO(processed))
        proc_size = len(processed)
        print(f"  Processed: {proc_img.size} {proc_size/1024:.1f}KB")

        # Verify
        assert len(processed) > 0, "Processed image empty"
        assert proc_img.size[0] > 100, "Width too small"
        assert proc_img.size[1] > 100, "Height too small"

        orig_ratio = orig_img.width / orig_img.height
        proc_ratio = proc_img.width / proc_img.height
        ratio_diff = abs(orig_ratio - proc_ratio)
        print(f"  Aspect ratio: {orig_ratio:.3f} -> {proc_ratio:.3f} (diff: {ratio_diff:.3f})")
        assert ratio_diff < 0.05, f"Aspect ratio changed too much: {ratio_diff}"

        print("  Result: OK\n")


def test_ocr_text_extraction():
    """Test OCR text extraction simulation."""
    print("\n=== Testing OCR text extraction ===\n")

    # Example: what Tesseract would extract from biopsy forms
    test_cases = [
        {
            "name": "Histerectomia",
            "text": "Histerectomia abdominal radical\nBiopsia de tejido endometrial",
            "expected": "Histerectomia abdominal radical",
        },
        {
            "name": "Laparatomia",
            "text": "Laparatomia Exploración\nComeolopía",
            "expected": "Laparatomia Exploración",
        },
    ]

    for test in test_cases:
        print(f"Testing: {test['name']}")
        lines = [line.strip() for line in test['text'].split("\n") if line.strip()]
        procedure = next((line for line in lines if len(line) > 3), None)
        print(f"  Extracted: {procedure}")
        print(f"  Expected: {test['expected']}")
        assert procedure == test['expected'], f"Mismatch: {procedure} != {test['expected']}"
        print("  Result: OK\n")


async def run_real_extraction_end_to_end():
    """Call the real vision LLM (no mocks) against real biopsy request photos.

    Named without a `test_` prefix on purpose: this makes a real, billed LLM
    call and pytest's default discovery has no `testpaths` restriction, so a
    `test_`-prefixed async function here gets silently collected and run
    unattended in CI (which does have OPENROUTER_API_KEY configured) — with
    no asyncio plugin loaded for this root-level manual script, that failed
    outright. The fix is exclusion from collection, not adding a marker: a
    live paid API call must never run unattended in automated CI regardless.

    Validates the generalized prompt on actual hard cases: handwritten,
    photographed on a bed/table/desk, varying legibility. Ground truth isn't
    available for these (no oracle), so success means the pipeline returns a
    plausible price_question OR safely reports VISION_UNCERTAIN — it must
    never crash and must never look confident on a claim it can't back up."""
    print("\n=== Testing real vision extraction (live API call) ===\n")

    for name, path in E2E_IMAGES.items():
        if not path.exists():
            print(f"SKIP: {name} not found at {path}")
            continue

        print(f"Testing {name} ({path.name})...")
        img_bytes = path.read_bytes()
        try:
            result = await extract_procedure_query(img_bytes, "", tenant_slug=_LAB_TENANT_SLUG)
        except Exception as exc:
            print(f"  FAIL: raised {exc!r} (must never crash)\n")
            raise

        if result == VISION_UNCERTAIN:
            print("  Result: VISION_UNCERTAIN (safe default on illegible/ambiguous input)\n")
        else:
            print(f"  Result: {result}\n")
        assert result, "extraction must never return empty/falsy"


if __name__ == "__main__":
    test_preprocess_real_images()
    test_ocr_text_extraction()
    asyncio.run(run_real_extraction_end_to_end())
    print("\n=== All tests PASSED ===\n")
