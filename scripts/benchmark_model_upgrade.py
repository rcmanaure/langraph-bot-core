#!/usr/bin/env python
"""Baseline benchmark for the chat/vision/rerank model upgrade (issue #21).

Measures per-stage latency and a small known-answer accuracy check. Run
once against the current models, then again after swapping app/config.py's
model defaults (or the real .env, if it overrides them) -- so the upgrade's
effect is a number, not an impression, per this ticket's own acceptance
criteria.

No live Postgres/indexed catalog needed: the chat check sends a small
fixed catalog context directly to the model (same prompt shape as
generate.py); the rerank check sends a small fixed candidate list; the
vision check sends a synthetic generated image, since no real labeled
doctor's-order image is available in this environment (the ones
tests/test_medical_orders_real.py and scripts/benchmark_accuracy.py
reference live on a different developer's machine and are absent here) --
that check is latency-only, not accuracy, and says so in its own output
rather than fabricating a score nothing here can support.

Makes real, billed API calls. Not part of the pytest suite -- run by hand:
    .venv/Scripts/python.exe scripts/benchmark_model_upgrade.py
"""
import asyncio
import base64
import time
from io import BytesIO

from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image, ImageDraw

from app.config import settings
from app.services.llm import get_chat_llm, get_vision_llm
from app.services.rerank import _call_rerank_api

_CATALOG_CONTEXT = """\
- *SRP009* Pulmón – PAFF: $90.00
- *SRP011* Lobectomía: $240.00
- *SRP020* Biopsia de piel: $55.00
"""

_SYSTEM = f"""Eres un asistente de diagnóstico histológico. Sos formal y profesional.
Para precios usa ÚNICAMENTE el contexto. BREVE: máximo 4-5 líneas.
Contexto:
{_CATALOG_CONTEXT}
"""

_CHAT_CASES = [
    ("cuanto cuesta la biopsia de pulmon", "90"),
    ("cuanto cuesta la lobectomia", "240"),
    ("cuanto cuesta la biopsia de piel", "55"),
]


async def _bench_chat() -> dict:
    llm = get_chat_llm()
    results = []
    for query, expected in _CHAT_CASES:
        start = time.perf_counter()
        response = await llm.ainvoke([SystemMessage(content=_SYSTEM), HumanMessage(content=query)])
        elapsed = time.perf_counter() - start
        results.append({
            "query": query,
            "correct": expected in response.content,
            "seconds": round(elapsed, 2),
        })
    return {
        "model": settings.openai_model,
        "cases": results,
        "accuracy": sum(r["correct"] for r in results) / len(results),
        "avg_seconds": round(sum(r["seconds"] for r in results) / len(results), 2),
    }


async def _bench_rerank() -> dict:
    query = "biopsia de pulmon"
    # Index 0 is the only document that actually matches the query --
    # correct behavior ranks it first.
    documents = [
        "SRP009 Pulmón PAFF biopsia pulmonar",
        "SRP020 Biopsia de piel",
        "SRP011 Lobectomía pulmonar",
    ]
    start = time.perf_counter()
    results = await _call_rerank_api(query, documents, top_n=3)
    elapsed = time.perf_counter() - start
    top_index = max(results, key=lambda r: r["relevance_score"])["index"]
    return {
        "model": settings.rerank_model,
        "top_result_correct": top_index == 0,
        "seconds": round(elapsed, 2),
    }


def _synthetic_test_image() -> bytes:
    img = Image.new("RGB", (400, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 80), "BIOPSIA DE PULMON", fill="black")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def _bench_vision() -> dict:
    llm = get_vision_llm()
    img_b64 = base64.b64encode(_synthetic_test_image()).decode()
    start = time.perf_counter()
    response = await llm.ainvoke([
        HumanMessage(content=[
            {"type": "text", "text": "Qué procedimiento médico dice esta imagen? Respondé solo el nombre."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ])
    ])
    elapsed = time.perf_counter() - start
    return {
        "model": settings.openai_vision_model,
        "seconds": round(elapsed, 2),
        "sample_response": response.content[:80],
        "note": "latency only -- no real labeled doctor's-order image available in this environment",
    }


async def main() -> None:
    print("chat:", await _bench_chat())
    print("rerank:", await _bench_rerank())
    print("vision:", await _bench_vision())


if __name__ == "__main__":
    asyncio.run(main())
