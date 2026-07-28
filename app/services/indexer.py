import io
import json
import logging
import uuid

import filetype
import pypdf
from sqlalchemy import text

from app.config import settings
from app.db import AsyncSessionLocal
from app.models import DocumentChunk, IndexJob, IndexJobStatus
from app.services import md_catalog
from app.services.llm import get_embeddings
from app.services.security import scan_chunk_for_injection

logger = logging.getLogger(__name__)

_EMBED_BATCH = 20  # chunks per embedding call


def _extract_pages(content: bytes, filename: str) -> list[tuple[str, int]]:
    kind = filetype.guess(content)
    if kind and kind.mime == "application/pdf":
        reader = pypdf.PdfReader(io.BytesIO(content))
        return [(page.extract_text() or "", i) for i, page in enumerate(reader.pages)]
    return [(content.decode("utf-8", errors="replace"), 0)]


def _chunk_page(text: str, source: str, page: int) -> list[dict]:
    size, overlap = settings.chunk_size, settings.chunk_overlap
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    merged, current = [], ""

    def _tail(chunk: str) -> str:
        # chunk_overlap is only honored inside the oversized-paragraph split
        # below (a fixed-stride sliding window) — every other chunk boundary
        # (the common case: paragraphs merged up to `size`) used to start the
        # next chunk from scratch, silently dropping the configured overlap
        # at every split point. Carry the tail of the just-finalized chunk
        # forward so continuity holds at every boundary, not just that one.
        return chunk[-overlap:] if overlap > 0 else ""

    for para in paragraphs:
        if len(para) > size:
            if current:
                merged.append(current)
            for start in range(0, len(para), size - overlap):
                merged.append(para[start : start + size])
            current = _tail(merged[-1])
        elif not current:
            current = para
        elif len(current) + 2 + len(para) <= size:
            current += "\n\n" + para
        else:
            merged.append(current)
            tail = _tail(current)
            current = f"{tail}\n\n{para}" if tail else para
    if current:
        merged.append(current)
    return [
        {"content": c.strip(), "source": source, "page": page}
        for c in merged
        if len(c.strip()) > 50 and not scan_chunk_for_injection(c)
    ]


def _extract_jsonl_chunks(content: bytes, filename: str) -> tuple[list[dict], dict | None]:
    """Parse JSONL catalog format — each line = one pre-structured embedding chunk.

    Supported fields per line:
      id        (str)   — item code, used as source key
      name      (str)   — display name
      price     (float) — price in USD
      type      (str)   — "biopsy" | "protocol" | "cytology" | etc. Alias: "doc_type"
                          (e.g. "info" | "price_table" | "policy" | "faq").
                          Special value "org_metadata" skips embedding and instead
                          returns tenant config (expertise_area, contact_url) as second
                          element of the returned tuple.
      category  (str)   — catalog section
      keywords  (list)  — trigger terms for retrieval (critical for protocols)
      text      (str)   — pre-built embedding text; overrides auto-construction.
                          Alias: "content" (used when "text" is absent).
      title     (str)   — optional heading prepended before the embedding text —
                          useful for section-level chunks (e.g. a price table
                          covering several items under one heading).
      description (str) — additional context injected into embedding text
    """
    items: list[dict] = []
    for line_num, raw in enumerate(content.decode("utf-8", errors="replace").splitlines(), start=1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            items.append(json.loads(raw))
        except json.JSONDecodeError:
            logger.warning("jsonl_skip_invalid_line file=%s line=%d", filename, line_num)
    return _records_to_chunks(items, filename)


def _records_to_chunks(records: list[dict], filename: str) -> tuple[list[dict], dict | None]:
    """Build embedding chunks + optional org_metadata from pre-parsed records —
    shared by the JSONL line parser above and app/services/md_catalog.py's
    Markdown-to-records converter, so both input formats produce identical
    chunk/metadata shape without duplicating this logic."""
    chunks: list[dict] = []
    org_meta: dict | None = None
    for line_num, item in enumerate(records, start=1):
        item_type = item.get("type") or item.get("doc_type")

        if item_type == "org_metadata":
            org_meta = item
            logger.info("jsonl_org_metadata file=%s", filename)
            continue

        # Use pre-built text if provided; otherwise construct from structured fields.
        # Keywords are always appended to the embedding regardless of whether text
        # is provided — separating text narrative from retrieval keywords is valid
        # schema design, but the indexer must merge both into the vector.
        prebuilt_text = item.get("text") or item.get("content")
        if prebuilt_text:
            embed_text = str(prebuilt_text)
            if item.get("price") is not None and "$" not in embed_text:
                embed_text += f" ${item['price']:.2f}"
            if item.get("keywords"):
                kws = item["keywords"] if isinstance(item["keywords"], list) else [item["keywords"]]
                embed_text += "\nIndicaciones: " + ", ".join(str(k) for k in kws)
        else:
            parts: list[str] = []
            # "code" (real catalog/business code, e.g. "GIN012") takes
            # priority over "id" (internal source-key slug, e.g.
            # "biopsy-gin012") for what the LLM echoes back as the item's
            # displayed code — found live: the bot was showing customers the
            # internal slug instead of the code printed on the real price
            # list. "id" still drives source-key uniqueness elsewhere,
            # unaffected by this display-only change.
            if item.get("code") or item.get("id"):
                parts.append(str(item.get("code") or item["id"]))
            if item.get("name"):
                parts.append(str(item["name"]))
            if item.get("price") is not None:
                parts.append(f"${item['price']}")
            if item_type:
                parts.append(f"Tipo: {item_type}")
            if item.get("category"):
                parts.append(f"Categoría: {item['category']}")
            if item.get("keywords"):
                kws = item["keywords"] if isinstance(item["keywords"], list) else [item["keywords"]]
                parts.append("Indicaciones: " + ", ".join(str(k) for k in kws))
            if item.get("description"):
                parts.append(str(item["description"]))
            embed_text = "\n".join(parts)

        if item.get("title"):
            embed_text = f"{item['title']}\n{embed_text}"

        if not embed_text or scan_chunk_for_injection(embed_text):
            continue

        # Embedding models cap at ~8k tokens; 6000 chars ≈ safe upper bound for any item
        if len(embed_text) > 6000:
            logger.warning("jsonl_text_truncated file=%s line=%d chars=%d", filename, line_num, len(embed_text))
            embed_text = embed_text[:6000]

        item_id = str(item.get("id", line_num))
        meta: dict = {}
        if item.get("id") is not None:
            meta["id"] = str(item["id"])
        if item.get("price") is not None:
            meta["price"] = float(item["price"])
        if item_type:
            meta["type"] = str(item_type)
        if item.get("category"):
            meta["category"] = str(item["category"])
        if item.get("keywords"):
            kws = item["keywords"] if isinstance(item["keywords"], list) else [item["keywords"]]
            meta["keywords"] = [str(k) for k in kws]

        chunks.append({
            "content": embed_text,
            "source": f"{filename}:{item_id}",
            "page": line_num,
            "chunk_type": str(item_type) if item_type else None,
            "metadata": meta or None,
        })

    return chunks, org_meta


async def run_index_job(
    job_id: uuid.UUID,
    content: bytes,
    filename: str,
    tenant_id: int,
    namespace: str,
    replace_all: bool = False,
) -> None:
    async with AsyncSessionLocal() as db:
        job = await db.get(IndexJob, job_id)
        if not job:
            logger.error("index_job_not_found job=%s", job_id)
            return
        job.status = IndexJobStatus.RUNNING
        await db.commit()

    try:
        # Replace-on-re-upload: delete stale chunks before inserting new ones
        # so the namespace stays clean without manual cleanup. Default scope
        # is this filename only (plain "file.md" or JSONL-keyed "file.jsonl:ID"),
        # so unrelated documents for the same tenant are untouched. replace_all
        # widens this to the whole tenant — e.g. when replacing an entire
        # catalog with a new source file/format.
        async with AsyncSessionLocal() as db:
            if replace_all:
                result = await db.execute(
                    text("DELETE FROM document_chunks WHERE tenant_id = :tid"),
                    {"tid": tenant_id},
                )
            else:
                result = await db.execute(
                    text("""
                        DELETE FROM document_chunks
                         WHERE tenant_id = :tid
                           AND (source = :src OR source LIKE :prefix)
                    """),
                    {"tid": tenant_id, "src": filename, "prefix": f"{filename}:%"},
                )
            replaced = result.rowcount
            await db.commit()
        if replaced:
            logger.info("index_replaced_stale job=%s filename=%s deleted=%d replace_all=%s",
                         job_id, filename, replaced, replace_all)

        all_chunks: list[dict] = []
        org_meta: dict | None = None

        if filename.lower().endswith(".jsonl"):
            all_chunks, org_meta = _extract_jsonl_chunks(content, filename)
        else:
            # Markdown gets a structured-parse attempt first (md_catalog.convert):
            # generic text/paragraph chunking below would slice price tables
            # mid-row, detaching codes from prices (see md_catalog.py docstring
            # for the real drift this caused). Falls back to generic chunking
            # when the .md doesn't match the catalog shape (no price tables,
            # none of the known section headers) — a tenant's free-text info
            # doc is a valid, different use case, not an error.
            records = None
            if filename.lower().endswith((".md", ".markdown")):
                records = md_catalog.convert(content.decode("utf-8", errors="replace"))
            if records:
                all_chunks, org_meta = _records_to_chunks(records, filename)
                logger.info("md_catalog_parsed job=%s filename=%s records=%d", job_id, filename, len(records))
            else:
                pages = _extract_pages(content, filename)
                for text_content, page_num in pages:
                    all_chunks.extend(_chunk_page(text_content, filename, page_num))

        if org_meta:
            async with AsyncSessionLocal() as db:
                await db.execute(
                    text("""
                        UPDATE tenants
                           SET expertise_area = COALESCE(:ea, expertise_area),
                               contact_url    = COALESCE(:cu, contact_url)
                         WHERE id = :tid
                    """),
                    {
                        "ea": str(org_meta["expertise_area"])[:255] if org_meta.get("expertise_area") else None,
                        "cu": str(org_meta["contact_url"])[:512] if org_meta.get("contact_url") else None,
                        "tid": tenant_id,
                    },
                )
                await db.commit()
            logger.info("org_metadata_applied job=%s tenant=%d", job_id, tenant_id)

        if not all_chunks:
            raise ValueError("No content could be extracted from file")

        async with AsyncSessionLocal() as db:
            job = await db.get(IndexJob, job_id)
            job.chunks_total = len(all_chunks)
            await db.commit()

        embedder = get_embeddings()
        done = 0

        for i in range(0, len(all_chunks), _EMBED_BATCH):
            batch = all_chunks[i : i + _EMBED_BATCH]
            vecs = await embedder.aembed_documents([c["content"] for c in batch])

            async with AsyncSessionLocal() as db:
                db.add_all([
                    DocumentChunk(
                        tenant_id=tenant_id,
                        job_id=job_id,
                        namespace=namespace,
                        source=c["source"],
                        page=c["page"],
                        content=c["content"],
                        embedding=vec,
                        chunk_type=c.get("chunk_type"),
                        metadata_=c.get("metadata"),
                    )
                    for c, vec in zip(batch, vecs)
                ])
                done += len(batch)
                job = await db.get(IndexJob, job_id)
                job.chunks_done = done
                await db.commit()

        async with AsyncSessionLocal() as db:
            job = await db.get(IndexJob, job_id)
            job.status = IndexJobStatus.DONE
            await db.commit()

        logger.info("index_job_done job=%s chunks=%d", job_id, done)

    except Exception as exc:
        logger.exception("index_job_failed job=%s", job_id)
        async with AsyncSessionLocal() as db:
            job = await db.get(IndexJob, job_id)
            if job:
                job.status = IndexJobStatus.FAILED
                job.error_message = str(exc)[:500]
                await db.commit()
