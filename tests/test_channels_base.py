"""Unit tests for the shared bounded dedup cache (app/channels/base.py).

Each channel owns its own SeenKeys instance (see telegram.py's _SEEN and
whatsapp.py's _SEEN) — this is the class both share, tested once here rather
than once per channel.
"""
from app.channels.base import SeenKeys


def test_first_call_returns_false():
    assert SeenKeys().check_and_add("wamid.abc123") is False


def test_second_call_same_key_returns_true():
    seen = SeenKeys()
    seen.check_and_add("wamid.dup1")
    assert seen.check_and_add("wamid.dup1") is True


def test_different_keys_are_not_duplicates():
    seen = SeenKeys()
    assert seen.check_and_add("wamid.x1") is False
    assert seen.check_and_add("wamid.x2") is False


def test_lru_eviction_after_max_entries():
    seen = SeenKeys(max_entries=100)
    for i in range(101):
        seen.check_and_add(f"wamid.evict-{i}")
    assert len(seen) == 100
    assert seen.check_and_add("wamid.evict-0") is False  # was evicted, so not a dup
