from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from fastapi import Request


def thread_id_for(tenant_slug: str, user_id: str, channel: str) -> str:
    """The graph checkpoint key — see ADR-002. The single place this format is
    written; parse it back with app/graph/thread.py:parse_thread_part."""
    return f"tenant:{tenant_slug}:user:{user_id}:channel:{channel}"


MediaKind = Literal["image", "audio", "document"]


@dataclass
class MediaRef:
    """A pointer to media the channel holds, resolvable to bytes via
    ChannelAdapter.fetch_media.

    Carries size_bytes so the inbound turn can reject oversized media without
    downloading it. None means the channel didn't say — the turn does not
    gate on an unknown size.
    """
    id: str
    kind: MediaKind
    size_bytes: int | None = None
    mime_type: str | None = None
    filename: str | None = None


@dataclass
class Inbound:
    """A parsed inbound message: the only thing that crosses from a channel
    adapter into the inbound turn.

    `text` and `media` are exclusive — text messages carry no refs, media
    messages carry no text (a caption travels in `caption`). More than one
    ref means the channel batched them (a Telegram album).
    """
    tenant_slug: str
    channel: str
    user_id: str
    chat_id: str
    message_id: str = ""
    caption: str = ""
    text: str | None = None
    media: list[MediaRef] = field(default_factory=list)

    @property
    def thread_id(self) -> str:
        return thread_id_for(self.tenant_slug, self.user_id, self.channel)


class ChannelAdapter(Protocol):
    """Everything that varies between channels. Everything that doesn't lives
    in the inbound turn (app/channels/turn.py) instead.

    Concrete adapters: TelegramAdapter (channels/telegram.py),
    WhatsAppAdapter (channels/whatsapp.py).
    """
    channel: str

    async def verify(self, request: "Request") -> bool:
        """Return True if the request's authentication credential is valid.
        The enforcement point named by ADR-004 — a webhook route must call
        this before doing anything else with the body."""
        ...

    def dedup_key(self, body: dict) -> str | None:
        """Stable key identifying this delivery, for drop-on-redelivery.
        None when the payload carries no usable id.

        Must be unique across tenants: Telegram's update_id is sequential per
        bot, not globally, so two tenants can emit the same one and a
        tenant-blind key would silently drop one of their messages."""
        ...

    async def parse(self, body: dict) -> list["Inbound"]:
        """Parse a raw webhook payload into zero or more inbound messages.

        A list, not a single value: one WhatsApp payload can carry several
        messages, and each becomes its own turn. Telegram returns 0 or 1.
        """
        ...

    async def acknowledge(self, inbound: "Inbound") -> None:
        """Signal receipt to the user (typing indicator, read receipt).
        Best-effort — the turn swallows failures here."""
        ...

    async def fetch_media(self, ref: "MediaRef") -> bytes:
        """Download the bytes behind a MediaRef."""
        ...

    async def send(self, inbound: "Inbound", text: str) -> None:
        """Deliver a reply to the originating user."""
        ...


class SeenKeys:
    """Bounded LRU of dedup keys already handled.

    Per-process, so it only dedupes redeliveries hitting the same worker —
    fine while entrypoint.sh pins --workers 1 (see app/runtime.py). A second
    worker needs a shared adapter behind this same interface.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self._seen: OrderedDict[str, bool] = OrderedDict()
        self._max = max_entries

    def check_and_add(self, key: str) -> bool:
        """True if the key was already seen. Records it either way."""
        if key in self._seen:
            return True
        self._seen[key] = True
        if len(self._seen) > self._max:
            self._seen.popitem(last=False)
        return False

    def clear(self) -> None:
        self._seen.clear()

    def __len__(self) -> int:
        return len(self._seen)


@dataclass
class ChannelEvent:
    """Legacy — superseded by Inbound. Still referenced by the unmigrated
    WhatsApp handler; delete with that migration."""
    tenant_slug: str
    channel: str
    user_id: str
    chat_id: str
    text: str
    thread_id: str
