"""Wraps a real ChannelAdapter to suppress replies for one turn -- used
while a thread is under human control, where the bot must not speak but
media still needs fetching so the operator reads text, not a placeholder.
See docs/adr/ADR-009-human-control.md.

Not a channel: `ChannelAdapter` is a structural Protocol with no runtime
checks, so this only needs to implement the methods the turn actually calls
-- acknowledge and fetch_media delegate, send is a no-op.
"""
from app.channels.base import ChannelAdapter, Inbound, MediaRef


class SilentAdapter:
    def __init__(self, adapter: ChannelAdapter) -> None:
        self._adapter = adapter
        self.channel = adapter.channel

    async def acknowledge(self, inbound: Inbound) -> None:
        await self._adapter.acknowledge(inbound)

    async def fetch_media(self, ref: MediaRef) -> bytes:
        return await self._adapter.fetch_media(ref)

    async def send(self, inbound: Inbound, text: str) -> None:
        pass
