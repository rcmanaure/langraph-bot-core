from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String, Text, func
from sqlalchemy.ext.hybrid import hybrid_property

from app.models.base import Base

# Free-text voice/register description injected into the generation prompt
# (see app/graph/nodes/generate.py). Free text, not an enum, because tenants
# span arbitrary business types (clinic, gym, bakery...) — a fixed preset list
# can't cover that and would require a code change per new tenant vertical.
DEFAULT_TONE_DESCRIPTION = (
    "cálido y cercano, como una persona del negocio respondiendo por chat. "
    "Nada de lenguaje robótico, emojis casuales están bien"
)


class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True)
    slug = Column(String(64), unique=True, nullable=False, index=True)
    api_key_hash = Column(String(64), unique=True, nullable=False)
    webhook_secret = Column(String(64), nullable=False)
    bot_token = Column(String(128), unique=True, nullable=False)
    plan = Column(String(32), default="free")
    expertise_area = Column(String(255), nullable=True, default="")
    tone_description = Column(
        Text, nullable=False, default=DEFAULT_TONE_DESCRIPTION, server_default=DEFAULT_TONE_DESCRIPTION
    )
    # Free-text domain/jargon context fed into the RAG and vision prompts so the
    # model interprets vertical-specific terminology (e.g. medical/lab jargon).
    # Separate from expertise_area (which stays short — it doubles as a display
    # label in greetings/off-topic messages and the admin table) — see
    # tone_description above for the same short-label-vs-long-content split.
    specialization_context = Column(Text, nullable=False, default="", server_default="")
    contact_url = Column(String(512), nullable=True)
    example_questions = Column(JSON, nullable=True)
    web_search_enabled = Column(Boolean, default=False, server_default="false")
    doc_structure_summary = Column(Text, nullable=True)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # WhatsApp
    wa_phone_number_id = Column(String(100), nullable=True)
    _wa_access_token = Column("wa_access_token", Text, nullable=True)
    _wa_app_secret = Column("wa_app_secret", Text, nullable=True)
    wa_verify_token = Column(String(100), nullable=True)
    channels = Column(Text, nullable=True, server_default="telegram")

    # Portal login
    portal_password_hash = Column(String(128), nullable=True)

    @hybrid_property
    def wa_access_token(self):
        from app.crypto import decrypt_value
        return decrypt_value(self._wa_access_token) if self._wa_access_token else None

    @wa_access_token.setter
    def wa_access_token(self, value):
        from app.crypto import encrypt_value
        self._wa_access_token = encrypt_value(value) if value else None

    @hybrid_property
    def wa_app_secret(self):
        from app.crypto import decrypt_value
        return decrypt_value(self._wa_app_secret) if self._wa_app_secret else None

    @wa_app_secret.setter
    def wa_app_secret(self, value):
        from app.crypto import encrypt_value
        self._wa_app_secret = encrypt_value(value) if value else None
