"""Policy-as-code: declarative per-tenant authorization.

Handles quotas, rate limiting, and plan-based feature access.
Policies are evaluated at trust boundaries (index uploads, queries).

Usage:
    policy = TenantPolicy(tenant_slug="acme", plan="basic")
    if not policy.can_upload_doc(doc_count=5):
        raise HTTPException(429, "Reached document limit for Basic plan")
"""
from dataclasses import dataclass
from typing import Literal

from app.config import PLAN_LIMITS


@dataclass
class TenantPolicy:
    """Declarative policy for a single tenant.

    Plan choices: "free" | "basic" | "pro"
    Defaults to "free" when missing from config.
    """
    tenant_slug: str
    plan: Literal["free", "basic", "pro"] = "free"

    def _limit(self, key: str) -> int:
        """Get limit for this plan, fallback to free if plan missing."""
        return PLAN_LIMITS.get(self.plan, PLAN_LIMITS["free"])[key]

    def can_upload_doc(self, doc_count: int) -> bool:
        """Check if tenant can upload another document."""
        return doc_count < self._limit("docs")

    def get_limits(self) -> dict:
        """Get all limits for this plan."""
        return self._limit("docs"), self._limit("chunks"), self._limit("queries_monthly")

    def get_pricing(self) -> dict:
        """Get pricing info for this plan."""
        prices = {"free": 0, "basic": 5, "pro": 10}
        return {
            "plan": self.plan,
            "price_usd": prices.get(self.plan, 0),
            "limits": {
                "docs": self._limit("docs"),
                "chunks": self._limit("chunks"),
                "queries_monthly": self._limit("queries_monthly"),
            }
        }
