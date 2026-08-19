"""Test plan-based policy enforcement."""
from app.policies import TenantPolicy


def test_free_plan_limits():
    """Free plan has 5 docs, 500 chunks, 500 queries/month."""
    policy = TenantPolicy(tenant_slug="test", plan="free")
    docs, chunks, queries = policy.get_limits()
    assert docs == 5
    assert chunks == 500
    assert queries == 500


def test_basic_plan_limits():
    """Basic plan has 20 docs, 2000 chunks, 2000 queries/month."""
    policy = TenantPolicy(tenant_slug="test", plan="basic")
    docs, chunks, queries = policy.get_limits()
    assert docs == 20
    assert chunks == 2000
    assert queries == 2000


def test_pro_plan_limits():
    """Pro plan has 100 docs, 10k chunks, 10k queries/month."""
    policy = TenantPolicy(tenant_slug="test", plan="pro")
    docs, chunks, queries = policy.get_limits()
    assert docs == 100
    assert chunks == 10000
    assert queries == 10000


def test_can_upload_doc_within_limit():
    """Can upload when below limit."""
    policy = TenantPolicy(tenant_slug="test", plan="free")
    assert policy.can_upload_doc(4) is True  # 4 < 5
    assert policy.can_upload_doc(5) is False  # 5 >= 5


def test_pricing_info():
    """Pricing info includes USD cost."""
    policies = {
        "free": (0, 5, 500),
        "basic": (5, 20, 2000),
        "pro": (10, 100, 10000),
    }
    for plan, (price, docs, queries) in policies.items():
        policy = TenantPolicy(tenant_slug="test", plan=plan)
        info = policy.get_pricing()
        assert info["plan"] == plan
        assert info["price_usd"] == price
        assert info["limits"]["docs"] == docs


def test_default_plan_is_free():
    """Tenants default to free plan."""
    policy = TenantPolicy(tenant_slug="test")
    assert policy.plan == "free"
    assert policy.can_upload_doc(5) is False  # Free limit is 5
