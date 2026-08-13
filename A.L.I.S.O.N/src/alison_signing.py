"""alison_signing.py -- Tier 3 policy signing scaffold (HMAC-SHA256).

Scaffold-only by design (v3): the policy file carries a `signing` block with
`enabled: false`. When enabled, privileged actions are denied unless the
policy file carries a valid HMAC-SHA256 signature computed over its canonical
JSON with the operator's key (supplied at runtime via ALISON_POLICY_KEY --
never stored on disk). `sign_policy()` is provided for offline key
management tooling.
"""
import hashlib
import hmac
import json


def canonical_policy_bytes(policy):
    """Stable canonical serialization: signing metadata excluded, keys sorted."""
    work = {k: v for k, v in (policy or {}).items() if k != "signing"}
    return json.dumps(work, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_policy(policy, key, key_id="placeholder-tier3"):
    """Return the signing block for `policy` signed with `key` (HMAC-SHA256)."""
    if not key:
        raise ValueError("sign_policy requires a non-empty key")
    sig = hmac.new(key.encode("utf-8"), canonical_policy_bytes(policy),
                   hashlib.sha256).hexdigest()
    return {"enabled": True, "key_id": key_id,
            "algorithm": "hmac-sha256", "signature": sig}


def verify_policy_signature(policy, key):
    """True iff policy['signing'] is present, enabled, and its HMAC-SHA256
    signature matches. Deny-by-default on any inconsistency."""
    try:
        signing = (policy or {}).get("signing") or {}
        if not signing.get("enabled"):
            return True
        expected = signing.get("signature")
        if not expected:
            return False
        if not key:
            return False
        sig = hmac.new(key.encode("utf-8"), canonical_policy_bytes(policy),
                       hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, expected)
    except Exception:
        return False