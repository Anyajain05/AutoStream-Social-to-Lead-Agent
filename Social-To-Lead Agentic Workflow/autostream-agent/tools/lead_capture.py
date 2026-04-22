"""
tools/lead_capture.py
Mock lead capture tool for AutoStream agent.
"""
import json
from datetime import datetime


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Mock API call to capture a qualified lead.
    In production, this would POST to a CRM or marketing automation system.

    Args:
        name:     Full name of the prospect
        email:    Email address of the prospect
        platform: Social/video platform they create on (YouTube, Instagram, etc.)

    Returns:
        dict with status and lead_id
    """
    timestamp = datetime.utcnow().isoformat()
    lead_id = f"LEAD-{abs(hash(email)) % 100000:05d}"

    # ── Simulated CRM write ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  ✅  LEAD CAPTURED SUCCESSFULLY")
    print("=" * 55)
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"  Lead ID  : {lead_id}")
    print(f"  Time     : {timestamp} UTC")
    print("=" * 55 + "\n")

    return {
        "status": "success",
        "lead_id": lead_id,
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": timestamp,
    }
