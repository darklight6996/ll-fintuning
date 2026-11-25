"""
core/stealth.py

THE STEALTH LAYER -- makes it safe to point the framework at a REAL, authorized,
scoped target (a bug-bounty program, a client engagement) instead of only a lab.

On a lab you want speed; on a real target, blasting sqlmap --crawl / full nuclei
/ ffuf wordlists is a few hundred noisy requests that get you rate-limited,
WAF-banned, or in violation of the program's rules -- and burns the operator's
account. The surgical move (one crafted request, read carefully) is also the
un-bannable one, so this layer pushes default behavior toward it on real targets:

  * PACING + JITTER  -- a randomized delay before each ACTIVE request/action, so
    the agent never machine-guns a host.
  * ADAPTIVE BACKOFF -- when a response looks rate-limited / WAF-blocked
    (429/403/"access denied"/vendor signatures), the delay multiplies; quiet
    responses let it ease back down.
  * SCANNER GATING   -- high request-volume actions (the loud scanners) are
    gated OFF on a real target unless explicitly re-enabled, leaving the
    surgical actions (http_probe, browser_probe, targeted probes, self_script).

It is FRAMEWORK-LEVEL and OFF BY DEFAULT: with no stealth flag, every method is a
no-op and behavior (labs, loopback smoke tests) is byte-identical to before.
Activated per engagement via Scope.stealth=True or env PENTEST_STEALTH=1.

Env overrides (only meaningful when stealth is on):
  PENTEST_STEALTH=1 / 0                 force stealth on / off
  PENTEST_STEALTH_MIN_DELAY (sec)       min inter-request delay   (default 1.5)
  PENTEST_STEALTH_MAX_DELAY (sec)       max inter-request delay   (default 4.0)
  PENTEST_STEALTH_ALLOW_SCANNERS=1      permit high-volume scanners under stealth
"""

import os
import random
import time
from typing import Tuple

_WAF_SIGNS = (
    "429", "too many requests", "rate limit", "rate-limit",
    "403 forbidden", "access denied", "request blocked", "you have been blocked",
    "cloudflare", "akamai", "incapsula", "mod_security", "modsecurity",
    "web application firewall", "captcha", "attention required",
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except Exception:
        return default


class StealthController:
    def __init__(self, enabled: bool, min_delay: float = 0.0, max_delay: float = 0.0,
                 aggressive_allowed: bool = True):
        self.enabled = enabled
        self.min_delay = max(0.0, min_delay)
        self.max_delay = max(self.min_delay, max_delay)
        self.aggressive_allowed = aggressive_allowed
        self._backoff = 1.0
        self._max_backoff = 8.0

    @classmethod
    def from_scope(cls, scope) -> "StealthController":
        env = os.environ.get("PENTEST_STEALTH")
        stealth = bool(getattr(scope, "stealth", False))
        if env == "1":
            stealth = True
        elif env == "0":
            stealth = False
        if not stealth:
            return cls(enabled=False)
        return cls(
            enabled=True,
            min_delay=_env_float("PENTEST_STEALTH_MIN_DELAY", 1.5),
            max_delay=_env_float("PENTEST_STEALTH_MAX_DELAY", 4.0),
            aggressive_allowed=(os.environ.get("PENTEST_STEALTH_ALLOW_SCANNERS") == "1"),
        )

    # -- scanner gating ----------------------------------------------------

    def is_action_allowed(self, defn) -> Tuple[bool, str]:
        """Gate high request-volume actions on a real target unless re-enabled."""
        if not self.enabled or self.aggressive_allowed:
            return True, ""
        if getattr(defn, "request_volume", "low") == "high":
            return False, (
                f"Stealth profile active: high request-volume action "
                f"'{getattr(defn, 'action_id', '?')}' is gated off to avoid rate-limit/WAF "
                f"bans on a real target. Use surgical actions (http_probe, browser_probe, "
                f"targeted probes) or set PENTEST_STEALTH_ALLOW_SCANNERS=1 to permit scanners."
            )
        return True, ""

    # -- pacing + adaptive backoff ----------------------------------------

    def pace(self, is_active: bool) -> float:
        """Sleep a jittered delay before an active request. Returns the slept seconds."""
        if not self.enabled or not is_active:
            return 0.0
        base = random.uniform(self.min_delay, self.max_delay)
        delay = min(base * self._backoff, self.max_delay * self._max_backoff)
        if delay > 0:
            time.sleep(delay)
        return delay

    def note_result(self, text: str) -> None:
        """Inspect tool output for block/rate-limit signatures and adjust backoff."""
        if not self.enabled:
            return
        low = (text or "").lower()
        if any(sig in low for sig in _WAF_SIGNS):
            self._backoff = min(self._backoff * 2.0, self._max_backoff)
        else:
            self._backoff = max(1.0, self._backoff * 0.9)

    @property
    def backoff(self) -> float:
        return self._backoff
