"""
Shogunet network fallback controller
====================================

Deterministic safety latches for the networked fleet, mirroring ShugoCore's
``FallbackController`` contract exactly:

- ``pause``      (default): reject new network sends, let in-flight drain
- ``safe_state`` : read-only traffic only; side-effecting actions refused
- ``halt``       : terminal until process restart

The controller never decides anything with a model: it reacts to *reports*
from the transport chain (``TransportChain.health()``) and the memory mesh
(conflict storms), and latches a **governor** that must engage state first
-- the notify-before-latch contract ShugoCore depends on. If the governor
cannot engage, the controller does NOT claim it latched.

Severity map (mirrors ShugoCore's ``DEFAULT_SEVERITIES`` for its own
network triggers):

- ``network_transport_exhausted`` -> pause
- ``network_peer_lost``           -> pause
- ``memory_sync_conflict_storm``  -> safe_state
- ``audit_chain_broken``          -> halt
"""

import logging
import threading
import time
from collections import defaultdict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class NetworkFallbackHalt(RuntimeError):
    """Raised when the HALT escalation fires; unwinds the calling path."""


NETWORK_FALLBACK_SEVERITIES = {
    "network_transport_exhausted": "pause",
    "network_peer_lost": "pause",
    "memory_sync_conflict_storm": "safe_state",
    "audit_chain_broken": "halt",
}

_IMMEDIATE = {"audit_chain_broken"}
_STORM_THRESHOLD = 20          # conflicts per minute -> safe-state latch


class NetworkFallbackController:
    """Latches a governor from network-transport and memory-mesh signals."""

    def __init__(self, governor: Any, audit: Optional[Any] = None,
                 severities: Optional[Dict[str, str]] = None,
                 conflict_storm_threshold: int = _STORM_THRESHOLD):
        self.governor = governor
        self.audit = audit
        self.severities = {**NETWORK_FALLBACK_SEVERITIES,
                           **(severities or {})}
        self.conflict_storm_threshold = max(1, int(conflict_storm_threshold))
        self.mode = "normal"
        self._violations: Dict[str, int] = defaultdict(int)
        self._conflict_window: list = []
        self._lock = threading.Lock()

    # -- reactive reports -------------------------------------------------------

    def report_violation(self, kind: str, detail: str = "") -> None:
        """Record a network/mesh violation; escalates when its rule fires."""
        kind = str(kind)
        with self._lock:
            self._violations[kind] += 1
            count = self._violations[kind]
        severity = str(self.severities.get(kind, "pause")).lower()
        if kind == "memory_sync_conflict_storm":
            # Thresholded: only fires once a 60s window holds enough conflicts.
            now = time.monotonic()
            with self._lock:
                self._conflict_window.append(now)
                window = [t for t in self._conflict_window
                          if now - t <= 60.0]
                self._conflict_window = window
            if len(self._conflict_window) < self.conflict_storm_threshold:
                return
        if severity == "halt" or kind in _IMMEDIATE:
            self._fire(kind, detail)
            return
        # Any other configured trigger fires on first report.
        self._fire(kind, detail)

    # -- internals ---------------------------------------------------------------

    def _fire(self, kind: str, detail: str) -> None:
        severity = str(self.severities.get(kind, "pause")).lower()
        reason = f"{kind}: {str(detail)[:150]}"
        self._audit("network_fallback_trigger",
                    {"trigger": kind, "severity": severity,
                     "detail": str(detail)[:200]})
        if severity == "halt":
            self.governor.halt(reason)
            with self._lock:
                self.mode = "halted"
            self._audit("network_fallback_halt", {"reason": reason})
            raise NetworkFallbackHalt(reason)
        if severity == "safe_state":
            self.governor.safe_state(reason)
            with self._lock:
                self.mode = "safe_state"
        else:
            self.governor.pause(reason)
            with self._lock:
                self.mode = "paused"

    def resume(self, resumed_by: str = "") -> None:
        """Operator resume (attribution required). HALT is terminal."""
        self.governor.resume(resumed_by=resumed_by)
        with self._lock:
            self._violations.clear()
            self._conflict_window.clear()
            self._mode = "normal"
        self._audit("network_fallback_resume",
                    {"resumed_by": str(resumed_by)[:120]})

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"mode": self.mode, "violations": dict(self._violations)}

    def _audit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.audit is None:
            return
        try:
            self.audit.append(event_type, payload)
        except Exception:
            pass