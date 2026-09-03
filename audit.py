"""
Shogunet network audit log
==========================

Append-only, hash-chained audit log mirroring ShugoCore's ``AuditChain``
construction and its ``shugocore-verify-audit`` CLI contract. Every network
event -- pairing, handshake refusal, breaker open, memory conflict, policy
refusal -- lands here with a previous-hash link, so the fleet's transport
history is tamper-evident.

Each record:

    {"seq": N, "ts": iso, "event": str, "payload": {...},
     "prev": sha256(prev_record_bytes), "hash": sha256(this_record_bytes)}

Verification walks the chain and recomputes every link; any edit, deletion
or reorder is detected. A failed append never raises into the sender.
"""

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

GENESIS_HASH = "0" * 64


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AuditChain:
    """Append-only hash-chained audit log persisted as JSONL."""

    def __init__(self, path: str):
        self.path = str(path)
        self._lock = threading.Lock()
        self._seq = 0
        self._tail = GENESIS_HASH
        # Live tail subscribers (the fleet dashboard's SSE feed). Callbacks
        # receive every appended record dict; subscriber failures are ignored.
        self._subscribers: List[Callable[[Dict[str, Any]], None]] = []
        self._load_existing()

    # -- internals -----------------------------------------------------------

    def _load_existing(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record.get("seq"), int) \
                            and record["seq"] > self._seq:
                        self._seq = record["seq"]
                    if isinstance(record.get("hash"), str) \
                            and len(record["hash"]) == 64:
                        self._tail = record["hash"]
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("audit load failed: %s", exc)

    def _record_hash(self, record: Dict[str, Any]) -> str:
        body = {"seq": record["seq"], "ts": record["ts"],
                "event": record["event"], "payload": record["payload"],
                "prev": record["prev"]}
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

# -- surface ---------------------------------------------------------------

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a live-tail callback invoked with every new record."""
        self._subscribers.append(callback)

    def append(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Append an event; the chain hash closes over the previous record."""
        with self._lock:
            self._seq += 1
            record = {
                "seq": self._seq,
                "ts": _utc_now_iso(),
                "event": str(event_type),
                "payload": dict(payload or {}),
                "prev": self._tail,
            }
            record["hash"] = self._record_hash(record)
            self._tail = record["hash"]
            try:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            except OSError as exc:
                # A failed audit write must never crash the message path.
                logger.warning("audit append failed: %s", exc)
        # Notify outside the lock: a slow subscriber must never stall the
        # audited code path, and its failures must never surface here.
        for callback in list(self._subscribers):
            try:
                callback(record)
            except Exception:
                logger.warning("audit subscriber failed", exc_info=True)
        return dict(record)

    def verify(self) -> List[Dict[str, Any]]:
        """Walk the chain; returns a list of one dict per problem found."""
        problems: List[Dict[str, Any]] = []
        previous = GENESIS_HASH
        seen: set = set()
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        problems.append({"line": line_number,
                                         "reason": "corrupt json",
                                         "detail": str(exc)})
                        continue
                    if record.get("prev") != previous:
                        problems.append({"line": line_number,
                                         "seq": record.get("seq"),
                                         "reason": "broken link"})
                        continue
                    expected = self._record_hash(record)
                    if record.get("hash") != expected:
                        problems.append({"line": line_number,
                                         "seq": record.get("seq"),
                                         "reason": "hash mismatch"})
                    if record.get("seq") in seen:
                        problems.append({"line": line_number,
                                         "seq": record.get("seq"),
                                         "reason": "duplicate seq"})
                    seen.add(record.get("seq"))
                    previous = record.get("hash", GENESIS_HASH)
        except FileNotFoundError:
            return problems
        except OSError as exc:
            problems.append({"reason": "io error", "detail": str(exc)})
        return problems

    @property
    def tail(self) -> str:
        with self._lock:
            return str(self._tail)

    def __len__(self) -> int:
        with self._lock:
            return int(self._seq)


def cli_main() -> int:
    """Verify an audit chain: ``python3 -m shugonet-audit path``."""
    import sys
    if len(sys.argv) != 2:
        print("usage: python3 -m audit <audit_path>")
        return 2
    chain = AuditChain(sys.argv[1])
    problems = chain.verify()
    if problems:
        for problem in problems:
            print("INVALID:", problem)
        return 1
    print(f"OK: {len(chain)} records, tail {chain.tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())