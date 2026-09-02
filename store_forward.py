"""
Shogunet store-and-forward
==========================

Bounded JSONL write-ahead log outbox plus a dedup inbox. This is the
always-present final tier of the transport fallback chain: when every
transport is down -- or the only link is a duty-cycled LoRa radio that can
drain a queue slowly -- messages wait here and replay deterministically on
recovery. The WAL pattern mirrors ShugoCore's episodic-memory journal:
append survives crashes, replay restores state, compaction bounds growth.
"""

import base64
import json
import os
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

DEFAULT_MAX_PENDING = 1000


class OutboxStore:
    """Append-only WAL of unsent frames with explicit done-marking."""

    def __init__(self, path: str,
                 max_pending: int = DEFAULT_MAX_PENDING):
        self.path = str(path)
        self.max_pending = max(1, int(max_pending))
        self._pending: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._replay()

    # -- replay ---------------------------------------------------------------

    def _replay(self) -> None:
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
                    if "done" in record:
                        self._pending.pop(int(record["done"]), None)
                    elif "msg_id" in record:
                        entry = self._entry_from_record(record)
                        if entry is not None:
                            self._pending[entry["msg_id"]] = entry
        except FileNotFoundError:
            return
        except OSError:
            return

    def _entry_from_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            frame = base64.b64decode(str(record["frame_b64"]), validate=True)
            return {
                "msg_id": int(record["msg_id"]),
                "peer": str(record.get("peer", "*")),
                "frame": frame,
                "queued_at": float(record.get("queued_at", time.time())),
                "meta": dict(record.get("meta") or {}),
            }
        except (KeyError, TypeError, ValueError):
            return None

    def _append(self, record: Dict[str, Any]) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            pass   # WAL failure must never crash the send path

    # -- operations --------------------------------------------------------------

    def enqueue(self, msg_id: int, peer: Optional[str], frame: bytes,
                meta: Optional[Dict[str, Any]] = None) -> bool:
        """Persist an unsent frame; False when the outbox is at capacity."""
        entry = {
            "msg_id": int(msg_id) & 0xFFFFFFFF,
            "peer": str(peer or "*"),
            "frame": bytes(frame),
            "queued_at": time.time(),
            "meta": dict(meta or {}),
        }
        with self._lock:
            if len(self._pending) >= self.max_pending and entry["msg_id"] not in self._pending:
                return False
            self._pending[entry["msg_id"]] = entry
        self._append({"msg_id": entry["msg_id"], "peer": entry["peer"],
                      "frame_b64": base64.b64encode(entry["frame"]).decode("ascii"),
                      "queued_at": entry["queued_at"], "meta": entry["meta"]})
        return True

    def mark_done(self, msg_id: int) -> bool:
        msg_id = int(msg_id) & 0xFFFFFFFF
        with self._lock:
            removed = self._pending.pop(msg_id, None) is not None
        if removed:
            self._append({"done": msg_id})
        return removed

    def pending(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(entry) for entry in self._pending.values()]

    def __len__(self) -> int:
        with self._lock:
            return len(self._pending)

    def compact(self) -> int:
        """Rewrite the WAL keeping only pending entries; returns kept count.

        The whole rewrite happens under ``self._lock`` so a concurrent
        ``enqueue``/``mark_done`` can never append to the live file between the
        snapshot and ``os.replace`` -- doing so would silently drop those
        records when the temp file replaces the original. This is the
        crash/durability race that a threaded fleet's WAL churn would hit.
        """
        tmp_path = self.path + ".tmp"
        with self._lock:
            kept = list(self._pending.values())
            try:
                with open(tmp_path, "w", encoding="utf-8") as handle:
                    for entry in kept:
                        record = {"msg_id": entry["msg_id"], "peer": entry["peer"],
                                  "frame_b64": base64.b64encode(entry["frame"]).decode("ascii"),
                                  "queued_at": entry["queued_at"],
                                  "meta": entry["meta"]}
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                os.replace(tmp_path, self.path)
            except OSError:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return len(kept)


class InboxDedup:
    """Bounded LRU of (sender, msg_id) for at-least-once dedup.

    ``_order`` is a deque so eviction is O(1); a list's ``pop(0)`` made every
    insert after capacity O(n) -- quadratic churn on a busy fleet's inbound
    stream (4096-entry shifts per message past the cap).
    """

    def __init__(self, capacity: int = 4096):
        self.capacity = max(1, int(capacity))
        self._seen: Dict[Any, None] = {}
        self._order: Deque[Any] = deque()
        self._lock = threading.Lock()

    def is_duplicate(self, sender: str, msg_id: int) -> bool:
        key = (str(sender), int(msg_id) & 0xFFFFFFFF)
        with self._lock:
            if key in self._seen:
                return True
            if len(self._order) >= self.capacity:
                oldest = self._order.popleft()
                self._seen.pop(oldest, None)
            self._seen[key] = None
            self._order.append(key)
        return False
