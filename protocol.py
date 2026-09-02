"""
Shogunet wire protocol (v1)
===========================

One envelope protocol spoken over every transport, from 5G to LoRa.

Frame layout -- 16-byte fixed binary header, then one codec payload:

    offset  size  field
    ------  ----  ------------------------------
    0       3     magic   b"SGN"
    3       1     version PROTOCOL_VERSION
    4       1     codec   CODEC_JSON | CODEC_COMPACT
    5       1     priority class (CLASS_CONTROL..CLASS_BULK)
    6       1     msg type (uint8, MESSAGE_TYPES registry)
    7       1     flags (bit 0 = segmented)
    8       4     msg_id (uint32, big-endian)
    12      4     payload length (uint32, big-endian)

The priority class rides in the header so the fallback chain can route and
queue without decoding the payload:

- P0 control (announce / heartbeat / ack / tombstone) is eligible on every
  link, including LoRa;
- P1 memory deltas (memory_fact / reinforce / memory_digest) are the
  constrained-link workhorse of the codependent memory mesh -- over LoRa a
  ``reinforce`` message is a few dozen bytes;
- P2 (tasks, queries, promotion proposals) and P3 (bulk snapshots, audit
  shipping) require IP-class or segmentation-capable links.

Codecs
------
- ``CODEC_JSON``: payload is a compact JSON object with fixed keys (``s``
  sender, ``r`` recipient, ``t`` topic, ``re`` realm, ``ts`` created_at, ``p``
  payload dict). Broadband links, where bytes are cheap.

- ``CODEC_COMPACT``: fixed fields packed binary, then payload values as
  self-describing TLV entries keyed by the FIELD_NAMES registry. Payload keys
  outside the registry fall back to a single JSON ``_blob`` entry, so the
  codec is always lossless. Used on EDGE / WiFi-Halow / LoRa / BLE, where
  every byte is airtime.

Both codecs are deterministic, bounded, and validated on decode: unknown
versions, unknown types, class/type mismatches, oversized frames and
non-printable strings are refused as ``ProtocolError`` -- malformed input is
rejected at the boundary, never propagated raw into memory or decisions.
"""

import json
import math
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from security import sanitize_text

PROTOCOL_VERSION = 1
MAGIC = b"SGN"
HEADER_LEN = 16
HEADER_FMT = ">3s5BII"   # magic, ver, codec, class, type, flags, msg_id, plen

CODEC_JSON = 0
CODEC_COMPACT = 1
CODEC_NAMES = {CODEC_JSON: "json", CODEC_COMPACT: "compact"}

FLAG_SEGMENTED = 0x01

# Hard cap on a serialized envelope (header included). Constrained media may
# cap lower via their LinkProfile; segmentation (below) spans chunks.
MAX_ENVELOPE_BYTES = 65535

# Field caps (mirror the caps of ShugoCore's mobile node layer).
MAX_AGENT_ID = 48
MAX_TOPIC = 64
MAX_REALM = 8
MAX_STR_VALUE = 2048
MAX_PAYLOAD_DEPTH = 4
MAX_PAYLOAD_ITEMS = 64
MAX_SEGMENTS = 255

MESSAGE_TYPES = {
    "announce": 0,
    "heartbeat": 1,
    "ack": 2,
    "tombstone": 3,
    "memory_fact": 4,
    "reinforce": 5,
    "memory_digest": 6,
    "memory_query": 7,
    "memory_response": 8,
    "task_request": 9,
    "task_result": 10,
    "promotion_proposal": 11,
    "bulk_snapshot": 12,
    "audit_ship": 13,
}
TYPE_NAMES = {v: k for k, v in MESSAGE_TYPES.items()}

CLASS_CONTROL = 0
CLASS_MEMORY = 1
CLASS_TASK = 2
CLASS_BULK = 3
CLASS_NAMES = {CLASS_CONTROL: "control", CLASS_MEMORY: "memory",
               CLASS_TASK: "task", CLASS_BULK: "bulk"}

TYPE_CLASS = {
    "announce": CLASS_CONTROL, "heartbeat": CLASS_CONTROL,
    "ack": CLASS_CONTROL, "tombstone": CLASS_CONTROL,
    "memory_fact": CLASS_MEMORY, "reinforce": CLASS_MEMORY,
    "memory_digest": CLASS_MEMORY,
    "memory_query": CLASS_TASK, "memory_response": CLASS_TASK,
    "task_request": CLASS_TASK, "task_result": CLASS_TASK,
    "promotion_proposal": CLASS_TASK,
    "bulk_snapshot": CLASS_BULK, "audit_ship": CLASS_BULK,
}

# Payload field registry for the compact codec. Field 0 ("_blob") is the
# lossless fallback for any key outside the registry.
FIELD_NAMES = {
    "_blob": 0, "fact_id": 1, "content": 2, "kind": 3, "salience": 4,
    "origin": 5, "vector": 6, "target": 7, "amount": 8, "digests": 9,
    "query": 10, "top_k": 11, "results": 12, "task": 13, "status": 14,
    "error": 15, "manifest": 16, "count": 17, "since": 18, "grants": 19,
    "reason": 20, "ttl": 21, "qos": 22, "realm": 23, "port": 24,
}
FIELD_IDS = {v: k for k, v in FIELD_NAMES.items()}

TAG_STR = 0
TAG_UINT = 1
TAG_FLOAT = 2
TAG_JSON = 3


class ProtocolError(ValueError):
    """Raised when a frame or envelope violates the wire contract."""


def new_msg_id() -> int:
    """Random uint32 message id (dedup keyed by (sender, msg_id))."""
    return int.from_bytes(os.urandom(4), "big")


@dataclass
class Envelope:
    """A Shogunet message. ``recipient == "*"`` is a broadcast."""

    msg_id: int
    msg_type: str
    sender: str
    recipient: str = "*"
    topic: str = ""
    realm: str = "*"
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def priority(self) -> int:
        return TYPE_CLASS[self.msg_type]

    def normalized(self) -> "Envelope":
        """Sanitized copy enforcing field caps (applied before encoding)."""
        return Envelope(
            msg_id=int(self.msg_id) & 0xFFFFFFFF,
            msg_type=str(self.msg_type),
            sender=sanitize_text(self.sender, MAX_AGENT_ID),
            recipient=sanitize_text(self.recipient, MAX_AGENT_ID) or "*",
            topic=sanitize_text(self.topic, MAX_TOPIC),
            realm=sanitize_text(self.realm, MAX_REALM) or "*",
            payload=_bound_payload(self.payload),
            created_at=float(self.created_at),
        )


def _bound_payload(value: Any, depth: int = 0) -> Any:
    """Recursively bound a payload: depth/item caps, string caps, no NaN."""
    if depth > MAX_PAYLOAD_DEPTH:
        raise ProtocolError("payload nesting too deep")
    if isinstance(value, dict):
        if len(value) > MAX_PAYLOAD_ITEMS:
            raise ProtocolError("payload has too many keys")
        return {sanitize_text(k, MAX_TOPIC): _bound_payload(v, depth + 1)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_PAYLOAD_ITEMS:
            raise ProtocolError("payload list too long")
        return [_bound_payload(v, depth + 1) for v in value]
    if isinstance(value, str):
        return sanitize_text(value, MAX_STR_VALUE)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else 0.0
    return sanitize_text(value, MAX_STR_VALUE)

# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _encode_value(fid: int, value: Any, out: bytearray) -> None:
    """Append one TLV entry: [fid:1][tag:1][value]."""
    if isinstance(value, bool) or value is None:
        data = json.dumps(value, separators=(",", ":")).encode("utf-8")
        out += struct.pack(">BBH", fid, TAG_JSON, len(data)) + data
    elif isinstance(value, int):
        out += struct.pack(">BBQ", fid, TAG_UINT, int(value) & 0xFFFFFFFFFFFFFFFF)
    elif isinstance(value, float):
        out += struct.pack(">BBd", fid, TAG_FLOAT, float(value))
    elif isinstance(value, str):
        text = sanitize_text(value, MAX_STR_VALUE).encode("utf-8")[:65535]
        out += struct.pack(">BBH", fid, TAG_STR, len(text)) + text
    else:
        data = json.dumps(_bound_payload(value), separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
        data = data.encode("utf-8")[:65535]
        out += struct.pack(">BBH", fid, TAG_JSON, len(data)) + data


def _encode_compact_body(env: Envelope) -> bytes:
    """Fixed fields packed binary, then payload values as TLV entries.

    Keys outside FIELD_NAMES collapse into a single JSON ``_blob`` entry so
    the codec stays lossless no matter what the payload holds.
    """
    buf = bytearray()
    for name, cap in (("sender", MAX_AGENT_ID), ("recipient", MAX_AGENT_ID),
                      ("topic", MAX_TOPIC), ("realm", MAX_REALM)):
        raw = sanitize_text(getattr(env, name), cap).encode("utf-8")[:255]
        buf.append(len(raw))
        buf += raw
    buf += struct.pack(">d", float(env.created_at))
    entries = bytearray()
    blob: Dict[str, Any] = {}
    payload = env.payload if isinstance(env.payload, dict) else {"_value": env.payload}
    for key, val in payload.items():
        fid = FIELD_NAMES.get(str(key))
        if fid is None:
            blob[str(key)] = val
        else:
            _encode_value(fid, val, entries)
    if blob:
        _encode_value(FIELD_NAMES["_blob"], _bound_payload(blob), entries)
    if len(entries) > 65535:
        raise ProtocolError("compact payload too large")
    buf += struct.pack(">H", len(entries))
    buf += entries
    return bytes(buf)


def _encode_json_body(env: Envelope) -> bytes:
    body = {
        "s": sanitize_text(env.sender, MAX_AGENT_ID),
        "r": sanitize_text(env.recipient, MAX_AGENT_ID) or "*",
        "t": sanitize_text(env.topic, MAX_TOPIC),
        "re": sanitize_text(env.realm, MAX_REALM) or "*",
        "ts": float(env.created_at),
        "p": _bound_payload(env.payload),
    }
    try:
        return json.dumps(body, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"payload not JSON-serializable: {exc}") from exc


def encode(envelope: Envelope, codec: int = CODEC_JSON) -> bytes:
    """Serialize an envelope to a wire frame.

    Deterministic per (envelope, codec): the JSON codec sorts keys and the
    compact codec walks payload insertion order, so identical inputs produce
    identical bytes (digest-comparable, test-assertable).
    """
    if not isinstance(envelope, Envelope):
        raise ProtocolError("encode expects an Envelope")
    msg_type = str(envelope.msg_type)
    if msg_type not in MESSAGE_TYPES:
        raise ProtocolError(f"unknown message type '{msg_type}'")
    if codec not in CODEC_NAMES:
        raise ProtocolError(f"unknown codec {codec}")
    env = envelope.normalized()
    if not env.sender:
        raise ProtocolError("sender required")
    body = (_encode_json_body(env) if codec == CODEC_JSON
            else _encode_compact_body(env))
    if len(body) > MAX_ENVELOPE_BYTES - HEADER_LEN:
        raise ProtocolError("envelope exceeds MAX_ENVELOPE_BYTES")
    header = struct.pack(HEADER_FMT, MAGIC, PROTOCOL_VERSION, codec,
                         TYPE_CLASS[msg_type], MESSAGE_TYPES[msg_type], 0,
                         env.msg_id & 0xFFFFFFFF, len(body))
    return header + body

# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def _decode_value(tag: int, data: bytes, offset: int) -> Tuple[Any, int]:
    """Parse one TLV value; returns (value, new_offset)."""
    if tag == TAG_STR:
        (n,) = struct.unpack_from(">H", data, offset)
        offset += 2
        return data[offset:offset + n].decode("utf-8", "replace"), offset + n
    if tag == TAG_UINT:
        (v,) = struct.unpack_from(">Q", data, offset)
        return v, offset + 8
    if tag == TAG_FLOAT:
        (v,) = struct.unpack_from(">d", data, offset)
        return v, offset + 8
    if tag == TAG_JSON:
        (n,) = struct.unpack_from(">H", data, offset)
        offset += 2
        try:
            return json.loads(data[offset:offset + n].decode("utf-8")), offset + n
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("corrupt JSON entry") from exc
    raise ProtocolError(f"unknown value tag {tag}")


def _decode_compact_body(data: bytes) -> Envelope:
    if len(data) < 13:   # 4 length bytes + timestamp
        raise ProtocolError("compact body truncated")
    off = 0
    fields: Dict[str, str] = {}
    for name in ("sender", "recipient", "topic", "realm"):
        n = data[off]
        off += 1
        if off + n > len(data):
            raise ProtocolError("compact body truncated")
        fields[name] = data[off:off + n].decode("utf-8", "replace")
        off += n
    (ts,) = struct.unpack_from(">d", data, off)
    off += 8
    (n_entries,) = struct.unpack_from(">H", data, off)
    off += 2
    end = off + n_entries
    if end > len(data):
        raise ProtocolError("compact entries truncated")
    payload: Dict[str, Any] = {}
    while off < end:
        fid, tag = data[off], data[off + 1]
        off += 2
        value, off = _decode_value(tag, data, off)
        name = FIELD_IDS.get(fid, f"f{fid}")
        if fid == FIELD_NAMES["_blob"] and isinstance(value, dict):
            payload.update(value)
        else:
            payload[name] = value
    return Envelope(
        msg_id=0,            # filled by decode() from the header
        msg_type="",         # filled by decode() from the header
        sender=sanitize_text(fields["sender"], MAX_AGENT_ID),
        recipient=sanitize_text(fields["recipient"], MAX_AGENT_ID) or "*",
        topic=sanitize_text(fields["topic"], MAX_TOPIC),
        realm=sanitize_text(fields["realm"], MAX_REALM) or "*",
        payload=_bound_payload(payload),
        created_at=float(ts),
    )


def _decode_json_body(data: bytes) -> Envelope:
    try:
        body = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("corrupt JSON body") from exc
    if not isinstance(body, dict):
        raise ProtocolError("JSON body must be an object")
    try:
        return Envelope(
            msg_id=0,
            msg_type="",
            sender=sanitize_text(body.get("s", ""), MAX_AGENT_ID),
            recipient=sanitize_text(body.get("r", "*"), MAX_AGENT_ID) or "*",
            topic=sanitize_text(body.get("t", ""), MAX_TOPIC),
            realm=sanitize_text(body.get("re", "*"), MAX_REALM) or "*",
            payload=_bound_payload(body.get("p", {})),
            created_at=float(body.get("ts", time.time())),
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"invalid JSON body fields: {exc}") from exc


def peek_header(frame: bytes) -> Dict[str, Any]:
    """Validate and parse only the fixed header (cheap routing/queueing)."""
    if not isinstance(frame, (bytes, bytearray)) or len(frame) < HEADER_LEN:
        raise ProtocolError("frame shorter than header")
    (magic, version, codec, cls, typ, flags,
     msg_id, plen) = struct.unpack_from(HEADER_FMT, frame, 0)
    if magic != MAGIC:
        raise ProtocolError("bad magic")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")
    if codec not in CODEC_NAMES:
        raise ProtocolError(f"unknown codec {codec}")
    if typ not in TYPE_NAMES:
        raise ProtocolError(f"unknown message type id {typ}")
    type_name = TYPE_NAMES[typ]
    if cls != TYPE_CLASS[type_name]:
        raise ProtocolError("header class does not match message type")
    if plen != len(frame) - HEADER_LEN:
        raise ProtocolError("payload length mismatch")
    return {"version": version, "codec": codec, "class": cls,
            "msg_type": type_name, "flags": flags, "msg_id": msg_id,
            "payload_len": plen, "total_len": len(frame)}


def decode(frame: bytes) -> Envelope:
    """Deserialize a wire frame into a fully validated Envelope."""
    head = peek_header(frame)
    body = bytes(frame[HEADER_LEN:])
    env = (_decode_json_body(body) if head["codec"] == CODEC_JSON
           else _decode_compact_body(body))
    env.msg_id = head["msg_id"]
    env.msg_type = head["msg_type"]
    return env

# ---------------------------------------------------------------------------
# Segmentation (small-MTU media: BLE, LoRa multi-packet bursts)
# ---------------------------------------------------------------------------

_SEG_OVERHEAD = HEADER_LEN + 6   # segment prefix: idx(1) count(1) total(4)


def segment_frame(frame: bytes, chunk_size: int) -> List[bytes]:
    """Split a frame into wire chunks for small-MTU media.

    Every chunk is a self-contained frame sharing the original msg_id and
    msg type, with FLAG_SEGMENTED set and a segment prefix
    ``(idx, count, total_payload_len)`` so reassembly can be validated.
    Frames that already fit are returned unchanged (unsegmented).
    """
    head = peek_header(frame)   # validates before we touch anything
    if chunk_size < _SEG_OVERHEAD + 1:
        raise ProtocolError("chunk_size too small for segmentation")
    if len(frame) <= chunk_size:
        return [bytes(frame)]
    payload = bytes(frame[HEADER_LEN:])
    per_chunk = chunk_size - _SEG_OVERHEAD
    count = (len(payload) + per_chunk - 1) // per_chunk
    if count > MAX_SEGMENTS:
        raise ProtocolError("frame needs too many segments")
    chunks: List[bytes] = []
    for idx in range(count):
        prefix = struct.pack(">BBI", idx, count, len(payload))
        piece = payload[idx * per_chunk:(idx + 1) * per_chunk]
        header = struct.pack(HEADER_FMT, MAGIC, PROTOCOL_VERSION,
                             head["codec"], head["class"],
                             MESSAGE_TYPES[head["msg_type"]],
                             head["flags"] | FLAG_SEGMENTED,
                             head["msg_id"], len(prefix) + len(piece))
        chunks.append(header + prefix + piece)
    return chunks


def defragment(chunks: Iterable[bytes]) -> Optional[bytes]:
    """Reassemble segmented frames.

    Returns the frame once every piece has arrived, ``None`` while pieces
    are missing. Duplicates are tolerated; mixed identities or inconsistent
    metadata raise ``ProtocolError``.
    """
    frames = [bytes(c) for c in chunks]
    if not frames:
        return None
    heads = [peek_header(f) for f in frames]
    for frame, head in zip(frames, heads):
        if not head["flags"] & FLAG_SEGMENTED:
            return frame   # a complete frame arrived among the chunks
    first = heads[0]
    key = (first["msg_type"], first["msg_id"])
    total: Optional[int] = None
    plen_total: Optional[int] = None
    pieces: Dict[int, bytes] = {}
    for frame, head in zip(frames, heads):
        if (head["msg_type"], head["msg_id"]) != key:
            raise ProtocolError("defragment input spans multiple frames")
        if len(frame) < _SEG_OVERHEAD:
            raise ProtocolError("segment prefix truncated")
        idx, count, plen = struct.unpack_from(">BBI", frame, HEADER_LEN)
        if total is None:
            total, plen_total = count, plen
        if count != total or plen != plen_total:
            raise ProtocolError("inconsistent segment metadata")
        if idx >= total:
            raise ProtocolError("segment index out of range")
        pieces.setdefault(idx, frame[_SEG_OVERHEAD:])
    if total is None or plen_total is None:
        return None
    if len(pieces) < total:
        return None
    payload = b"".join(pieces[i] for i in range(total))
    if len(payload) != plen_total:
        raise ProtocolError("reassembled payload length mismatch")
    header = struct.pack(HEADER_FMT, MAGIC, first["version"], first["codec"],
                         first["class"], MESSAGE_TYPES[first["msg_type"]],
                         first["flags"] & ~FLAG_SEGMENTED,
                         first["msg_id"], plen_total)
    return header + payload


class SegmentAssembler:
    """Stateful reassembly for inbound frames on small-MTU media.

    Feed every inbound frame through ``add``: complete (unsegmented) frames
    pass through untouched; segmented frames accumulate per
    ``(msg_type, msg_id)`` and are emitted when complete. Bounded: at most
    ``max_pending`` in-flight frames, entries older than ``ttl_s`` evicted.
    """

    def __init__(self, max_pending: int = 16, ttl_s: float = 60.0):
        self._pending: Dict[Tuple[str, int], Tuple[float, List[bytes]]] = {}
        self._order: List[Tuple[str, int]] = []
        self._max_pending = max(1, int(max_pending))
        self._ttl = max(1.0, float(ttl_s))
        self._lock = threading.Lock()

    def add(self, frame: bytes) -> Optional[bytes]:
        head = peek_header(frame)
        if not head["flags"] & FLAG_SEGMENTED:
            return bytes(frame)
        key = (head["msg_type"], head["msg_id"])
        now = time.monotonic()
        with self._lock:
            if key not in self._pending:
                self._pending[key] = (now, [])
                self._order.append(key)
                while len(self._order) > self._max_pending:
                    oldest = self._order.pop(0)
                    self._pending.pop(oldest, None)
            entry = self._pending[key]
            self._pending[key] = (entry[0], entry[1] + [bytes(frame)])
            chunks = self._pending[key][1]
        assembled = defragment(chunks)
        if assembled is not None:
            with self._lock:
                self._pending.pop(key, None)
                if key in self._order:
                    self._order.remove(key)
            return assembled
        with self._lock:   # prune stale partials
            for stale in list(self._order):
                if now - self._pending[stale][0] > self._ttl:
                    self._pending.pop(stale, None)
                    self._order.remove(stale)
        return None




