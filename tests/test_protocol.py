"""Shogunet wire protocol tests: header, codecs, validation, segmentation."""

import struct
import unittest

import protocol
from protocol import (CODEC_COMPACT, CODEC_JSON, Envelope, HEADER_LEN,
                      ProtocolError, SegmentAssembler, decode, encode,
                      segment_frame)


def make_env(**overrides):
    base = dict(
        msg_id=12345,
        msg_type="reinforce",
        sender="agent-a",
        recipient="agent-b",
        topic="/shugunet/agent-a/memory",
        realm="phys",
        payload={"target": "agent-b", "amount": 2, "fact_id": 7},
    )
    base.update(overrides)
    return Envelope(**base)


class TestCodecRoundTrip(unittest.TestCase):

    def _assert_roundtrip(self, codec):
        frame = encode(make_env(), codec)
        (plen,) = struct.unpack_from(">I", frame, 12)
        self.assertEqual(len(frame), HEADER_LEN + plen)
        out = decode(frame)
        self.assertEqual(out.msg_id, 12345)
        self.assertEqual(out.msg_type, "reinforce")
        self.assertEqual(out.sender, "agent-a")
        self.assertEqual(out.recipient, "agent-b")
        self.assertEqual(out.topic, "/shugunet/agent-a/memory")
        self.assertEqual(out.realm, "phys")
        self.assertEqual(out.payload,
                         {"target": "agent-b", "amount": 2, "fact_id": 7})
        self.assertEqual(out.priority(), protocol.CLASS_MEMORY)

    def test_json_roundtrip(self):
        self._assert_roundtrip(CODEC_JSON)

    def test_compact_roundtrip(self):
        self._assert_roundtrip(CODEC_COMPACT)

    def test_compact_smaller_than_json(self):
        json_frame = encode(make_env(), CODEC_JSON)
        compact = encode(make_env(), CODEC_COMPACT)
        self.assertLess(len(compact), len(json_frame))

    def test_blob_fallback_lossless(self):
        e = make_env(payload={"custom_key": "custom-value", "amount": 1})
        out = decode(encode(e, CODEC_COMPACT))
        self.assertEqual(out.payload["custom_key"], "custom-value")
        self.assertEqual(out.payload["amount"], 1)

    def test_deterministic_encoding(self):
        for codec in (CODEC_JSON, CODEC_COMPACT):
            first = encode(make_env(created_at=1000.0), codec)
            second = encode(make_env(created_at=1000.0), codec)
            self.assertEqual(first, second)

    def test_every_message_type_roundtrips(self):
        for msg_type in protocol.MESSAGE_TYPES:
            e = make_env(msg_type=msg_type, payload={"count": 1})
            for codec in (CODEC_JSON, CODEC_COMPACT):
                out = decode(encode(e, codec))
                self.assertEqual(out.msg_type, msg_type)
                self.assertEqual(out.payload, {"count": 1})

class TestValidation(unittest.TestCase):

    def _patched(self, offset, value):
        frame = bytearray(encode(make_env()))
        frame[offset] = value
        return bytes(frame)

    def test_bad_magic(self):
        frame = bytearray(encode(make_env()))
        frame[0] = 0x58
        with self.assertRaises(ProtocolError):
            decode(bytes(frame))

    def test_bad_version(self):
        with self.assertRaises(ProtocolError):
            decode(self._patched(3, 99))

    def test_bad_codec(self):
        with self.assertRaises(ProtocolError):
            decode(self._patched(4, 7))

    def test_class_mismatch(self):
        # reinforce is CLASS_MEMORY(1); header claiming CLASS_CONTROL is refused
        with self.assertRaises(ProtocolError):
            decode(self._patched(5, protocol.CLASS_CONTROL))

    def test_unknown_type_id(self):
        with self.assertRaises(ProtocolError):
            decode(self._patched(6, 200))

    def test_payload_len_mismatch(self):
        with self.assertRaises(ProtocolError):
            decode(encode(make_env())[:-1])

    def test_short_frame(self):
        with self.assertRaises(ProtocolError):
            decode(b"SGN")

    def test_unknown_type_rejected_at_encode(self):
        with self.assertRaises(ProtocolError):
            encode(make_env(msg_type="meme"))

    def test_empty_sender_rejected(self):
        with self.assertRaises(ProtocolError):
            encode(make_env(sender=""))

    def test_oversize_rejected(self):
        # Individual strings cap at MAX_STR_VALUE, so true oversize needs many
        # capped values: 40 x ~2 KB >> the 64 KB frame ceiling.
        payload = {f"k{i}": "x" * 2048 for i in range(40)}
        with self.assertRaises(ProtocolError):
            encode(make_env(msg_type="bulk_snapshot", payload=payload))

    def test_depth_cap(self):
        deep = {"a": {"b": {"c": {"d": {"e": 1}}}}}
        with self.assertRaises(ProtocolError):
            encode(make_env(payload=deep))

    def test_nan_coerced(self):
        out = decode(encode(make_env(payload={"amount": float("nan")})))
        self.assertEqual(out.payload["amount"], 0.0)


class TestSanitization(unittest.TestCase):

    def test_control_chars_never_survive(self):
        out = decode(encode(make_env(sender="agent\x01a\nb")))
        self.assertNotIn("\x01", out.sender)
        self.assertTrue(all(ch.isprintable() for ch in out.sender))

    def test_field_caps(self):
        out = decode(encode(make_env(sender="a" * 200, topic="t" * 200,
                                     realm="r" * 99)))
        self.assertEqual(len(out.sender), protocol.MAX_AGENT_ID)
        self.assertEqual(len(out.topic), protocol.MAX_TOPIC)
        self.assertEqual(len(out.realm), protocol.MAX_REALM)

    def test_string_payload_value_capped(self):
        out = decode(encode(make_env(payload={"content": "y" * 9999})))
        self.assertEqual(len(out.payload["content"]), protocol.MAX_STR_VALUE)


class TestSegmentation(unittest.TestCase):

    def setUp(self):
        self.big = encode(make_env(msg_type="memory_fact",
                                   payload={"content": "z" * 400}),
                          CODEC_JSON)

    def test_small_frame_passthrough(self):
        small = encode(make_env())
        self.assertEqual(segment_frame(small, 4096), [small])

    def test_chunk_too_small(self):
        with self.assertRaises(ProtocolError):
            segment_frame(self.big, 10)

    def test_roundtrip_via_assembler(self):
        chunks = segment_frame(self.big, 48)
        self.assertGreater(len(chunks), 1)
        assembler = SegmentAssembler()
        emitted = [out for out in (assembler.add(c) for c in chunks) if out]
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0], self.big)
        self.assertEqual(decode(emitted[0]).payload["content"], "z" * 400)

    def test_incomplete_returns_none(self):
        chunks = segment_frame(self.big, 48)
        assembler = SegmentAssembler()
        self.assertIsNone(assembler.add(chunks[0]))
        self.assertIsNone(assembler.add(chunks[1]))

    def test_duplicate_piece_tolerated(self):
        chunks = segment_frame(self.big, 48)
        assembler = SegmentAssembler()
        assembler.add(chunks[0])
        assembler.add(chunks[0])   # duplicate
        out = None
        for chunk in chunks[1:]:
            got = assembler.add(chunk)
            if got is not None:
                out = got
        self.assertEqual(out, self.big)


if __name__ == "__main__":
    unittest.main()

