"""Vendored-bridge parity: ShugoCore's shugonet_bridge.py must carry the
same action-type sets, fallback severities and namespace as Shogunet's
canonical shugocore_adapter.py. Skipped when no ShugoCore checkout exists.
"""

import importlib.util
import os
import unittest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bridge_path():
    candidates = [
        os.environ.get("SHUGOCORE_PATH"),
        os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "..", "Shugocore", "shugonet_bridge.py"),
        os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "..", "ShugoCore", "shugonet_bridge.py"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    return None


class TestBridgeSync(unittest.TestCase):

    def setUp(self):
        path = bridge_path()
        if not path:
            self.skipTest("ShugoCore checkout (shugonet_bridge.py) not found")
        spec = importlib.util.spec_from_file_location(
            "shugonet_bridge_vendored", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.bridge = module
        import shugocore_adapter
        self.canonical = shugocore_adapter

    def test_side_effecting_action_types_match(self):
        self.assertEqual(self.canonical.NETWORK_ACTION_TYPES,
                         self.bridge.NETWORK_ACTION_TYPES)

    def test_read_action_types_match(self):
        self.assertEqual(self.canonical.NETWORK_READ_ACTION_TYPES,
                         self.bridge.NETWORK_READ_ACTION_TYPES)

    def test_fallback_severities_match(self):
        self.assertEqual(self.canonical.NETWORK_FALLBACK_SEVERITIES,
                         self.bridge.NETWORK_FALLBACK_SEVERITIES)

    def test_namespace_matches(self):
        self.assertEqual(self.canonical._NAMESPACE, self.bridge._NAMESPACE)

    def test_handler_contract_surface_matches(self):
        # The vendored bridge must expose the same registration seam.
        for symbol in ("ShugonetExecutionHandler", "register_network_handlers",
                       "attach_network_fallbacks"):
            self.assertTrue(hasattr(self.bridge, symbol), symbol)
            self.assertTrue(hasattr(self.canonical, symbol), symbol)


if __name__ == "__main__":
    unittest.main()
