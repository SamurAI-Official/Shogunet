"""Subprocess driver for ``test_shugocore_integration``.

Runs against a REAL ShugoCore checkout and prints a JSON report on stdout.
Kept in its own module because module shadowing and ``sys.path`` order are
process-global: this has to execute out-of-process to mean anything.

Not collected by the test runner (leading underscore + no Test prefix).
"""

import json
import os
import sys


def main():
    shogunet, shugocore = sys.argv[1], sys.argv[2]
    report = {}

    # -- 1. ShugoCore's own modules first, as in a real ShugoCore process ----
    sys.path.insert(0, shugocore)
    import policy
    import execution_layer
    from execution_layer import ExecutionLayer
    from fallbacks import FallbackController

    # -- 2. Simulate a Shogunet-hosted process ------------------------------
    sys.path.insert(0, shogunet)
    import importlib.util

    def _load_by_path(name):
        """Load Shogunet's module explicitly -- a bare ``import security``
        would hand back ShugoCore's copy, which step 1 already cached."""
        spec = importlib.util.spec_from_file_location(
            "_sgon_" + name, os.path.join(shogunet, name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    sgon_security = _load_by_path("security")
    sgon_audit = _load_by_path("audit")
    import shugocore_adapter
    import shugocore_bridge
    # Shogunet's twins now sit in sys.modules under the plain names -- this is
    # exactly what a bare ``import security`` in the bridge used to pick up.
    sys.modules["security"] = sgon_security
    sys.modules["audit"] = sgon_audit

    class Agent:
        def send(self, peer, topic, payload):
            return {"status": "success"}

        def query(self, query, peers=None, top_k=None):
            return []

        def sync(self, peer=None, since=None):
            return {"status": "success"}

        def list_agents(self):
            return []

        def status(self):
            return {"mode": "normal"}

    shugocore_abs = os.path.abspath(shugocore)
    all_types = sorted(shugocore_adapter.NETWORK_ACTION_TYPES
                       | shugocore_adapter.NETWORK_READ_ACTION_TYPES)

    # -- 3. register_network_handlers against a REAL ExecutionLayer ----------
    layer = ExecutionLayer()
    try:
        registered = shugocore_adapter.register_network_handlers(layer, Agent())
        report["register_raised"] = None
    except Exception as exc:
        registered = []
        report["register_raised"] = "%s: %s" % (type(exc).__name__, exc)
    report["registered"] = sorted(registered or [])
    report["handlers_installed"] = sorted(layer._handlers)
    report["all_types"] = all_types
    report["all_registered"] = sorted(registered or []) == all_types

    # The sets execution_layer validates against must have learned the
    # spatial/NRR types, and must still be the objects it holds.
    report["policy_side_effecting_has_nrr"] = (
        "network_nrr_render" in policy.NETWORK_ACTION_TYPES
        and "network_spatial_observe" in policy.NETWORK_ACTION_TYPES)
    report["policy_read_has_spatial"] = (
        "network_spatial_query" in policy.NETWORK_READ_ACTION_TYPES
        and "network_fleet_map" in policy.NETWORK_READ_ACTION_TYPES)
    report["policy_sets_in_place"] = (
        policy.NETWORK_ACTION_TYPES is execution_layer.NETWORK_ACTION_TYPES
        and policy.NETWORK_READ_ACTION_TYPES
        is execution_layer.NETWORK_READ_ACTION_TYPES)
    report["policy_module_is_shugocore"] = os.path.dirname(
        os.path.abspath(policy.__file__)) == shugocore_abs

    # -- 4. attach_network_fallbacks (previously skipped by the raise) ------
    controller = FallbackController(governor=object())
    shugocore_adapter.attach_network_fallbacks(controller)
    report["fallbacks_merged"] = all(
        kind in controller.severities
        for kind in shugocore_adapter.NETWORK_FALLBACK_SEVERITIES)

    # -- 5. bridge binds the REAL ShugoCore modules, not Shogunet's ---------
    report["configure"] = shugocore_bridge.configure(shugocore)
    report["shugocore_loaded"] = shugocore_bridge.shugocore_loaded()
    loaded = shugocore_bridge._LOADED
    origins = {}
    for name in ("security", "audit"):
        module = loaded.get(name)
        origins[name] = os.path.dirname(
            os.path.abspath(module.__file__)) if module is not None else None
    report["security_from"] = origins["security"]
    report["audit_from"] = origins["audit"]
    report["bridge_bound_to_checkout"] = (
        origins["security"] == shugocore_abs
        and origins["audit"] == shugocore_abs)
    report["sys_modules_restored"] = (sys.modules["security"] is sgon_security
                                      and sys.modules["audit"] is sgon_audit)
    # ShugoCore strips zero-width characters; Shogunet's local copy leaves
    # "a b". Proves real delegation, not merely a truthy shugocore_loaded().
    # (chr() so the zero-width char cannot be mangled by an editor/encoding.)
    zero_width = "a" + chr(0x200B) + "b"
    report["bridge_uses_shugocore_primitive"] = (
        shugocore_bridge.sanitize_text(zero_width) == "ab")
    report["local_would_differ"] = (
        sgon_security.sanitize_text(zero_width) != "ab")

    # -- 6. validate_url honours the 1.30 signature + host allowlist ---------
    # Report *correctness*, not the raw verdict: a rejected URL must read True.
    url_ok = shugocore_bridge.validate_url
    report["url_allowed_host"] = url_ok("http://127.0.0.1:9001", ["127.0.0.1"])
    report["url_blocked_host"] = not url_ok("http://evil.example", ["127.0.0.1"])
    report["url_creds_rejected"] = not url_ok("http://u:p@127.0.0.1:9001",
                                             ["127.0.0.1"])
    report["url_bad_scheme"] = not url_ok("file:///etc/passwd", ["127.0.0.1"])
    report["url_garbage_is_false"] = url_ok("not a url", ["127.0.0.1"]) is False

    print(json.dumps(report))


if __name__ == "__main__":
    main()
