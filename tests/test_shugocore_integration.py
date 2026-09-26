"""Shogunet <-> a real ShugoCore checkout: end-to-end integration tests.

``test_shugocore.py`` is hermetic -- it drives duck-typed doubles, which is
exactly why the real integration stayed broken: a double cannot reproduce
``ExecutionLayer.register_handler``'s ``ValueError``, and no in-process
double can reproduce ``sys.modules`` shadowing.

These tests run a real ShugoCore in a **subprocess** (module shadowing and
``sys.path`` order are process-global, so an in-process test proves nothing)
and assert the documented sequence works::

    register_network_handlers(engine.execution_layer, runtime)
    attach_network_fallbacks(engine.fallback_controller)

Skipped automatically when no ShugoCore checkout is available.
"""

import json
import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "_shugocore_integration_driver.py")


def shugocore_dir():
    """Locate a ShugoCore checkout (env var first, then sibling dirs)."""
    candidates = [os.environ.get("SHUGOCORE_PATH"),
                  os.path.join(REPO, "..", "Shugocore"),
                  os.path.join(REPO, "..", "ShugoCore")]
    for path in candidates:
        if path and os.path.isdir(path) \
                and os.path.isfile(os.path.join(path, "execution_layer.py")):
            return os.path.abspath(path)
    return None


def shugocore_python(checkout):
    """Prefer ShugoCore's own interpreter -- it has ShugoCore's deps."""
    venv = os.path.join(checkout, ".venv", "bin", "python")
    return venv if os.path.isfile(venv) else sys.executable


class TestShugocoreIntegration(unittest.TestCase):
    """The real integration, exercised against a real ShugoCore."""

    @classmethod
    def setUpClass(cls):
        cls.checkout = shugocore_dir()
        if not cls.checkout:
            raise unittest.SkipTest("no ShugoCore checkout found")
        proc = subprocess.run(
            [shugocore_python(cls.checkout), DRIVER, REPO, cls.checkout],
            capture_output=True, text=True, timeout=180, cwd=REPO)
        cls.stderr = proc.stderr
        if proc.returncode != 0:
            if "ModuleNotFoundError" in proc.stderr \
                    or "ImportError" in proc.stderr:
                raise unittest.SkipTest("ShugoCore deps unavailable: "
                                        + proc.stderr.strip()[-400:])
            raise AssertionError("driver failed (%d):\n%s"
                                 % (proc.returncode, proc.stderr[-3000:]))
        try:
            cls.report = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            raise AssertionError("driver produced no report:\n%s\n---\n%s"
                                 % (proc.stdout[-2000:], proc.stderr[-2000:]))

    def assertReported(self, key, detail=""):
        self.assertIs(self.report.get(key), True,
                      "%s -> %r %s\nstderr: %s"
                      % (key, self.report.get(key), detail,
                         self.stderr[-800:]))

    def test_registration_does_not_raise(self):
        self.assertIsNone(self.report.get("register_raised"),
                          "register_network_handlers raised: %s"
                          % self.report.get("register_raised"))

    def test_every_action_type_registers(self):
        self.assertReported("all_registered",
                            "\nexpected: %s\n     got: %s"
                            % (self.report.get("all_types"),
                               self.report.get("registered")))
        self.assertEqual(self.report.get("handlers_installed"),
                         self.report.get("all_types"))

    def test_execution_layer_policy_sets_are_patched(self):
        # register_handler validates against these two, not KNOWN_ACTION_TYPES.
        self.assertReported("policy_side_effecting_has_nrr")
        self.assertReported("policy_read_has_spatial")

    def test_policy_sets_are_mutated_in_place(self):
        # execution_layer holds references to these objects; a rebind would
        # leave it validating a stale set.
        self.assertReported("policy_sets_in_place")

    def test_shugocore_policy_was_the_one_patched(self):
        self.assertReported("policy_module_is_shugocore")

    def test_fallbacks_attach_after_registration(self):
        self.assertReported("fallbacks_merged")

    def test_bridge_binds_real_shugocore_modules(self):
        self.assertReported("configure")
        self.assertReported("shugocore_loaded")
        self.assertReported("bridge_bound_to_checkout",
                            "\nsecurity from %s\naudit    from %s"
                            % (self.report.get("security_from"),
                               self.report.get("audit_from")))

    def test_bridge_leaves_shogunet_modules_alone(self):
        self.assertReported("sys_modules_restored")

    def test_bridge_delegates_to_shugocore_primitive(self):
        # Guards the test's own discriminating power: the two implementations
        # must actually differ here, or the delegation check is vacuous.
        self.assertReported("local_would_differ")
        self.assertReported("bridge_uses_shugocore_primitive")

    def test_validate_url_enforces_allowlist(self):
        self.assertReported("url_allowed_host")
        self.assertReported("url_blocked_host",
                            "a non-allowlisted host must be rejected")
        self.assertReported("url_creds_rejected")
        self.assertReported("url_bad_scheme")
        self.assertReported("url_garbage_is_false")


if __name__ == "__main__":
    unittest.main()