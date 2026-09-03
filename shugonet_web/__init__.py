"""shugonet_web: compiled static assets for the Shogunet fleet dashboard.

The SPA source lives in ``dashboard/`` (TypeScript + SolidJS + Vite); the
build output is committed/placed here (``shugonet_web/static``) so a wheel
of Shogunet ships the operator console with zero JavaScript toolchain
requirements at install time. Build with::

    cd dashboard && npm ci && npm run build
"""

import os


def static_dir() -> str:
    """Absolute path of the compiled dashboard assets."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
