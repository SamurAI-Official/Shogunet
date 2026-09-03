"""Shogunet version (SemVer)."""

__version__ = "0.4.0"
VERSION = __version__

# Wire-format version of the frame protocol (mirrors protocol.PROTOCOL_VERSION;
# kept here so version checks never need to import the codec stack).
PROTOCOL_VERSION = 1


def is_compatible(other: str) -> bool:
    """True when ``other`` (a SemVer string) can talk to this build.

    Rule for the 0.x series: the minor version is the wire dialect -- same
    major and same minor is compatible, patch differences are always fine,
    any other minor (older *or* newer) is refused: a 0.3.x client cannot join
    a 0.4.x host, and a 0.5.x client speaks a dialect a 0.4.x host has never
    seen. Anything unparsable is refused.
    """
    try:
        parts = [int(p) for p in str(other).strip().split(".")[:3]]
    except (TypeError, ValueError):
        return False
    while len(parts) < 3:
        parts.append(0)
    try:
        ours = [int(p) for p in __version__.split(".")[:3]]
    except (TypeError, ValueError):    # pragma: no cover - self-version is fixed
        return False
    return parts[0] == ours[0] and parts[1] == ours[1]
