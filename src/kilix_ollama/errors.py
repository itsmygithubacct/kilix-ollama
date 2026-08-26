"""Stable error categories for the pre-contract development spine."""


class KilixOllamaError(RuntimeError):
    """Base class for a failure safe to present without a traceback."""


class BackendUnavailable(KilixOllamaError):
    """The requested compute backend is absent or not qualified."""


class PolicyDenied(KilixOllamaError):
    """A request crossed the local mediation policy."""


class RuntimeContractError(KilixOllamaError):
    """A runtime layout or launch specification is unsafe."""


class InstallError(KilixOllamaError):
    """An optional artifact could not be installed or verified safely."""


class UpstreamProtocolError(KilixOllamaError):
    """The upstream service returned malformed or incompatible data."""


__all__ = [
    "BackendUnavailable",
    "InstallError",
    "KilixOllamaError",
    "PolicyDenied",
    "RuntimeContractError",
    "UpstreamProtocolError",
]
