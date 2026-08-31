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


class RuntimeUnavailable(KilixOllamaError):
    """The private mediated runtime is absent or not eligible for an action."""


class ConversationError(KilixOllamaError):
    """Private conversation state is missing, malformed, or unsafe."""


class ToolError(KilixOllamaError):
    """A bounded tool request was invalid, denied, or failed safely."""


class TerminalError(KilixOllamaError):
    """The terminal UI could not establish or restore a safe local session."""


class AcceleratorQueueFull(KilixOllamaError):
    """The bounded shared-accelerator queue has no admission slot."""


class AcceleratorRequestCancelled(KilixOllamaError):
    """A queued accelerator request was cancelled before it ran."""


class AcceleratorRequestTimeout(KilixOllamaError):
    """A queued accelerator request exceeded its wait deadline."""


class AcceleratorUnavailable(KilixOllamaError):
    """The shared accelerator arbiter is no longer accepting work."""


class SupervisorError(KilixOllamaError):
    """The fixed local runtime could not be supervised safely."""


class SynthesisError(KilixOllamaError):
    """A local speech synthesis request failed safely."""


class SynthesisCancelled(SynthesisError):
    """A local speech synthesis request was cancelled."""


__all__ = [
    "AcceleratorQueueFull",
    "AcceleratorRequestCancelled",
    "AcceleratorRequestTimeout",
    "AcceleratorUnavailable",
    "BackendUnavailable",
    "ConversationError",
    "InstallError",
    "KilixOllamaError",
    "PolicyDenied",
    "RuntimeContractError",
    "RuntimeUnavailable",
    "SupervisorError",
    "SynthesisCancelled",
    "SynthesisError",
    "TerminalError",
    "ToolError",
    "UpstreamProtocolError",
]
