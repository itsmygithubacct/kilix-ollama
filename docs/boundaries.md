# Isolation and safety boundaries

Reference for [Kilix Ollama](../README.md).

The implementation establishes these boundaries:

- a deny-by-default API route and header policy;
- a private per-user XDG layout and same-UID peer requirement;
- CPU/Vulkan discovery with qualification-gated, fail-closed selection;
- immutable service descriptions and a hash-admitted process-group supervisor
  for the private Unix-listener candidate;
- a bounded live Unix-socket proxy that applies policy before forwarding and
  never falls back to a cloud endpoint;
- strict bounded NDJSON assembly and stateful terminal-control stripping across
  arbitrary streaming boundaries;
- a local-only full-screen TUI spike with bounded fragmented input parsing,
  Kitty-keyboard/bracketed-paste negotiation, control-free responsive frames,
  and terminal restoration on normal exit or catchable signals;
- owner-private, schema-checked, atomic conversation persistence and sanitized
  Markdown export;
- explicitly selected PNG/JPEG chat images copied into path-free,
  digest-addressed owner-private snapshots with byte, pixel, count, and tamper
  limits;
- four bounded read-only tools that require both session opt-in and per-call
  approval, with no shell, subprocess, network, or write tool;
- one shared, bounded accelerator arbiter that finishes an admitted LLM turn
  before prioritizing queued speech at utterance boundaries;
- exact-artifact one-shot and persistent Vulkan speech runners with bounded
  framing, cancellation, timeout, and process-group cleanup; and
- an opt-in, upstream-only Pocket TTS source catalog and atomic local
  conversion boundary with no bundled model or voice artifacts.

Ordinary host loopback is not an isolation boundary on a multi-user machine.
The Unix-listener delta has been locally verified, but neither transport has an
accepted release admission. Owner-isolated TCP supervision remains
unimplemented. The supervisor starts from a cleared, fixed environment and
never inherits caller-supplied
`OLLAMA_*`, proxy, preload, registry, origin, model-root, or library-path
values.
