# Kilix Ollama

Kilix Ollama is a local lifecycle, compatibility, and mediation layer for an
Ollama runtime managed by Plebian OS. This tree is a published work-branch
development spine for the 0.2.1 line, not a release-admitted artifact. Its CLI
diagnostics perform read-only discovery and emit provisional contracts. Its
library includes the candidate runtime
supervisor, live private mediation proxy, shared accelerator arbiter, and
one-shot and persistent Vulkan TTS runners. The CLI now provides an exact
artifact-free model catalog, one-off prompts, persistent chats, conversation
management, and a disabled-by-default read-only tool loop through that private
mediation boundary. It does not start the daemon. A separate explicit
optional-TTS command can download hash-pinned inputs from their immutable
upstream revision and convert them locally.

**Status:** development work, version `0.2.1.dev0`, on the default branch
`work/0.2.1-f105-bootstrap`. Plebian-OS does not install or launch this app:
Plebian-OS 0.2.3 runs its own loopback Ollama service and provisions the
runtime with its own `plebian-os-install-kilix-ollama-runtime`. Kilix's
Windows (WSL) app catalog lists it as **Ollama** at a pinned commit, and
Plebian-OS's opt-in `plebian-os-install-kilix-vulkan-tts` builds a
byte-identical copy of [native/kilix_tts_worker.cpp](native/kilix_tts_worker.cpp).

## What it does

- Discovers the local runtime and CPU/Vulkan support and reports provisional
  JSON records (`doctor`, `backend`, `policy`, `service-spec`, ...).
- Mediates every request through a private same-user Unix socket with a
  deny-by-default route and header policy, never a cloud endpoint.
- Lists an exact three-entry model catalog, inspects installed models, runs
  one-off prompts and persistent chats with fork, regenerate and model switch.
- Stores conversations privately and atomically, and exports them as
  sanitized Markdown; attaches explicitly selected PNG/JPEG images.
- Offers four read-only tools, off by default and approved per call.
- Downloads and converts an optional Pocket TTS model only on explicit request.

## Quick start

### Development commands

Run the deterministic test suite:

```sh
make check
```

Inspect the current machine without changing it:

```sh
./kilix-ollama doctor
./kilix-ollama backend
./kilix-ollama policy --json
./kilix-ollama service-contract --json
./kilix-ollama service-spec --transport unix --json
./kilix-ollama runtime-profiles --json
./kilix-ollama co-residency --json
./kilix-ollama tts-models --json
./kilix-ollama tts-status --verify --json
```

The JSON names end in `v0-unfrozen` by design. They are development records,
not accepted public schemas. The current Ollama
`0.32.14+plebian.security-license-r3` reference is likewise only a source
candidate; it is reported as unqualified even when its metadata
parses successfully.

## Usage

| Command | What it does |
|---|---|
| `kilix-ollama tui` | Local-only full-screen development interface |
| `kilix-ollama library [--installed]`, `models`, `ps`, `show MODEL` | Catalog, installed and running models |
| `kilix-ollama ask MODEL [--file F] [--output F] [--json] PROMPT` | One prompt, no stored conversation |
| `kilix-ollama chat MODEL [--message M] [--attach IMG] [--tools]` | Persistent chat |
| `kilix-ollama conversations list\|show\|delete\|export\|fork` | Manage the private conversation store |
| `kilix-ollama pull MODEL` | Exact install through the shared authority (currently fails closed) |
| `kilix-ollama tools --json` | Print the read-only tool registry |
| `kilix-ollama tts-models`, `tts-install`, `tts-status`, `tts-remove` | Optional local TTS conversion |
| `kilix-ollama doctor`, `backend`, `policy`, `service-contract`, `service-spec`, `runtime-profiles`, `co-residency` | Read-only diagnostics and contracts |

Prompt and chat commands need `--development-runtime` to admit the exact local
runtime candidate. Details and examples are in
[docs/product-commands.md](docs/product-commands.md). `make install
PREFIX=/usr/local` installs the launcher and the `kilix_ollama` package.

## How it works

```
kilix-ollama CLI / TUI
      |  same-UID peer check, private per-user XDG paths
      v
mediation proxy  -- deny-by-default routes and headers, bounded NDJSON,
      |             terminal-control stripping, no cloud fallback
      v
Ollama runtime   -- fixed root-owned launcher
                    /usr/lib/plebian-os/ollama/ollama-current
                    (supervised from a cleared, fixed environment)
```

The library holds the candidate runtime supervisor, the live mediation proxy,
a shared accelerator arbiter that finishes an LLM turn before queued speech,
and one-shot and persistent Vulkan speech runners. Backend selection stays on
CPU until measured evidence qualifies an exact Vulkan profile. The full list
of boundaries is in [docs/boundaries.md](docs/boundaries.md).

## Current limits

The conversation and attachment schemas are implemented but remain v0 and
unaccepted. Generic document attachments, interactive/TUI attachment selection,
image placement, owned-snapshot garbage collection, and attachment-inclusive
exports are not implemented yet. There is
no public daemon entry point, release-admitted full-screen terminal UI, avatar UI,
emotional-state or lip-motion integration, shared-authority model transfer, or
release-admitted product voice. The terminal product, supervisor, live
mediation server, shared arbiter, and synthesis runners are unfrozen candidates
with deterministic local coverage. Native Pocket output currently fails the
local application-quality gate and remains research-only. Model and weight
artifacts remain deliberately absent from the repository. The optional TTS
catalog and installer are a local conversion contract, not model or runtime
admission. Python and the standard library remain a provisional control-plane
choice; downloads reuse Kilix Content, and conversion plus native inference
require separately packaged, pinned closures.

## Documentation

- [docs/boundaries.md](docs/boundaries.md) — the isolation and safety boundaries the implementation establishes
- [docs/product-commands.md](docs/product-commands.md) — TUI, catalog, prompts, chats, conversations, attachments, tools, runtime installation
- [docs/tts-conversion.md](docs/tts-conversion.md) — optional Pocket TTS download and offline conversion
- [docs/vulkan-closure.md](docs/vulkan-closure.md) — Vulkan discovery, tuning candidates and co-residency measurements
- [native/README.md](native/README.md) — the native TTS worker source of record
- [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) — upstream notices
- [CHANGELOG.md](CHANGELOG.md) — unreleased changes

## Related repositories

- [plebian-os](https://github.com/itsmygithubacct/plebian-os#readme) — owns the Ollama runtime installer, the private-listener patch and the Vulkan TTS worker build
- [kilix](https://github.com/itsmygithubacct/kilix#readme) — lists this app in its Windows (WSL) catalog
- [kilix-content](https://github.com/itsmygithubacct/kilix-content#readme) — downloads reuse its installer

## Part of Plebian-OS

Kilix Ollama is one of the repositories behind
[Plebian-OS](https://github.com/itsmygithubacct/plebian-os#readme), a Debian
install whose desktop is [Kilix](https://github.com/itsmygithubacct/kilix#readme)
running the [Kilix 95](https://github.com/itsmygithubacct/kilix-95#readme)
desktop through [Pleb](https://github.com/itsmygithubacct/pleb#readme).
It is a development client that Plebian-OS does not install; Kilix's Windows
(WSL) catalog offers it. The map of every repository and how it ships is on
the [components page](https://github.com/itsmygithubacct/plebian-os/blob/main/docs/components.md).

## License

Kilix Ollama is licensed under the MIT License. No Ollama or llama.cpp source,
binary, model, voice, or weight artifact is vendored in this tree; see
`THIRD-PARTY-NOTICES.md`.
