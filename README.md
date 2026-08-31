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

## Development commands

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

## Local product commands

Open the explicit development TUI in a real terminal. It contacts no runtime,
downloads no model, and treats pasted input as inert until the external Kilix
pane integration and prompt editor are accepted:

```sh
./kilix-ollama tui
```

The local TUI owns alternate-screen and terminal-mode cleanup. The separate
real-Kilix nested-pane and hard-crash force-restore proof remains an external
P9 gate.

Browse the exact three-entry metadata catalog without contacting a runtime, or
merge installed state from the private mediation socket:

```sh
./kilix-ollama library
./kilix-ollama library --installed
./kilix-ollama models
./kilix-ollama ps
./kilix-ollama show granite4.1:3b --development-runtime
```

Run one prompt from arguments, standard input, or one non-symlinked UTF-8 file.
Plain output is sanitized while streaming; JSON, JSON Lines, and atomic file
output are also available:

```sh
./kilix-ollama ask granite4.1:3b --development-runtime 'Explain this tree.'
./kilix-ollama ask granite4.1:3b --development-runtime --file prompt.txt --json
./kilix-ollama ask granite4.1:3b --development-runtime --output answer.md 'Summarize.'
```

Start a persistent terminal chat, run a single scripted turn, or manage the
owner-private conversation store:

```sh
./kilix-ollama chat granite4.1:3b --development-runtime
./kilix-ollama chat granite4.1:3b --development-runtime --message 'Hello' --json
./kilix-ollama chat granite3.2-vision:2b --development-runtime --message 'Describe this.' --attach image.png --json
./kilix-ollama chat --conversation CONVERSATION_ID --regenerate --json --development-runtime
./kilix-ollama chat --conversation CONVERSATION_ID --switch-model granite3.2-vision:2b --message 'Continue' --json --development-runtime
./kilix-ollama conversations list
./kilix-ollama conversations fork CONVERSATION_ID --messages 2 --title 'Alternative'
./kilix-ollama conversations export CONVERSATION_ID --output transcript.md
```

Completed turns commit as one compare-and-swap update. Generation cancelled
before commit, malformed tool output, an exhausted tool budget, or a concurrent
conversation update leaves the previously stored conversation intact.
Regeneration replaces only the suffix after the last user message. Forking
refuses a prefix that splits a tool exchange, and model switching verifies the
target is installed before it changes stored state. Interactive chat also
accepts `/regenerate`, `/fork`, and `/model MODEL`.

`--attach` accepts up to four explicitly selected PNG/JPEG files for a scripted
`--message`. The client opens each source-path component through no-follow
directory descriptors, enforces per-file and aggregate byte limits plus bounded
image dimensions/pixels, and retains the validated bytes in memory until the
turn succeeds. Before first use, explicitly provision the owner-only snapshot
directory:

```sh
install -d -m 0700 "${XDG_DATA_HOME:-$HOME/.local/share}/kilix-ollama/attachments"
```

The client never creates or repairs that trust root. It copies successful turns
through its verified descriptor into mode-0600 content-addressed snapshots
before committing the conversation. Invalid input, unavailable or failed
inference, a stale conversation race, or a missing/unsafe trust root publishes
no new snapshot. Conversation records retain only typed digest, media, size and
dimension metadata—never the selected path or filename. Every reuse rechecks
the snapshot digest and image metadata before runtime contact.
Markdown export states the omitted image count and never embeds attachment
bytes implicitly.

`--tools` offers exactly four read-only tools: UTC time, bounded arithmetic,
reading an explicitly selected text file, and bounded search beneath an
explicitly selected directory. Every call still requires interactive approval.
Tools are absent from the request by default. `tools --json` prints the exact
registry without enabling it.

No prompt command performs an implicit pull. `pull` currently fails before
runtime or network access because the shared F100 carrier/receipt authority is
0 of 1 available. `--development-runtime` admits only the exact local R3
candidate; it never admits an unknown runtime, cloud endpoint, or cloud-tagged
model.

Service descriptions use only the fixed root-owned launcher
`/usr/lib/plebian-os/ollama/ollama-current`. Diagnostics reject a symlink,
unexpected owner, group/world-writable component, or substituted ancestor in
that trusted path.

Plebian OS carries the audited private-listener patch and a small fail-closed
runtime installer. The verified native closure is not shipped, and no public
carrier has been accepted. Development installation therefore requires the
exact local closure explicitly:

```sh
sudo plebian-os-install-deps --vulkan-nouveau
sudo plebian-os-install-kilix-ollama-runtime --closure /path/to/exact/closure
```

The installer verifies the complete 119,645,888-byte executable/library set,
relative library links, notices, runtime package tuple, and downstream patch
identity before publishing the fixed launcher atomically. It downloads
nothing and contains no model or weight artifact.

## Optional local TTS conversion

`tts-models` is metadata-only and never accesses the network or model bytes.
The current candidate pins the Kyutai Pocket TTS English source weight,
tokenizer, and model card at one immutable upstream revision, along with the
expected Q8_0 GGUF model and projector identities. It intentionally includes
no speaker embedding, reference audio, generated audio, or product voice.

Installation is an explicit user action:

```sh
./kilix-ollama tts-install --accept-model-terms
./kilix-ollama tts-install --accept-model-terms --discard-verified-source
./kilix-ollama tts-status --verify
./kilix-ollama tts-remove --confirm
```

Before downloading, the CLI identifies Kyutai, CC BY 4.0, the pinned model
card, and its prohibited-use notice. Downloads come only from the exact
upstream revision into a user-owned content-addressed cache. Every byte count
and SHA-256 must match before conversion or atomic admission. HTTP 401/403
sends the user back to the upstream acceptance flow; Kilix stores no account
credentials and never falls back to another mirror.

Conversion is offline through the fixed root-owned launcher
`/usr/lib/plebian-os/ollama/pocket-tts-convert-hf-to-gguf`. Plebian OS keeps its
large Python conversion environment opt-in; prepare the measured candidate
with:

```sh
sudo plebian-os-install-deps --vulkan-tts
sudo plebian-os-install-ollama-converter
sudo plebian-os-install-kilix-vulkan-tts
```

The dependency command installs the common Vulkan runtime, six exact direct
Debian converter pins, and two native-build pins. The converter installer
verifies an immutable llama.cpp source archive and installs a 1.6 MiB
Pocket-only offline wrapper. The native installer builds the persistent worker
from that same 36,889,653-byte archive and admits only its independently
reproduced 66,532,472-byte runtime closure. It needs 1 GiB of temporary free
space, then discards the archive and build tree. Neither installer downloads
model weights, and none of these commands runs automatically. The measured
converter package closure is roughly 150 MiB to download and 0.8 GiB installed.
The closure is not release-admitted and an actual install fails closed until it
is present. A successful conversion retains the pinned model card and records
source, converter, and output identities locally.
`--discard-verified-source` removes the verified source inputs and cache only
after the 123,629,056-byte GGUF pair verifies; the default retains them. Peak
model conversion requires at least 512 MiB free in addition to the separately
installed converter closure.

Removal requires confirmation and refuses a directory with any unexpected,
unverified, or substituted content. None of these operations authorizes Kilix
to mirror or publish source weights, converted weights, or voices.

## Vulkan closure

The runtime discovery contract recognizes:

- `libvulkan1` and `mesa-vulkan-drivers` for Vulkan;
- `firmware-nvidia-graphics` when the selected DRM path is Nouveau; and
- `vulkan-tools` as qualification-image tooling, not a runtime dependency.

Detection is not qualification. Until measured correctness, stability, and
performance evidence is admitted for an exact runtime and hardware profile,
automatic selection remains on CPU and an explicit Vulkan request fails.

The tree now carries one measured-but-unqualified tuning candidate named
`nvk-tu106-qwen35-4b-r1`. It is bound to the exact TU106, NVK, Ollama, and
llama.cpp candidate tuple. It enables the Vulkan runner and MMVQ, limits the
runtime to one 4096-token model, and disables llama.cpp host prompt snapshots
and recurrent context checkpoints to avoid a measured cross-prompt slowdown in
the Qwen 3.5 hybrid path. The profile cannot qualify or select itself; a release
integrator must provide matching external evidence first. It contains no model
location, model digest, or weight artifact.

A second measured candidate, `nvk-tu106-qwen35-4b-avatar-r1`, adds a 256 MiB
Ollama VRAM reservation for a separately managed small Vulkan speech worker.
Its companion co-residency record keeps one 4096-token Qwen model resident,
uses a 512-token/128-batch speech shape, requires first-use shader warm-up, and
finishes an LLM turn before prioritizing speech. Active overlap is recorded as
a degraded fallback. This is scheduling metadata only: it neither installs nor
identifies a speech model, and it cannot admit itself.

The persistent Pocket worker retains its verified model pair between requests
but creates fresh inference context, sampler, and generator state for each
utterance. Text uses a private binary pipe protocol and is never placed in
argv, prompt files, or logs. On the local TU106/NVK development machine, twelve
fixed-seed requests were byte-identical and warm synthesis ran faster than real
time, but conversational prompts later exposed malformed near-silent output.
Container validity and repeatability were therefore insufficient qualification.
Both Pocket runners now reject silent or near-silent PCM before an application
can play it, and this native candidate is not admitted for application speech.
A real co-residency probe confirmed only that Qwen and Pocket do not execute on
the accelerator concurrently. These are local development results, not an
accepted schema, broad hardware qualification, or CUDA-parity evidence.

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

## License

Kilix Ollama is licensed under the MIT License. No Ollama or llama.cpp source,
binary, model, voice, or weight artifact is vendored in this tree; see
`THIRD-PARTY-NOTICES.md`.
