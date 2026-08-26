# Kilix Ollama

Kilix Ollama is a local lifecycle, compatibility, and mediation layer for an
Ollama runtime managed by Plebian OS. This tree is an unpublished development
spine for the 0.2.1 line. Its diagnostics perform read-only discovery and emit
provisional contracts. A separate, explicit optional-TTS command can download
hash-pinned inputs directly from their immutable upstream revision and convert
them locally; no command starts Ollama or performs inference.

The initial implementation establishes five boundaries:

- a deny-by-default API route and header policy;
- a private per-user XDG layout and same-UID peer requirement;
- CPU/Vulkan discovery with qualification-gated, fail-closed selection; and
- immutable candidate service descriptions for either a private Unix listener
  or a loopback listener isolated inside an owner-specific network namespace;
  and
- an opt-in, upstream-only Pocket TTS source catalog and atomic local
  conversion boundary with no bundled model or voice artifacts.

Ordinary host loopback is not an isolation boundary on a multi-user machine.
Neither transport has been selected for the release, and no command in this
revision launches either one. The service description also starts
from a cleared, fixed environment and never inherits caller-supplied
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

Service descriptions use only the fixed root-owned launcher
`/usr/lib/plebian-os/ollama/ollama-current`. Diagnostics reject a symlink,
unexpected owner, group/world-writable component, or substituted ancestor in
that trusted path.

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
```

The first command installs the common Vulkan runtime and six exact direct
Debian converter pins. The second verifies an immutable llama.cpp source
archive and installs a 1.6 MiB Pocket-only offline wrapper; it does not download
model weights. Its measured package closure is roughly 150 MiB to download and
0.8 GiB installed, so neither command runs automatically. The closure is still
not release-admitted and an actual install fails closed until it is present. A
successful conversion retains the pinned model card and records source,
converter, and output identities locally. `--discard-verified-source` removes
the verified source inputs and cache only after the 123,629,056-byte GGUF pair
verifies; the default retains them. Peak model conversion requires at least
512 MiB free in addition to the independently installed converter closure.

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

## Current limits

There is no conversation schema, supervisor, live mediation server, UI,
product voice, or synthesis service in this revision. Model and weight
artifacts remain deliberately absent from the repository. The optional TTS
catalog and installer are an unfrozen local-conversion contract, not model or
runtime admission. Python and the standard library are a provisional
implementation choice for this local spike; downloads reuse Kilix Content,
and conversion requires a separately packaged, pinned converter closure.

## License

Kilix Ollama is licensed under the MIT License. No Ollama or llama.cpp source,
binary, model, voice, or weight artifact is vendored in this tree; see
`THIRD-PARTY-NOTICES.md`.
