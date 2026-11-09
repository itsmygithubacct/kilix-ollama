# Optional local TTS conversion

Reference for [Kilix Ollama](../README.md).

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
