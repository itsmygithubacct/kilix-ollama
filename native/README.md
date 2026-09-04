# `native/` — source of record, not a build tree

Nothing in this repository compiles these files, and `make install` does not
install them. They are here as the source of record for worker binaries that
**`plebian-os`** builds against the admitted `llama.cpp` closure and installs
into the image.

`kilix_tts_worker.cpp` includes `llama.cpp` internals (`arg.h`, `common.h`,
`llama.h`, `mtmd.h`) that are not vendored here, so it cannot be built from this
tree even in principle. Adding a build rule would produce a second binary that
the runtime never loads — the runtime resolves
`/usr/lib/plebian-os/ollama/kilix-tts-worker-current` and validates it through
`kilix_ollama.trust`.

## Why the manifest exists

A tracked source that no rule consumes can drift silently from the binary that
actually ships. `SOURCE-MANIFEST.json` pins each file's SHA-256 and names the
closure it belongs to, and `tests/test_native_worker_source.py` enforces it, so
editing a source without updating its declaration fails `make check`.

**What this does and does not catch.** It catches drift *inside this
repository*: an edited, added or removed source, and a runtime closure constant
that no longer matches the one the source is declared against. It does **not**
prove the shipped binary was built from these bytes — that binding lives in
`plebian-os`, which owns the build, and nothing observable from this repository
can establish it. Closing that half needs `plebian-os` to record the same
source digest against the same closure id.
