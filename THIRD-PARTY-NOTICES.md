# Third-party notices

This development tree currently vendors no third-party source code, model
weights, voice data, generated shaders, or binary runtime artifacts. It does
carry a metadata-only optional-model catalog containing immutable upstream
locations, byte counts, digests, attribution, and licence references.

Kilix Ollama is designed to mediate an independently built and verified Ollama
runtime. Ollama is MIT-licensed and remains an external component. Vulkan
support is provided by external Debian packages, including the Vulkan loader,
Mesa drivers, and provider firmware selected by Plebian OS. Their copyright
and licence files remain with their packages.

The optional Pocket TTS candidate identifies
`kyutai/pocket-tts-without-voice-cloning` at the immutable revision
`d29db7978e464fb90cb3359ee0c69a273b9142cc`. Its model card declares CC BY 4.0
and supplies an additional prohibited-use notice. The explicit installer
downloads the pinned model card directly from upstream and retains it with the
local conversion. Users must credit Kyutai, link the CC BY 4.0 licence,
preserve supplied notices, indicate changes, avoid implied endorsement, and
comply with the pinned acceptable-use conditions. In particular, use must not
facilitate unauthorized voice impersonation, deception, unlawful or harmful
content, or privacy invasion. See:

- https://creativecommons.org/licenses/by/4.0/
- https://huggingface.co/kyutai/pocket-tts-without-voice-cloning

No Pocket TTS model, tokenizer, GGUF conversion, speaker material, or generated
audio is distributed by this repository. Optional bytes are fetched only from
the pinned upstream revision after explicit acceptance and verified locally.
The separately installed Plebian OS converter obtains llama.cpp source at
commit `7e4c0a96880dae4fc4268ad441f8a6446bd5460a`, retains its MIT license and a
modification notice, and does not obtain model bytes.

The repository's native persistent-worker source is original Kilix code that
uses llama.cpp's public llama, ggml-backend, and mtmd interfaces. Plebian OS
builds the external llama.cpp source independently from the same immutable
commit, verifies the complete native output closure, and retains the upstream
MIT license. Neither that source archive nor its compiled libraries, generated
Vulkan shaders, or worker executable is carried in this repository.

The development Ollama runtime remains outside this repository. Its Plebian OS
installer accepts only a local exact closure and verifies the candidate's
Ollama, llama.cpp, Go, native-library, and module notice tree. That tree is
provisional audit material, not a frozen release notice set; it must be
regenerated from any eventual release artifact. No carrier URL or runtime byte
is represented here while carrier approval remains open.
