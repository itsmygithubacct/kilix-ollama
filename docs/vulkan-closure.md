# Vulkan closure

Reference for [Kilix Ollama](../README.md).

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
