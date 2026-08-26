# Changelog

## Unreleased

- Add a qualification-gated NVK/TU106 Qwen 3.5 runtime tuning profile.
- Fix service candidates at one 4096-token model and explicitly enable Vulkan
  only for a qualified Vulkan backend.

## 0.2.1 — development

- Add the unpublished, model-free application spine.
- Add fixed runtime layout and launch specifications without starting a daemon.
- Add internal Vulkan/Nouveau discovery with qualification-gated selection.
- Add a deny-by-default Ollama API mediation policy and bounded health parsers.
- Add deterministic human and JSON diagnostics.
