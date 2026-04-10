# Phase 17: Inference Gap Closure - Discussion Log

**Session:** 2026-04-05
**Participants:** User + Claude

---

## Area 1: DeepSeek Routing

**Q:** How should DeepSeek requests reach the llama.cpp backend?
Options: (A) backends/factory.zig abstraction, (B) dedicated handler like chat_gptoss.zig, (C) direct in mod.zig

**A:** "B and then A as a fallback to other models so we can expedite and collect llama.cpp benefits.
The thing with passing abstraction is that we lose control of MLX which is the intent of this project.
I want to keep it simple so we can add A as a backlog item, go for B for all models, get the code uber
organized, without stubs, without noops and mocks, and then spin out a library for model running on
mlx for zig."

**Decision:** Handler-per-model pattern (B). New `chat_deepseek.zig` + `LlamaBackend` in server.zig.
`backends/factory.zig` as backlog item, not Phase 17 scope. Architecture direction: MLX-Swift style,
eventual library extraction.

---

## Area 2: Speculative Decoding (GAP-07)

**Q:** Remove dead speculation code or keep with documentation?

**A:** "remove it, zero dead code, create a plan and let's implement it or drop if the functionality
is not needed."

**Decision:** Full removal of `--draft-model`, `--no-speculation`, `--speculation-depth` flags,
config fields, init block, null pass-throughs. No backlog item — evaluate need from scratch later.

---

## Area 3: MLX Backend Factory (GAP-05) + backends/ cleanup

**Q:** What happens to `factory.zig`, `mlx_backend.zig`, `backend.zig`?

**Follow-up concern:** "I want to know what happens to models that are working now. Remove or merge
lost code. Review the llama.cpp decision, we should have this backend but it shouldn't mean that we
kill our project objective."

**Discussion:** Mapped all current working paths. llama.cpp is valid for DeepSeek (MoE/GGUF) but
must not become the primary path. `backend.zig` holds shared types needed by both surviving backends
and future Gemma 4. `factory.zig` and `mlx_backend.zig` are pure stubs with no callers.

**Gemma 4 check:** User asked to verify consistency with Phase 18. Confirmed: Gemma 4 E4B is
MLX-native → will use `chat_gemma4.zig` + `mlx_gemma4_backend.zig` (same pattern as GPT-OSS).
`backend.zig` shared types are reused. No conflict.

**A:** "Yes it does feel right. Make sure it is in line with our Gemma 4 plans."

**Decision:** Delete `factory.zig` + `mlx_backend.zig`. Keep `llama_cpp.zig`, `mlx_gptoss_backend.zig`,
`backend.zig`. Update `backends/mod.zig`.

---

## Area 4: Prompt Cache parseIndex (GAP-06)

**Q:** Fix now or defer? (Not in Phase 17 success criteria.)

**A:** "fix it, always improve and keep optimizations for speed"

**Decision:** Fix `cache/prompt_cache.zig:433` — real `index.json` disk reads, read once at init,
keep in memory. Speed is a first-class constraint.

---

## Architectural Theme

User articulated a consistent vision across all four discussions:
- Zero dead code, zero stubs, zero mocks
- MLX stays primary and directly controlled (MLX-Swift style)
- llama.cpp is a specific tool for specific models (DeepSeek), not a generic backend
- End goal: extract a clean Zig MLX inference library

This theme should guide all Phase 17 implementation decisions.
