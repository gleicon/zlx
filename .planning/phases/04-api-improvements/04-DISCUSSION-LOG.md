# Phase 4: API Improvements - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-01
**Phase:** 4-API Improvements
**Areas discussed:** All gray areas (default decisions applied)

---

## Discussion Summary

User opted to skip detailed gray area discussion with "move on" directive. Applied standard OpenAI API-compatible defaults for all decisions.

---

## Stop Sequence Detection

| Option | Description | Selected |
|--------|-------------|----------|
| Token-level checking | Check after each token generated | ✓ |
| Word-level checking | Check only at word boundaries | |
| Trie-based matching | Complex multi-sequence trie structure | |

**User's choice:** Applied default — token-level checking with simple suffix matching
**Notes:** Standard OpenAI behavior is to stop immediately when any stop sequence is matched in the generated text. Multi-character string matching at token boundaries is sufficient for single-user server.

---

## Logprobs Implementation

| Option | Description | Selected |
|--------|-------------|----------|
| Store during generation | Keep logits array, extract top-k | ✓ |
| Re-compute on demand | Run forward pass again for logprobs | |
| Approximate with sampling | Track only sampled token's neighbors | |

**User's choice:** Applied default — store logits during generation
**Notes:** Memory tradeoff acceptable for single-user server. MLX already provides logits as f32 array before softmax.

---

## Sampling Parameter Architecture

| Option | Description | Selected |
|--------|-------------|----------|
| API layer validation | Validate and clamp at API boundary | ✓ |
| Inference layer validation | Pass raw values, validate in generator | |
| MLX-native validation | Rely on MLX to handle invalid values | |

**User's choice:** Applied default — API layer validation with clamping
**Notes:** Early validation provides clearer error messages and consistent behavior.

---

## Seed-Based Determinism

| Option | Description | Selected |
|--------|-------------|----------|
| PCG32 RNG | Standard deterministic PRNG | ✓ |
| SplitMix64 | Alternative deterministic RNG | |
| std.crypto.random with seed | Not possible — CSPRNG | |

**User's choice:** Applied default — PCG32 RNG when seed specified
**Notes:** PCG32 is well-tested, fast, and provides good statistical quality for sampling.

---

## Error Handling Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| OpenAI-compatible | Match OpenAI error response format | ✓ |
| Custom format | Simpler structure specific to zlx | |
| Verbose debugging | Include stack traces in errors | |

**User's choice:** Applied default — OpenAI-compatible format
**Notes:** Maximum client compatibility with existing OpenAI SDKs and tools.

---

## Request Timeouts

| Option | Description | Selected |
|--------|-------------|----------|
| Total request timeout | 60s from start to finish | ✓ |
| Per-token timeout | Cancel if single token takes too long | |
| No timeout | Current behavior — unlimited | |

**User's choice:** Applied default — 60s total request timeout
**Notes:** Per-token timeout less useful for LLM inference where early tokens may take longer.

---

## the agent's Discretion

The following areas were left to the agent's discretion during implementation:

1. **RNG initialization details** — PCG32 seeding implementation
2. **Penalty calculation algorithm** — Exact math for presence/frequency penalties
3. **Logit bias storage format** — Hash map vs array lookup
4. **Error message phrasing** — Keep helpful but concise
5. **Timeout cancellation mechanism** — Signal handling or async cancellation

---

## Deferred Ideas

None captured during this discussion.

---

*End of discussion log for Phase 4*
