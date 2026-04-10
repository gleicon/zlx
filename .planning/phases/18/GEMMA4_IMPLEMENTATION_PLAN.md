# Gemma 4 Implementation Plan - Using llama.cpp

## Current State (as of 2026-04-08)

### ✅ Completed:
1. **Model Detection & Registry**
   - `gemma4-e4b` alias registered in `src/download/mod.zig`
   - `gemma4` architecture added to all enums
   - Config parsing handles `text_config` nesting for multimodal models
   - Detection order fixed (Gemma 4 checked before GPT-OSS)

2. **Basic MLX Transformer Stub**
   - `src/mlx.zig/src/gemma4.zig` created with architecture definitions
   - SlidingWindowAttention, MLP, TransformerBlock structs defined
   - Config loader that parses Gemma 4's JSON structure

### ⚠️ Blockers Identified:
1. **MLX GenerationState Complexity**
   - GenerationState is hardcoded to `qwen.Transformer`
   - Making it generic requires touching 10+ files
   - Complex iterator pattern with cache management

2. **Native MLX Limitations**
   - Would need full weight loading implementation
   - Sliding window attention needs careful implementation
   - Testing requires complete pipeline

## Proposed Solution: llama.cpp Backend

### Why llama.cpp:
1. ✅ Already integrated in zlx (built as submodule)
2. ✅ Gemma 4 GGUF models readily available
3. ✅ Handles sliding window attention natively
4. ✅ Proven working with OpenCode format
5. ✅ Can convert MLX safetensors to GGUF or download pre-converted

### Implementation Approach:

#### Phase 1: Create llama.cpp-based Gemma 4 Handler
**File:** `src/api/chat_gemma4.zig` (update existing)

```zig
// Use llama.cpp bindings instead of MLX
// - Link against libllama (already built)
// - Use llama_model_load_from_file for GGUF
// - Implement chat template for Gemma 4
// - Support <|turn|> tokens and reasoning format
```

#### Phase 2: GGUF Model Setup
**Options:**
1. **Convert existing model:**
   ```bash
   # Use llama.cpp's convert script or huggingface-cli
   python convert_hf_to_gguf.py ./models/gemma4-e4b
   ```

2. **Download pre-converted:**
   ```bash
   # From bartowski or other HF repos
   huggingface-cli download bartowski/gemma-4-e4b-it-GGUF
   ```

#### Phase 3: Chat Template Implementation
**Gemma 4 Format:**
```
<|turn|>user
User message<|turn|>model
Model response<|turn|>user
...
```

**Features:**
- Default temperature 0.3 (per Gemma 4 best practices)
- Reasoning tokens support (for thinking models)
- E4B (every 4 bits) quantization awareness

### Architecture Decision:

```
┌─────────────────────────────────────────────────────────┐
│                    zlx Server                           │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   Qwen      │  │  DeepSeek   │  │    Gemma 4      │ │
│  │  (MLX.zig)  │  │  (MLX.zig)  │  │  (llama.cpp)    │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
│         │                │                    │          │
│         └────────────────┴────────────────────┘          │
│                           │                             │
│              ┌────────────┴────────────┐               │
│              │   OpenAI-compatible    │               │
│              │     /v1/chat/completions │               │
│              └─────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

### Files to Create/Modify:

#### New Files:
1. `src/backends/llama_gemma4.zig` - llama.cpp wrapper for Gemma 4
2. `tests/gemma4_test.zig` - Integration tests
3. `docs/GEMMA4_SETUP.md` - Setup guide

#### Modified Files:
1. `src/api/chat_gemma4.zig` - Route to llama.cpp backend
2. `src/api/server.zig` - Add Gemma 4 dispatch
3. `src/chat/templates.zig` - Add Gemma 4 chat format
4. `src/download/mod.zig` - Update alias info

### Testing Strategy:

1. **Unit Tests:**
   - Config loading from JSON
   - Chat template formatting
   - Model detection

2. **Integration Tests:**
   - Server startup with `--model gemma4-e4b`
   - Single completion request
   - Streaming completion request
   - OpenCode compatibility check

3. **Manual Tests:**
   - Connect OpenCode to localhost:8080
   - Generate code snippet
   - Verify response format

### OpenCode Integration:

**Configuration:**
```json
{
  "models": [{
    "title": "Gemma 4 E4B (Local)",
    "provider": "openai",
    "model": "gemma4-e4b",
    "apiBase": "http://localhost:8080/v1"
  }]
}
```

**Expected Behavior:**
- 4-bit quantized weights load quickly
- KV cache uses ~3GB for typical context
- Generation speed: 10-30 tokens/sec (depending on prompt)
- Proper handling of long contexts (up to 32k tokens)

### Timeline:

**Phase 1 (2-3 hours):**
- Create llama.cpp wrapper
- Implement chat template
- Basic handler

**Phase 2 (1-2 hours):**
- Model setup (convert or download)
- Integration tests

**Phase 3 (1 hour):**
- OpenCode testing
- Documentation

### Success Criteria:

1. ✅ `./zig-out/bin/zlx --model gemma4-e4b` starts without errors
2. ✅ `curl http://localhost:8080/v1/models` shows Gemma 4
3. ✅ Chat completion returns valid response
4. ✅ OpenCode connects and generates code
5. ✅ Tests pass: `zig build test`

### Risks & Mitigation:

| Risk | Mitigation |
|------|-----------|
| GGUF conversion fails | Use pre-converted model from HF |
| llama.cpp binding issues | Use subprocess approach (llama-cli) |
| Memory issues | Add memory check before loading |
| Slow performance | Document expected speed, suggest smaller context |

### Next Steps:

1. ✅ Approve this plan
2. Create llama.cpp backend wrapper
3. Download/setup GGUF model
4. Implement and test
5. Document for OpenCode users

---

**Status:** Ready to implement
**Priority:** High (blocks Gemma 4 usage)
**Estimated Time:** 4-6 hours
