# MCP Auto-Discovery for zlx

## Goal
Enable OpenCode to auto-discover zlx via `.well-known/mcp.json` endpoint, eliminating manual configuration.

## Current State
- zlx is an OpenAI-compatible inference server
- OpenCode requires manual config: `~/.config/opencode/config.json`
- No auto-discovery mechanism exists

## Target State
- OpenCode probes `http://localhost:8081/.well-known/mcp.json`
- Auto-discovers zlx capabilities
- One-click setup: just enter `http://localhost:8081` in OpenCode

## Implementation Plan

### Phase 1: MCP Server Metadata Endpoint (30 min)
**File:** `src/api/handlers.zig`

Add endpoint: `GET /.well-known/mcp.json`

```json
{
  "name": "zlx",
  "version": "1.1.0",
  "description": "Local LLM inference server with OpenAI-compatible API",
  "transport": {
    "type": "openai-http",
    "url": "http://127.0.0.1:8081/v1"
  },
  "capabilities": {
    "chat_completions": true,
    "streaming": true,
    "models_list": true,
    "tools": false
  },
  "models": [
    {
      "id": "qwen2.5-coder-1.5b",
      "name": "Qwen 2.5 Coder 1.5B",
      "size_gb": 1.0
    }
  ]
}
```

### Phase 2: Route Registration (15 min)
**File:** `src/api/server.zig`

Register new route:
```zig
router.get("/.well-known/mcp.json", handlers.handleMcpDiscovery);
```

### Phase 3: CORS Headers (15 min)
**File:** `src/api/handlers.zig`

Ensure endpoint returns:
```
Access-Control-Allow-Origin: *
Content-Type: application/json
```

### Phase 4: Testing (30 min)

1. Start zlx
2. Test endpoint: `curl http://localhost:8081/.well-known/mcp.json`
3. Configure OpenCode with just the base URL
4. Verify auto-discovery works

### Phase 5: Documentation (15 min)

Update README:
```markdown
## OpenCode Auto-Configuration

Just enter `http://localhost:8081` in OpenCode settings.
OpenCode will auto-discover zlx capabilities via MCP protocol.
```

## Alternative: Full MCP Server (Future)

Instead of just discovery, zlx could act as a full MCP tool server:

**Tools exposed:**
- `chat_completion` - Generate text with local model
- `list_models` - Show available models
- `get_model_info` - Get model metadata

**Benefits:**
- OpenCode uses zlx through native MCP interface
- Better integration with MCP ecosystem
- Support for multiple AI clients (Claude, ChatGPT, etc.)

**Effort:** 4-6 hours

## Decision

**Option A (Quick):** Just add discovery endpoint (1-2 hours)
- Minimal code change
- OpenCode can auto-configure
- Still uses OpenAI-compatible API

**Option B (Full):** Implement MCP tool server (4-6 hours)
- Native MCP integration
- Better long-term ecosystem support
- More complex, needs tool schema design

**Recommendation:** Option A for now. Add full MCP server later if needed.

## Success Criteria

- [ ] `curl http://localhost:8081/.well-known/mcp.json` returns valid JSON
- [ ] OpenCode discovers zlx without manual config file
- [ ] CORS headers allow browser access
- [ ] Documentation updated

## References

- SEP-1649: MCP Server Cards - https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1649
- MCP Specification: https://modelcontextprotocol.io/specification/2025-03-26/basic/transports
- OpenCode MCP Support: https://github.com/anomalyco/opencode/issues/7054
