# GGUF Conversion Guide for MoE Models

This guide explains how to convert DeepSeek and GPT-OSS models to GGUF format for use with llama.cpp backend.

## Overview

The llama.cpp backend requires models in GGUF format. Pre-converted GGUF models are available from community sources on HuggingFace, but if you need to convert from the original weights, follow this guide.

## DeepSeek-Coder-V2-Lite

### Pre-converted GGUF (Recommended)

Download pre-converted GGUF from TheBloke:
```bash
wget https://huggingface.co/TheBloke/deepseek-coder-v2-lite-GGUF/resolve/main/deepseek-coder-v2-lite.Q4_K_M.gguf \
    -O ./models/deepseek-coder-v2-lite.Q4_K_M.gguf
```

Available quantizations:
- `Q4_K_M` (~4.5GB) - Recommended, good quality/size tradeoff
- `Q5_K_M` (~5.5GB) - Better quality, larger size
- `Q6_K` (~6.5GB) - Best quality, largest size

### Manual Conversion (if needed)

If you have the original safetensors:

```bash
# Clone llama.cpp (or use the one in src/llama.cpp/)
cd src/llama.cpp

# Install Python dependencies
pip install -r requirements/requirements-convert.txt

# Convert to GGUF
python convert_hf_to_gguf.py \
    /path/to/deepseek-coder-v2-lite \
    --outfile ./deepseek-coder-v2-lite.Q4_K_M.gguf \
    --outtype Q4_K_M
```

## GPT-OSS-20B

### Pre-converted GGUF (Recommended)

Download from bartowski or unsloth (when available):
```bash
# Primary source (when available)
wget https://huggingface.co/bartowski/GPT-OSS-20B-GGUF/resolve/main/GPT-OSS-20B-Q4_K_M.gguf \
    -O ./models/GPT-OSS-20B-Q4_K_M.gguf

# Alternative mirror
wget https://huggingface.co/unsloth/GPT-OSS-20B-GGUF/resolve/main/GPT-OSS-20B-Q4_K_M.gguf \
    -O ./models/GPT-OSS-20B-Q4_K_M.gguf
```

### Manual Conversion from OpenAI Weights

1. Download original weights from OpenAI (requires HuggingFace token):
```bash
huggingface-cli login
huggingface-cli download openai/gpt-oss-20b \
    --local-dir ./gpt-oss-20b-original \
    --local-dir-use-symlinks False
```

2. Convert to GGUF:
```bash
cd src/llama.cpp
pip install -r requirements/requirements-convert.txt

# Q4_K_M (recommended, ~11GB)
python convert_hf_to_gguf.py \
    /path/to/gpt-oss-20b-original \
    --outfile ./GPT-OSS-20B-Q4_K_M.gguf \
    --outtype Q4_K_M

# Or Q5_K_M for better quality (~13GB)
python convert_hf_to_gguf.py \
    /path/to/gpt-oss-20b-original \
    --outfile ./GPT-OSS-20B-Q5_K_M.gguf \
    --outtype Q5_K_M
```

## Testing Converted Models

After conversion, test with zlx:

```bash
# DeepSeek
./zig-out/bin/zlx \
    --model ./models/deepseek-coder-v2-lite.Q4_K_M.gguf \
    --backend llama_cpp \
    --turboquant

# GPT-OSS
./zig-out/bin/zlx \
    --model ./models/GPT-OSS-20B-Q4_K_M.gguf \
    --backend llama_cpp \
    --turboquant
```

## Troubleshooting

### OOM During Conversion
- Use `--split-max-tensors 0` to reduce memory
- Ensure at least 32GB RAM for GPT-OSS conversion
- Use swap if necessary

### Tokenizer Issues
- Ensure `tokenizer.json` is present in source directory
- Check `tokenizer_config.json` for chat template

### Missing GGUF Metadata
- llama.cpp requires specific metadata for MoE models
- Ensure `--outtype` matches the model's supported quantizations
- Check GGUF version compatibility (v3 recommended)

## Quantization Selection

| Model | Quantization | Size | Quality | Use Case |
|-------|--------------|------|---------|----------|
| DeepSeek | Q4_K_M | 4.5GB | Good | Daily coding, 16GB RAM |
| DeepSeek | Q5_K_M | 5.5GB | Better | Quality coding, 24GB RAM |
| GPT-OSS | Q4_K_M | 11GB | Good | General chat, 16GB RAM |
| GPT-OSS | Q5_K_M | 13GB | Better | Quality chat, 24GB RAM |

Note: TurboQuant KV cache compression can reduce memory by ~60% on all models.
