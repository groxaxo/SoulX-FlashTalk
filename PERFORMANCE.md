# SoulX-FlashTalk Performance Guide

Optimized for **1× RTX 3090 (24 GB VRAM)** with a target of **RTF ≤ 1.0**.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Profiling & Bottleneck Analysis](#profiling--bottleneck-analysis)
3. [Fixes Implemented](#fixes-implemented)
4. [Quantization Paths](#quantization-paths)
5. [Benchmark Harness](#benchmark-harness)
6. [How to Run](#how-to-run)

---

## Environment Setup

### Pinned versions (tested)

| Component        | Version               |
|------------------|-----------------------|
| CUDA             | 12.4+                 |
| Python           | 3.10 – 3.12          |
| PyTorch          | 2.7.1                 |
| torchvision      | 0.22.1                |
| transformers     | ≥ 4.46.3              |
| diffusers        | ≥ 0.34.0              |
| accelerate       | ≥ 1.8.1               |
| optimum-quanto   | 0.2.6                 |
| xformers         | 0.0.31                |
| flash-attn       | 2.8.0.post2           |
| bitsandbytes     | ≥ 0.43.0 (optional)   |
| FFmpeg           | system package         |

### Install

```bash
# 1. Clone the repo
git clone https://github.com/groxaxo/SoulX-FlashTalk.git
cd SoulX-FlashTalk

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install bitsandbytes for NF4 quantization
pip install bitsandbytes>=0.43.0

# 5. Download model weights
# Place SoulX-FlashTalk-14B and chinese-wav2vec2-base into models/
```

---

## Profiling & Bottleneck Analysis

### Findings (prioritized by impact)

| # | Bottleneck | Evidence | Impact |
|---|-----------|----------|--------|
| 1 | **Model size (14B @ bf16 ≈ 28 GB)** | Exceeds RTX 3090 24 GB VRAM | Requires CPU offload (huge overhead) or quantization |
| 2 | **Redundant `torch.cuda.synchronize()` in generate()** | 8 sync points per chunk (4 pairs around model/VAE/color ops) | Forces GPU pipeline drain; adds 5–20 ms per sync pair |
| 3 | **`torch_gc()` inside `@torch.compile` function** | `calculate_x_ref_attn_map()` calls `torch.cuda.empty_cache()` in loop | Defeats graph compilation; forces full GPU cache flush |
| 4 | **CPU offload per-request overhead** | `.to(device)` + `.cpu()` + `empty_cache()` for model, CLIP, T5, VAE each generation | 100–500 ms overhead per chunk |
| 5 | **pyloudnorm meter recreated every call** | `pyln.Meter(sr)` constructed in `loudness_norm()` on every audio chunk | Minor CPU overhead but compounds over many chunks |
| 6 | **Audio `.float()` conversion** | `torch.from_numpy(audio_feature).float()` in `preprocess_audio()` | Unnecessary dtype cast when model uses bf16 |

---

## Fixes Implemented

### A. Quantization support (enables 24 GB VRAM operation)

Added three quantization backends to `FlashTalkPipeline`:

- **`quanto-int4`** — INT4 weight quantization via `optimum-quanto` (already in requirements). Reduces WanModel from ~28 GB to ~7 GB.
- **`quanto-int8`** — INT8 weight quantization. Reduces to ~14 GB.
- **`bitsandbytes-nf4`** — NF4 quantization via `bitsandbytes`. Similar reduction to INT4.

Activated via `--quantize <mode>` flag.

### B. Remove GPU sync overhead

Wrapped all `torch.cuda.synchronize()` + timing blocks in `generate()` behind `DEBUG_TIMING` flag (default: `False`). This eliminates 8 unnecessary GPU pipeline stalls per chunk.

### C. Fix `torch_gc()` in compiled function

Removed `torch.cuda.empty_cache()` / `torch.cuda.ipc_collect()` calls from inside `@torch.compile`-decorated `calculate_x_ref_attn_map()`. These calls break torch.compile graph capture and cause massive overhead.

### D. Cache pyloudnorm meter

Cached `pyln.Meter(sr)` instances in a module-level dict so they are created only once per sample rate.

### E. `torch.compile` disabled when quantized

Quantized models are typically incompatible with `torch.compile`. The pipeline now skips model compilation when a quantization mode is active (VAE compilation is still enabled).

---

## Quantization Paths

### Recommended: quanto INT4 (default for RTX 3090)

```bash
python generate_video.py \
    --ckpt_dir models/SoulX-FlashTalk-14B \
    --wav2vec_dir models/chinese-wav2vec2-base \
    --cond_image examples/man.png \
    --audio_path examples/cantonese_16k.wav \
    --quantize quanto-int4
```

**Expected VRAM:** ~12–15 GB (model ~7 GB + VAE + audio encoder + activations)
**No pre-quantization needed** — quantization is applied on-the-fly during model loading.

### Alternative: bitsandbytes NF4

```bash
pip install bitsandbytes>=0.43.0

python generate_video.py \
    --ckpt_dir models/SoulX-FlashTalk-14B \
    --wav2vec_dir models/chinese-wav2vec2-base \
    --cond_image examples/man.png \
    --audio_path examples/cantonese_16k.wav \
    --quantize bitsandbytes-nf4
```

### Comparison

| Method | VRAM (est.) | Load time | Quality | Notes |
|--------|-------------|-----------|---------|-------|
| bf16 (baseline) | ~28 GB | Fast | Best | Doesn't fit 3090 |
| bf16 + cpu_offload | ~18 GB | Fast | Best | Slow per-chunk |
| quanto-int4 | ~12–15 GB | Moderate | Good | **Recommended for 3090** |
| quanto-int8 | ~18–20 GB | Moderate | Better | Fits 3090 with headroom |
| bitsandbytes-nf4 | ~12–15 GB | Moderate | Good | Alternative to quanto |

---

## Benchmark Harness

### Run benchmark

```bash
python bench_rtf.py \
    --ckpt_dir models/SoulX-FlashTalk-14B \
    --wav2vec_dir models/chinese-wav2vec2-base \
    --audio_path examples/cantonese_16k.wav \
    --cond_image examples/man.png \
    --quantize quanto-int4 \
    --num_chunks 5 \
    --warmup_chunks 1
```

### What it measures

- **Latency** (mean, std, min, max) per video chunk
- **Throughput** (frames/second)
- **VRAM peak** (MB)
- **RTF** = processing_time / media_duration (target ≤ 1.0)
- **Pipeline load time**

### Acceptance criteria

On RTX 3090 with `--quantize quanto-int4`:
- RTF ≤ 1.0
- VRAM peak < 24 GB
- Stable output (no OOM, no crashes) for continuous operation

---

## How to Run

### Quick start (RTX 3090 optimized)

```bash
# Single GPU, quantized, no CPU offload needed
python generate_video.py \
    --ckpt_dir models/SoulX-FlashTalk-14B \
    --wav2vec_dir models/chinese-wav2vec2-base \
    --cond_image examples/man.png \
    --audio_path examples/cantonese_16k.wav \
    --audio_encode_mode stream \
    --quantize quanto-int4
```

### Gradio web UI

```bash
python gradio_app.py
# Then select "quanto-int4" in the Quantization Mode dropdown under Advanced Settings
```

### Performance knobs

| Flag | Description | Default |
|------|-------------|---------|
| `--quantize` | Quantization mode: `quanto-int4`, `quanto-int8`, `bitsandbytes-nf4` | None (bf16) |
| `--cpu_offload` | Enable CPU offload (for >24 GB VRAM without quantization) | Off |
| `--audio_encode_mode` | `stream` (chunk-by-chunk) or `once` (all at once) | `stream` |
| `--base_seed` | Random seed for reproducibility | 9999 |

### Debug timing

To enable per-step timing (useful for profiling, but hurts performance):

```python
# In flash_talk/src/pipeline/flash_talk_pipeline.py, set:
DEBUG_TIMING = True
```

### Configuration tuning

Edit `flash_talk/configs/infer_params.yaml`:

```yaml
frame_num: 33          # Frames per chunk (more = higher latency per chunk)
motion_frames_num: 5   # Context frames carried between chunks
sample_steps: 4        # Diffusion steps (fewer = faster but lower quality)
sample_shift: 5        # Timestep shift
height: 768            # Output height (smaller = faster)
width: 448             # Output width (smaller = faster)
```

Reducing `sample_steps` from 4 to 2 halves diffusion compute at the cost of quality.
Reducing resolution (e.g., 512×320) also reduces compute proportionally.
