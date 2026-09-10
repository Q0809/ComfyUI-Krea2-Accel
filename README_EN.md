# ComfyUI-Krea2-Accel

Inference-acceleration custom nodes for **locally-run Krea 2** (`Comfy-Org/Krea-2`, architecture tag `krea2` in ComfyUI).

At its core is a **TeaCache-style residual cache**: between adjacent sampling steps, the *total* update produced by the DiT's 28 transformer blocks barely changes, so the whole block stack can be skipped and the previous step's residual reused. No training, no weight conversion, no attention swap — it only changes the inference call path.

> **⚠️ Local models only.** This only accelerates `.safetensors` checkpoints loaded via `UNETLoader` / `Load Diffusion Model`. Krea 2's **cloud Partner Nodes** (which call Krea's official API) do not run the DiT locally, so this package does nothing for them.

---

## Contents

- [Nodes](#nodes)
- [Benchmarks](#benchmarks)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Parameters](#parameters)
- [Tuning: read the console, don't guess](#tuning-read-the-console-dont-guess)
- [How it works (source-level)](#how-it-works-source-level)
- [Relation to other accelerators](#relation-to-other-accelerators)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Limitations](#limitations)
- [License & credits](#license--credits)

---

## Nodes

All under the `krea2/acceleration` category in ComfyUI.

| Node | Internal name | What it does |
|------|---------------|--------------|
| **Krea2 Cache Patch (TeaCache)** | `Krea2CachePatch` | The main node. Injects the residual cache into MODEL — typically **1.3x–2x** faster |
| **Krea2 Cache Stats** | `Krea2CacheStats` | Prints hit rate, reuse count, theoretical speedup, per-step distances — **use it to verify the cache is actually firing** |
| **Krea2 Load Model (FP8)** | `Krea2LoadDiffusionModelFP8` | Optional. Loads weights at `fp8_e4m3fn` etc. (12.9B model: ~26 GB bf16 → ~12 GB fp8) |

The FP8 loader is **optional** — it is just a thin wrapper around ComfyUI's native loader, and if it fails to import, the other two nodes are unaffected.

---

## Benchmarks

Hardware: **RTX 4060 Laptop 8GB**. Model: **Krea2 Turbo fp8**. Workflow: three-stage upscale.

| Config | Total time | vs baseline |
|--------|-----------|-------------|
| Cache off | ~165 s | 1.00x |
| Cache on (first run, incl. warm-up) | 101 s | 1.63x |
| Cache on (steady state) | **85 s** | **1.94x** |

Workflow shape:

```
UNETLoader → Krea2CachePatch → LoraLoaderModelOnly
   → KSamplerAdvanced(7 steps) → LatentUpscaleBy(1.5x) → KSamplerAdvanced(6 steps)
   → UltimateSDUpscale(3 steps x 2 tiles)
```

**Why the gain grows when VRAM is tight.** 8 GB cannot hold the 12.5 GB Krea2, so ComfyUI constantly shuffles weights layer by layer. When a step is skipped, those 28 blocks' weights are **never loaded at all** — you save not just FLOPs but also the weight traffic over PCIe. Hence small-VRAM machines often benefit more than large-VRAM ones.

> Your actual speedup depends on step count, resolution, memory pressure, and how aggressive your `rel_l1_thresh` is. **More sampling steps usually means more reusable steps.**

---

## Installation

### Option A: git clone

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_NAME/ComfyUI-Krea2-Accel.git
```

Resulting layout:

```
ComfyUI/custom_nodes/ComfyUI-Krea2-Accel/
    __init__.py
    nodes.py
    krea2_cache.py
    krea2_fp8.py
```

### Option B: copy files

Copy the 4 `.py` files into `custom_nodes/<any folder name>/`. **No `pip install`** — the only dependencies are `torch` and `numpy`, both already present in ComfyUI.

### Verify

**You must restart ComfyUI.** The startup log should contain:

```
[ComfyUI-Krea2-Accel] v0.3.0 loaded OK — nodes: Krea2CachePatch, Krea2CacheStats, ...
```

No such line = the package failed to import; the traceback is printed right above it.

> ComfyUI **silently swallows import errors** from custom nodes and only says "some nodes failed to load", which is exactly why this package prints its own banner.

---

## Quick start

**Leave your model loader alone.** Keep your existing `UNETLoader` / `Load Diffusion Model` and just insert **Krea2 Cache Patch** between it and the sampler:

```
UNETLoader (Krea2-MuseByStable_v30Turbo_fp8.safetensors)
      │  MODEL
      ▼
Krea2 Cache Patch ──MODEL──► KSampler / KSamplerAdvanced ──► VAEDecode ──► SaveImage
      │
      └──MODEL──► Krea2 Cache Stats   (optional, to see the hit rate)
```

Recommended Turbo sampling settings: **steps = 8, cfg = 1.0, sampler = euler, scheduler = simple**.

That's it — one node, wired in. Everything after that is threshold tuning.

---

## Parameters

### Krea2 Cache Patch (TeaCache)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `model` | — | MODEL input |
| `enable` | `true` | When off, **the native forward runs untouched** — no speedup, no risk, ideal for A/B comparison |
| `rel_l1_thresh` | `0.30` | **The main knob.** Higher → more skipping → faster, but more quality risk |
| `start_percent` / `end_percent` | `0.0` / `1.0` | Only enable the cache within this sampling-progress range. Use `0.1` / `0.9` to protect the first/last steps |
| `max_skip_steps` | `3` | Cap on consecutive reuses, to prevent quality collapse from never recomputing |
| `cache_device` | `default` | `default` / `gpu` = keep the residual on the model's device; `cpu` = store in RAM (saves VRAM, slightly slower) |
| `coefficients` | empty | Advanced: 5 comma-separated coefficients of a quartic polynomial, e.g. `-450,280,-45,3.2,-0.02`. **Leave empty for the raw relative-L1 mode (recommended)** |

### Krea2 Cache Stats

```
Krea2 Cache (TeaCache)
  state: enabled
  model forwards: 7
  reuses (skipped): 2  (28.6%)
  theoretical speedup: 1.40x
  threshold rel_l1_thresh: 0.300
  active range: 0.00 ~ 1.00
  max consecutive skips: 3
  per-step relative L1: 0.14 0.16 0.19 0.23 0.30 0.42
  min 0.140 / median 0.230 / max 0.420
  suggested threshold: 0.46 for more skipping, 0.23 for quality
```

> **Stats settle at the *start of the next* sampling run.**
> The first image shows empty numbers — run a second image to see the first one's data.
> The trigger is "timestep went back up = a new sampling run began".

---

## Tuning: read the console, don't guess

After one image, the ComfyUI console prints:

```
[Krea2-Accel] last sampling: 7 model forwards, 2 reuses (28.6%)
[Krea2-Accel]   per-step relative L1: 0.14 0.16 0.19 0.23 0.30 0.42  | min 0.14 / median 0.23 / max 0.42  -> for more skipping set rel_l1_thresh near 0.46
```

**Key point: the criterion is cumulative.**
After distance `d₁`, one step is skipped if `d₁ < threshold`; the next step compares `d₁ + d₂` against the threshold. So **threshold ≈ the sum of two or three step distances you're willing to accept**, not a single-step distance.

| Goal | How to pick the threshold |
|------|---------------------------|
| Conservative (near-lossless) | median distance x 1 |
| **Balanced (recommended)** | median distance x 2 (the console already computes it for you) |
| Aggressive | median distance x 3, and raise `max_skip_steps` to 3~4 |

**Quick diagnosis:**

- **0 reuses** → threshold too low. Raise it.
- **Quality degraded** → lower it; or set `start_percent=0.1` / `end_percent=0.9`; or drop `max_skip_steps` to 1.
- **High reuse rate but little speedup** → the bottleneck isn't the DiT (could be VAE decode, LoRA, upscaling, or weight traffic).

---

## How it works (source-level)

### Krea 2's structure

Krea 2 is a **single-stream DiT** (`comfy.ldm.krea2.model.SingleStreamDiT`): text tokens and image tokens are concatenated into one sequence `combined`, which flows through `self.blocks` (28 `SingleStreamBlock`s, width 6144):

```python
combined = torch.cat((context, img), dim=1)
for i, block in enumerate(self.blocks):
    combined = block(combined, tvec, freqs, None,
                     timestep_zero_index=..., transformer_options=...)
```

Every block is **residual** (returns `x + Δ`) — precisely the property that makes caching valid.

### What the cache does

- **Cache hit**: block 0 returns `x + R` directly (where `R` is the cached total residual for the entire block stack), and blocks 1..N-1 degrade to identity → **all 28 blocks' attention + MLP are skipped**.
- **Cache miss**: all 28 blocks run normally, and `R` is refreshed at the last block.

### The hit criterion

Take block 0's **AdaLN modulation input**

```
m = (1 + prescale) * prenorm(x) + preshift
```

and compute its **relative L1 distance** to the previous step's `m`, accumulate it, and compare against `rel_l1_thresh` — exactly TeaCache's approach. This quantity is stable across adjacent steps, and costs only one RMSNorm plus a scalar reduction, so it is essentially free.

### Why this patching approach is safe

This is where most of the care went, and the reason v0.3.0 exists:

- **`_forward` is not rewritten** and no tensor shapes are touched — shapes remain ComfyUI's responsibility.
- `_forward` only gets a thin shell that **records the current timestep**, used to derive sampling progress for `start_percent` / `end_percent`.
- Each block's `forward` is wrapped by a **closure** that captures the **already-bound original method** (`orig = block.forward`) — it **never relies on Python's automatic `self` binding**.
  > The previous version assigned a plain function to an instance attribute; Python then no longer injects `self`, so `self` became the input Tensor and you got `'Tensor' object has no attribute '_krea2_cfg'`. Fixed in v0.3.0.
- **Any exception** in the criterion computation falls back to "compute normally" — the worst case is one unaccelerated image, never a broken graph.

---

## Relation to other accelerators

| Approach | Idea | vs this package |
|----------|------|-----------------|
| **This package (TeaCache residual cache)** | Skip the whole block stack | — |
| [TeaCache](https://github.com/AliyunContainerService/TeaCache) | Same idea, multi-model | The origin of the cache idea (Apache-2.0) |
| **CacheDiT** | Another implementation of a similar idea | Stackable, but pick one |
| **SageAttention** (KJNodes) | Faster attention kernel | **Stackable** — orthogonal (one computes less, one computes faster) |
| `torch.compile` / Compile Model | Graph capture + kernel fusion | Uncertain interaction; pick one |

Recommended combo: **this package + SageAttention** — one does less work, one does the work faster.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| No `[ComfyUI-Krea2-Accel] ... loaded OK` in the log | Import failed. Read the traceback **above** that point. Most common cause: missing files (all 4 `.py` files are required) |
| `SingleStreamDiT not found in this MODEL` | The node is connected to a non-krea2 model, or no loader is upstream. Connect a `UNETLoader` with a krea2 checkpoint first |
| `block N is not a SingleStreamBlock` | Not a single-stream DiT; unsupported |
| `coefficients invalid` | Must be exactly 5 comma-separated numbers. **Leave it empty** if unsure |
| 0% reuse | `rel_l1_thresh` too low. Use the console's suggested value |
| Blurry / lost detail | Lower the threshold; or `start_percent=0.1`; or `max_skip_steps=1` |
| Faster in theory, not in practice | Bottleneck is elsewhere (VAE, upscale, weight traffic). Check the theoretical speedup in `Krea2 Cache Stats` |
| Error mentions `_krea2_cfg` | You're still running v0.2.x files on disk. Fully overwrite with v0.3.0 |
| VRAM got tighter | Set `cache_device` to `cpu` (residual in RAM, slightly slower) |

---

## FAQ

**Q: Do I need to retrain or convert the model?**
No. Only the inference call path changes; not a single byte of the weights is modified.

**Q: Does it change the output?**
Slightly — this is a training-free, **mildly lossy** inference speedup. The more aggressive the threshold, the larger the difference. Do a visual check for anything important.

**Q: Does it work with LoRA?**
Yes. The benchmark workflow is literally `Krea2CachePatch → LoraLoaderModelOnly`.

**Q: Is it useful with very few steps (e.g. 4)?**
Yes, but the headroom is small — there are fewer steps to reuse. ~8 steps is a good target for Turbo models.

**Q: Why are the stats empty on the first image?**
Stats settle when a new sampling run is detected (timestep goes back up), so the first image's numbers appear when you start the second one.

**Q: Why no Krea 2-specific polynomial coefficients?**
The community currently only has fitted coefficients for models like qwen-image. The default **raw relative-L1 mode** is robust but needs manual threshold tuning; advanced users can fit their own and pass them via `coefficients`.

---

## Limitations

- Training-free but **mildly lossy** — verify visually for important renders.
- Interaction with `torch.compile` / Compile Model is uncertain; pick one.
- Supports only the `krea2` **single-stream DiT**; cloud API nodes are out of scope.
- Multi-stage sampling (e.g. hi-res fix) re-calibrates progress at the start of each new stage.

---

## License & credits

MIT — see [LICENSE](LICENSE).

The caching idea is adapted from [TeaCache](https://github.com/AliyunContainerService/TeaCache) (Apache-2.0, ali-vilab / ComfyUI-TeaCache). Thanks to the original authors.

Krea 2 model weights are property of Krea / Comfy-Org. This repository contains no weights.

---

*中文版: [README.md](README.md)*
