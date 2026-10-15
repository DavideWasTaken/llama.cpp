# Hybrid MoE cache

This fork extends the MoE expert cache of llama.cpp (`--moe-cache-mib`, [#29887](https://github.com/ggml-org/llama.cpp/pull/29887)) for machines where the link to the GPU is slow, such as PCIe 3.0.

## The problem

With `--n-cpu-moe`, the experts of some layers stay in system RAM and the CPU computes them. The MoE cache keeps the most recently used of these experts in VRAM and computes the whole layer on the GPU, uploading every expert that is missing from the cache before the layer runs.

On a fast link this is a large gain. On a slow link the uploads cost more than they save: on the [test system](#test-system) (RTX 5060 Ti on PCIe 3.0 x8, Qwen3.6-35B-A3B), the cache makes generation slower than no cache at all (24.9 vs 33.9 t/s, ~167 MiB uploaded per token). The same was reported in the PR for an RTX 2080 Ti on PCIe 3.0 with the same model.

## What the hybrid mode does

`--moe-cache-hybrid` changes how a cached layer runs:

1. **Hits on the GPU, misses on the CPU.** The experts that are in the cache are computed on the GPU, the others on the CPU from system RAM, and the two partial results are added. Nothing is uploaded on the critical path.
2. **CPU and GPU at the same time.** The scheduler copies the inputs of the CPU split before it launches the GPU split, so the CPU computes the misses while the GPU computes the hits.
3. **Frequency-based cache.** Each expert has a use count that decays at each step. A missed expert enters the cache only if it scores higher than the cached expert it would replace, within an upload budget per token (`--moe-cache-upload-mib`, default 16).
4. **Warm start.** `--moe-cache-profile FILE` saves the scores at exit and, at the next start, fills the cache with the experts that scored highest.

How the result stays exact: each expert of each token has a non-zero output in only one of the two branches. On the GPU, a missed expert reads a zero slot of the cache (one per position in the token, since `mul_mat_id` does not allow the same expert twice in a token). On the CPU, a cached expert has the id -1, which `mul_mat_id` now skips with a zero output.

## Usage

```bash
llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -ngl 99 -fa on --n-cpu-moe 40 \
  --moe-cache-mib 3200 --moe-cache-hybrid --moe-cache-profile qwen36.moeprof
```

- Put all the experts in RAM (`--n-cpu-moe` = number of layers) and give the VRAM they leave free to the cache.
- Stop the server with Ctrl+C so that the profile is saved.
- The hybrid path is used for batches of up to 16 tokens (generation, speculative decoding). Larger batches (prompt processing) use the regular offload path.
- Layers with per-expert biases or scales, LoRA adapters, or weights applied before the FFN do not use the cache in hybrid mode.

## Test system

| Part | |
|------|-|
| CPU | Intel Core i7-10700K, 8 cores / 16 threads, AVX2 (no AVX-512) |
| RAM | 2x16 GB DDR4-2666 CL16, dual channel (~29 GB/s measured read bandwidth) |
| GPU | NVIDIA GeForce RTX 5060 Ti 8 GB (Blackwell, sm_120), driver 610.88 |
| Link | PCIe 3.0 x8 (~7 GB/s), the limit of a Comet Lake CPU with an x8 card |
| Motherboard | ASUS PRIME Z590-P |
| OS / toolchain | Windows 11 Pro (build 26300), CUDA 12.8, MSVC 19.44 (Visual Studio 2022 Build Tools) |
| Build | `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=ON`, based on llama.cpp `23deec0` (2026-10-15) |
| Model | Qwen3.6-35B-A3B UD-Q4_K_M with the MTP layer (unsloth GGUF, 21.1 GB): 40 MoE layers, 256 experts, 8 used per token |
| Common flags | `-ngl 99 -fa on -t 8 -lm none -ctk q8_0 -ctv q8_0` |

Why this machine is the case for the hybrid mode: the CPU reads an expert from RAM about 4x faster than the link can upload it to the GPU, so a missed expert costs less on the CPU than in an upload.

## Results

The baseline is the fastest configuration without the cache (`--n-cpu-moe 33`). Same VRAM in all the runs.

| Generation (t/s)                                   | baseline | upstream cache | hybrid | hybrid + profile |
|----------------------------------------------------|---------:|---------------:|-------:|-----------------:|
| 1024 tokens, no speculative decoding               | 33.5     | 25.6           | 39.8   | **43.0 (+26%)**  |
| llama-server, MTP draft (2 runs, 3 prompts each)   | 38.0     | -              | -      | **49.7 (+31%)**  |

| Other measurements                    | baseline | hybrid + profile |
|---------------------------------------|---------:|-----------------:|
| Prompt processing, 2048 tokens (t/s)  | 361      | 304 (-16%)       |
| Cache hit rate                        | -        | 71%              |
| Perplexity, ubatch 16                 | 5.6560   | 5.6388           |

The perplexity difference is within the error (+/- 0.43): with the hybrid cache, some experts run on the GPU instead of the CPU. With the same settings, the scheduler change gives the same output bit for bit. In the MTP test the profile came from a different text than the measured prompts.

The upload budget matters: with the frequency policy, 8 MiB per token gives 60% hits, 16 MiB 66%, 32 MiB 70%, and 16 MiB is the fastest; on a slow link, more uploads cost more than the extra hits save.

## Limitations

- Tested with one model, one GPU, on Windows with CUDA.
- Prompt processing is slower than the baseline, since all the experts are in RAM.
- The CUDA `mul_mat_id` does not accept negative ids, so the CPU branch must stay on the CPU: the scheduler must not offload it, which limits the hybrid path to batches of 16 tokens.

## Changes

| File | Change |
|------|--------|
| `ggml/src/ggml-cpu/ggml-cpu.c` | `mul_mat_id`: a negative id skips the expert, its output row is zero |
| `ggml/src/ggml-backend.cpp` | prefetch of the inputs of a CPU split before the device split that precedes it |
| `src/llama-moe-cache.*` | hybrid mode, zero slots, frequency policy, profile |
| `src/llama-graph.cpp` | `build_moe_ffn` builds the experts twice (cache slots on the GPU, ids on the CPU) and adds them |
| `include/llama.h`, `common/` | `--moe-cache-hybrid`, `--moe-cache-upload-mib`, `--moe-cache-profile` |
| `tests/test-mul-mat-id-skip.cpp` | CPU test of the negative ids |
| `scripts/moe-cache-hybrid-check.ps1` | GPU check: test, perplexity, determinism, speed |
| `.github/workflows/moe-cache-hybrid-sync.yml` | weekly merge of upstream into a PR, CPU build and test |

## Keeping up with upstream

The `moe-cache-hybrid-sync` workflow merges upstream every week into a `sync/<date>-<sha>` branch. A clean merge is built and tested on CPU and opens a PR to `moe-cache-hybrid`; a conflict opens an issue. GitHub runners have no GPU, so before merging the PR run on a CUDA machine:

```powershell
.\scripts\moe-cache-hybrid-check.ps1 -Model C:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```
