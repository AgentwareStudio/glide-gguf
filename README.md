# GLiDE · Q4_K_S GGUF

Code and documentation for [Agentware/GLiDE-GGUF](https://huggingface.co/Agentware/GLiDE-GGUF). Model weights and tokenizer assets are hosted on Hugging Face.

A mixed-precision GGUF conversion of [Fastino GLiDE](https://huggingface.co/fastino/GLiDE), with its dedicated native decision-model runtime.

GLiDE selects among structured options using a 255-way readout. It is not an autoregressive chat model. This version supports text and uses full-question noncausal attention, with state reset between questions.

## Model

| Item | Value |
|---|---|
| Source | `fastino/GLiDE` · `1a9a7ed94379c2ea6f324176c9bc22fb9b7b7d60` |
| Parameters | 26.09B |
| File | `glide-Q4_K_S-q8emb-bf16restore.gguf` |
| Size | 15.65 GB / 14.57 GiB |
| Precision | 468 Q4_K, 12 Q5_K, 1 Q8_0, 17 BF16 and 353 F32 tensors |
| Readout | Original 255-way BF16 classifier |
| Runtime | `GlideGGUF` + `glide-logits-fa` |
| Context | 8,192 tokens in the evaluated configuration |

The source config identifies `Qwen3_5ForConditionalGeneration`; Fastino's original documentation describes a modified Qwen3.8-27B. Both are preserved in the included upstream files.

## Evaluation

All three versions completed the same 29,771 requests across five frozen public0.3 tasks. Results are percentages; BANKING77 and CLINC use macro-F1, the others accuracy.

| Task | Requests | BF16 | HF compact | This GGUF |
|---|---:|---:|---:|---:|
| BANKING77 | 3,080 | 91.908020 | 91.680961 | 91.866078 |
| CLINC150+OOS | 5,500 | 95.888608 | 95.946450 | 95.816106 |
| MMLU-Pro | 12,032 | 77.925532 | 77.468418 | 77.518285 |
| BBH fixed-option | 5,507 | 92.954422 | 92.482295 | 92.881787 |
| When2Call MCQ | 3,652 | 91.730559 | 91.456736 | 91.785323 |

The largest score drop from BF16 was 0.407 percentage points on MMLU-Pro. These are five task results, not an official aggregate Decision Index score. HF compact is a separate format, not this GGUF.

The evaluation uses zero-shot fixed-choice MMLU-Pro, 23 adapted fixed-option BBH tasks, and When2Call MCQ. The full pools include earlier recipe-selection examples; the quantization recipe was frozen before this full run. See [results.csv](evaluation/results.csv) and [metric definitions](evaluation/source/METRIC_SPEC.json).

## Running

See [RUNNING.md](RUNNING.md) for build commands and a Python example. The package includes the exact native sources, tokenizer, prompt construction, configuration and postprocessing used by the historical implementation.

The evaluated L4 profile uses Flash Attention, CPU token embeddings, four threads, and context/batch/ubatch of 8,192. The runtime depends on llama.cpp commit `8e5a58df87ab59dfceb5b961ae26691e39f30c24`. A standard chat GGUF runner cannot replace this decision-model wrapper.

## Current limitations

- The CPU/L4 probability-consistency check failed: some probability differences exceeded the original 0.01 mean / 0.03 maximum thresholds. The original deployment gate remains failed; see [gate results](evaluation/failed_gate_excerpt.json).
- Observed L4 GPU use reached 17,042 MiB with token embeddings on CPU. This configuration is not verified to fit a 16 GiB GPU; the reading is not a certified peak.
- Images, partial GPU-layer offload and oversized prompts are rejected. Questions run sequentially.
- This package preserves the historical results. Preparation checked file hashes and tokenizer/prompt construction, but did not rebuild the native executable or rerun GPU inference.

## License

[Fastino Community License](LICENSE), with upstream notices in [NOTICE](NOTICE) and [licenses/](licenses/).

Commercial use requires the attribution **Built with Fastino GLiDE**. Production commercial use by entities with consolidated annual revenue above USD 10 million requires a separate Fastino agreement. See the full license for definitions and conditions.

## Integrity

Weight SHA256: `c8af13959de7081a83421a615af1967932e69aac9cb43e7336fc28da784f4636`.

[SHA256SUMS](SHA256SUMS) covers the release files. The tokenizer and upstream source files are also checked by the wrapper against their pinned hashes.
