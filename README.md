# Sovereign AI Engine

[![CI](https://github.com/madhusudanpatnaik/sovereign-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/madhusudanpatnaik/sovereign-llm/actions/workflows/ci.yml)

Host and optimize a small open-weight model on your own hardware: fine-tune it with QLoRA,
quantize it, serve it with continuous batching and prefix caching — and **measure** the gap to a
frontier API on the same questions, with a gate that refuses an optimization that costs more
accuracy than it claims to.

> **What ran, and what did not.** The build machine is a 16 GB Apple M2 with no CUDA GPU, no API
> key, and too little free disk for a 4B checkpoint. So:
> **ran for real** — PubMedQA itself (downloaded, official split, label balance), the memory model
> against Qwen3-4B's real config, a complete fine-tune → generate → score loop on a tiny model on
> CPU (real transformers + peft + Trainer), the streaming client and load generator against a real
> HTTP server with known timing, and the Claude client through the real `anthropic` SDK over a
> mock transport; the compress step's recipe and calibration data through stand-ins for
> llm-compressor. **Not run** — QLoRA on the 4B model (bitsandbytes needs CUDA), vLLM (Linux +
> CUDA only), llm-compressor itself (not installed on it), and every accuracy number for a real
> model. The protocol below produces them on a
> GPU box; nothing in this README claims a result it did not measure.

```bash
uv venv --python 3.12 && VIRTUAL_ENV=.venv uv pip install -e ".[dev]"
uv run pytest -q                    # 64 passed, offline (one test trains a tiny model, ~20 s)
uv run sovereign data               # PubMedQA splits (downloads ~3 MB)
uv run sovereign memory --gpu-gib 24          # fetches config.json only
uv run sovereign smoke              # CPU fine-tune -> generate -> score on a tiny model
```

## Run it on another machine

Building and testing never downloads a model. Each machine installs from the
requirements files and fetches its own models; nothing large lives in the repository.

```bash
python3.12 -m venv .venv && source .venv/bin/activate    # or: uv venv --python 3.12
pip install -r requirements.txt                          # core: the offline demo and the evaluation
pip install -r requirements-train.txt                    # torch, transformers, peft, bitsandbytes (QLoRA)
pip install -r requirements-serve.txt                    # vLLM 0.31 + its bitsandbytes plugin
pip install -r requirements-compress.txt                 # llm-compressor: fp8 / w4a16 serving checkpoints
pip install -r requirements-frontier.txt                 # the Claude reference
pip install -r requirements-dev.txt                      # tests, mypy, ruff, CPU torch
cp .env.example .env    # fill in the keys you need, then:  set -a; . ./.env; set +a
sovereign download --list           # Qwen3-4B ~8 GB, PubMedQA ~3 MB
sovereign download                  # into the Hugging Face cache, where vLLM also looks
pytest -q && sovereign smoke
```

Training with NF4 and serving with vLLM need **Linux and a CUDA GPU** (24 GB is enough for the 4B model; see the memory table). The `train` and `serve` requirement files skip the CUDA-only packages on other platforms, so a laptop can still install them and run the tests and the CPU smoke run.


## The task: PubMedQA, the official way

`qiaojin/PubMedQA` `pqa_labeled`: 1,000 expert-labelled yes / no / maybe questions about PubMed
abstracts. Models see the question and the abstract **without its conclusion** (the
"reasoning-required" setting every published number uses).

| split | n | yes | no | maybe | |
|---|---|---|---|---|---|
| test | 500 | 276 | 169 | 55 | the **official** test set: PMIDs pinned to the PubMedQA repo by commit and sha256 |
| train | 450 | 245 | 154 | 51 | the other 500, split by a salted hash of the PMID |
| val | 50 | 31 | 15 | 4 | |

Majority-class accuracy on test is **55.2%** (macro-F1 23.7) — the floor. Published reference
points on this exact test set (reasoning-required): one human annotator **78.0%**; frontier models
roughly 76–82% (GPT-4o 78.4, o3 80.0 in the MedGemma technical report); 4B-class models roughly
68–74% (Gemma 3 4B 68.4, MedGemma 4B 73.4). No primary source reports Qwen3-4B on it — this
project's run would be the measurement.

Three traps the data module is built around, each with a test:
- `long_answer` is the abstract's conclusion — the answer in prose. Never in a prompt.
- In `pqa_labeled` the `context` dict also carries `reasoning_required_pred` /
  `reasoning_free_pred`: the two **human annotators' own answers** (78% / 90% accurate). Only
  `context["contexts"]` is read; a test plants markers in both fields and asserts neither reaches a
  prompt.
- `pqa_artificial` (211k "free" training examples) is 92.8% "yes" with **no "maybe" at all** — its
  labels come from whether the title's verb is negated. Fine-tuning on it naively teaches the model
  never to say maybe, so it is not used.

## The pieces

```
PubMedQA ─► data (official split, leak-free prompts)
              │
              ├─► train: QLoRA (NF4 base, LoRA on all linear layers, loss on the answer only)
              │            └─► adapter ──► merge (optional)
              │
              ├─► merge ──► compress: fp8 (no calibration) or w4a16 (GPTQ on train prompts)
              ├─► serve: vLLM — continuous batching, prefix caching, LoRA by name, NF4 in flight,
              │          or the compressed checkpoint (no quantization flag: it carries its config)
              │            └─► clients.OpenAICompatibleClient (streams: TTFT, per-token timing)
              ├─► clients.AnthropicClient (frontier reference, same prompts, same parser)
              │
              └─► evaluate (strict parsing, accuracy, macro-F1, paired bootstrap)
                    └─► GATE: promote the optimized model only if the drop is ruled out
    bench: closed-loop concurrency sweep / open-loop Poisson load / shared-prefix workload
    quant: exact weight memory per precision, KV cache per token, sequences that fit
```

- **`train.py` — QLoRA.** NF4 weights with double quantization and bf16 compute;
  `prepare_model_for_kbit_training`; LoRA r=16, α=32 on every linear layer (peft's `all-linear`,
  which is also right for models with fused projections like Phi-4-mini). **Loss only on the
  answer**: the conversation is tokenized up to the generation prompt and in full, the shared
  prefix is labelled -100, and a chat template that does not render the prompt as an exact prefix
  is an error rather than silently mislabelled data. Too-long examples are dropped and counted, not
  truncated. Written for transformers 5 (`dtype=`, `processing_class=`, float `warmup_steps`).
- **`quant.py` — memory, computed.** The model's config is instantiated on PyTorch's `meta`
  device, so the parameter count is exact for any architecture — tests pin Qwen3-4B
  (4,022,468,096), Qwen2.5-3B (3,085,938,688) and Phi-4-mini (3,836,021,760) to their published
  totals. bitsandbytes quantizes `nn.Linear` weights; embeddings, norms, biases and the (tied) output
  head stay 16-bit, and the estimator counts them that way.
- **`serve.py` — vLLM 0.31.** Builds the `vllm serve` command; vLLM does the continuous batching
  and prefix caching, this project measures them. Written against 0.31's actual flags: prefix
  caching is on by default, so the cache-off arm passes `--no-enable-prefix-caching`;
  `--enable-prompt-tokens-details` makes every response report its cached prompt tokens; NF4 serving
  needs `vllm-bnb-plugin` since vLLM 0.28; `--max-model-len` is mandatory (Qwen3-4B's native 262k
  context would want ~36 GiB of KV cache for one sequence); the LoRA rank must be one vLLM accepts.
- **`compress.py` — the serving arm.** The merged bf16 model quantized by llm-compressor:
  `FP8_DYNAMIC` (8-bit weights with a scale per channel, activations quantized on the fly, no
  calibration data) or `W4A16` by GPTQ (4-bit weights in groups of 128), calibrated on PubMedQA
  *training* prompts rendered through the chat template — never a test question. The output head
  stays 16-bit. `sovereign serve --precision fp8|w4a16` refuses a checkpoint without a
  quantization config, since vLLM would silently serve it unquantized.
- **`clients.py`** — one async interface: an OpenAI-compatible streaming client (TTFT and chunk
  timing per request; token counts and cache hits from the final usage chunk; a mid-stream error
  inside an HTTP 200 raises), an in-process `transformers` client, and Claude as the frontier
  reference (`claude-opus-5-5`, server-side refusal fallback; a refusal is scored as an invalid
  answer rather than aborting a 500-question run). `choice_constraint` asks vLLM's structured
  outputs to decode the reply as exactly yes / no / maybe (`sovereign eval --constrained`).
- **`evaluate.py`** — strict parsing (a reply must *start* with yes/no/maybe; anything else is
  invalid and wrong, never coerced), accuracy, macro-F1 (the labels are imbalanced), confusion with
  an "invalid" column, and a **paired bootstrap** for differences between systems on the same
  questions: with n = 500 a single system's 95% interval is about ±3.8 points, and only pairing
  makes a few-point gap resolvable.
- **`gate.py` — the promotion gate.** A candidate (say, the fine-tuned model in NF4) is promoted
  over a baseline (the same in bf16) only if both answered the same questions, the paired 95% CI's
  lower bound clears `-max_accuracy_drop`, the invalid-output rate is under a cap, and optionally
  p95 latency is under a cap. Every refusal says which rule failed by how much.
- **`bench.py`** — closed loop (N users; throughput vs concurrency — continuous batching's
  signature is tokens/s rising while time-per-output-token degrades slowly) and open loop (Poisson
  arrivals; latency under offered load, including queueing, which closed loops hide). A few-shot
  workload puts the same solved examples in front of every question — the shared prefix prefix
  caching reuses. Percentiles carry their n; a p99 from under 100 requests is flagged.

## Measured on the build machine

**Memory** — `sovereign memory`, from Qwen3-4B-Instruct-2507's real config (weights at each
precision; KV cache at 144 KiB/token; 1.5 GiB overhead; 4,096-token sequences):

| GPU | precision | weights | KV budget | concurrent 4k sequences |
|---|---|---|---|---|
| 24 GB | bf16 | 7.49 GiB | 12.61 GiB | 22 |
| 24 GB | int8 | 4.11 GiB | 15.99 GiB | 28 |
| 24 GB | **nf4** | **2.47 GiB** | 17.63 GiB | **31** (+41%) |
| 24 GB | fp8 | 4.11 GiB | 15.99 GiB | 28 |
| 24 GB | w4a16 | 2.47 GiB | 17.63 GiB | 31 |
| 80 GB | bf16 | 7.49 GiB | 63.01 GiB | 112 |
| 80 GB | nf4 | 2.47 GiB | 68.03 GiB | 120 (+7%) |

Quantizing a 4B model's weights buys a lot of batching headroom on a 24 GB card and almost none on
an 80 GB one: its payoff there is fitting a bigger model, not serving this one faster. FP8 and
W4A16 take the same memory as int8 and NF4; what they change is speed — dedicated vLLM kernels
instead of bitsandbytes dequantizing on the fly — which only a GPU run can measure. (The
estimator omits activations and the CUDA context beyond a fixed overhead — measure those on the GPU.)

**The fine-tuning loop** — `sovereign smoke`: a 2-layer Qwen2-architecture model built from a
config (nothing downloaded), a word-level tokenizer with Qwen's ChatML template, LoRA via
`train.build_trainer`: 16,384 trainable of 94,720 parameters; loss 3.30 → 0.003; frozen weights
bit-identical after training; 100% on 30 held-out toy questions through `generate` and the
real parser. This proves the plumbing — masking, collator, adapter wiring, trainer, generation,
scoring — not anything about PubMedQA.

**The measuring instruments** — the benchmark is tested against a real HTTP server with known
timing (prefill delay, per-token delay, a fixed number of concurrent slots): it recovers the decode
rate, shows throughput scaling until the slots saturate and TTFT jumping when requests queue
beyond them, counts failed requests, and computes the prefix-cache hit rate from usage chunks.

## Run it for real (a Linux box with one CUDA GPU)

```bash
pip install -r requirements-train.txt -r requirements-serve.txt -r requirements-frontier.txt
cp .env.example .env && set -a && . ./.env && set +a    # ANTHROPIC_API_KEY for the reference
sovereign download                               # Qwen3-4B + PubMedQA into the HF cache

sovereign eval --anthropic --out preds/claude.jsonl
sovereign serve --exec &                         # base model, bf16
sovereign eval --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-4B-Instruct-2507 --out preds/base-bf16.jsonl

sovereign train --precision nf4 --output-dir runs/qlora
sovereign serve --adapter pubmedqa=runs/qlora --exec &                    # LoRA on bf16
sovereign eval --base-url http://127.0.0.1:8000/v1 --model pubmedqa --out preds/qlora-bf16.jsonl
sovereign serve --adapter pubmedqa=runs/qlora --precision nf4 --exec &    # LoRA on NF4
sovereign eval --base-url http://127.0.0.1:8000/v1 --model pubmedqa --out preds/qlora-nf4.jsonl

sovereign merge runs/qlora runs/merged                            # the throughput arm:
sovereign serve --model runs/merged --exec &                      # merged, bf16 (the baseline)
sovereign eval --base-url http://127.0.0.1:8000/v1 --model runs/merged --out preds/merged-bf16.jsonl
sovereign compress runs/merged runs/merged-fp8 --scheme fp8       # needs requirements-compress.txt
sovereign serve --model runs/merged-fp8 --precision fp8 --exec &
sovereign eval --base-url http://127.0.0.1:8000/v1 --model runs/merged-fp8 --out preds/merged-fp8.jsonl
sovereign eval ... --model runs/merged-fp8 --constrained --out preds/merged-fp8-constrained.jsonl

sovereign compare preds/*.jsonl --reference claude        # the gap, with paired 95% CIs
sovereign gate --candidate preds/qlora-nf4.jsonl --baseline preds/qlora-bf16.jsonl
sovereign gate --candidate preds/merged-fp8.jsonl --baseline preds/merged-bf16.jsonl

sovereign bench --base-url http://127.0.0.1:8000/v1 --model pubmedqa \
    --concurrency 1,4,16,64 --max-tokens 128 --fixed-output       # continuous batching
sovereign bench ... --workload few-shot --shots 8 --concurrency 16   # then restart with
                                                                     # --no-prefix-caching and repeat
```

### Review before publishing (2026-10-17)

A full read of the code found no correctness bug in the pipeline; it found rough edges at the
command line, now fixed with a test each: `serve --exec` without vLLM installed died with a
`FileNotFoundError` (it now says to install `requirements-serve.txt`, on Linux with a CUDA GPU);
`--adapter` without `name=path` raised a Python error; a missing training, compression or frontier
library raised a traceback (each now names its requirements file). Tests are now type-checked
alongside the source.

## Why it's built this way — checked against the literature (2026-10)

| Decision | Evidence | Verdict |
|---|---|---|
| **PubMedQA, reasoning-required, the official 500-question test set; conclusion and annotator fields never in a prompt** | The setting every published number uses (Jin et al., 2019); the leak fields are documented in the dataset itself | **Holds** |
| **QLoRA: NF4 base, double quantization, LoRA on every linear layer, loss on the answer only** | QLoRA (Dettmers et al., 2023) matches 16-bit fine-tuning only when LoRA covers all linear layers; NF4 plus double quantization is its memory recipe. With 450 training questions, LoRA's bias toward learning less and forgetting less (Biderman et al., 2024) is the right side of the trade | **Holds** |
| **Serving the NF4 model in flight with vLLM's bitsandbytes plugin** | vLLM's own quantization guide recommends FP8 where the GPU supports it, and AWQ/GPTQ (Lin et al., 2024; Frantar et al., 2023) for 4-bit, with dedicated kernels; bitsandbytes is positioned for fine-tuning and quality, not serving throughput | **Fixed:** the throughput arm now exists — merge, `sovereign compress` to FP8 (or W4A16 by GPTQ, which has a stable llm-compressor recipe where AWQ's changed between versions), serve the checkpoint, and gate it against the bf16 merge. NF4 in flight stays as the arm that serves exactly the weights QLoRA trained against |
| **Continuous batching and prefix caching measured, not assumed; closed- and open-loop load** | Continuous batching (Orca, Yu et al., 2022) and paged KV memory (vLLM, Kwon et al., 2023) are why throughput rises with concurrency; a closed loop hides queueing delay, an open loop with Poisson arrivals shows it | **Holds** |
| **Paired bootstrap on the same 500 questions; a promotion gate on the CI's lower bound** | Paired resampling (Koehn, 2004) is the standard significance test for comparing systems on one test set; a single system's 95% interval at n = 500 is about ±3.8 points, so only pairing resolves a few-point gap (Dror et al., 2018) | **Holds** |
| **Strict answer parsing: a reply must start with yes / no / maybe** | Measures instruction following as well as knowledge. vLLM's structured outputs (a choice constraint) would make every reply valid, which removes a failure mode rather than measuring it | **Holds, and now both are measurable:** `--constrained` adds the choice-constrained arm, so format failures and knowledge failures can be told apart |

**Done after this review (2026-10):** the merged-FP8 / W4A16 serving arm with a gate against bf16,
and the choice-constrained decoding arm. **Still open:** running them, and the frontier
comparison — all need the GPU box and a key.

**References**
- Jin et al. *PubMedQA: A Dataset for Biomedical Research Question Answering.* EMNLP-IJCNLP 2019.
  [arXiv:1909.06146](https://arxiv.org/abs/1909.06146)
- Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
  [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
  [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Biderman et al. *LoRA Learns Less and Forgets Less.* TMLR 2024.
  [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)
- Lin et al. *AWQ: Activation-aware Weight Quantization.* MLSys 2024.
  [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- Frantar et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained
  Transformers.* ICLR 2023. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- vLLM documentation: quantization. [docs.vllm.ai](https://docs.vllm.ai/en/latest/features/quantization/)
- Yu et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI 2022.
  [usenix.org](https://www.usenix.org/conference/osdi22/presentation/yu)
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.*
  SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Koehn. *Statistical Significance Tests for Machine Translation Evaluation.* EMNLP 2004.
  [aclanthology.org/W04-3250](https://aclanthology.org/W04-3250/)
- Dror et al. *The Hitchhiker's Guide to Testing Statistical Significance in Natural Language
  Processing.* ACL 2018. [aclanthology.org/P18-1128](https://aclanthology.org/P18-1128/)

## Known gaps

- **No real-model number yet.** Every accuracy figure in this repo's code paths is waiting on the
  run above. The published numbers quoted are other people's, on the same split.
- **QLoRA training has not run on a GPU.** The CPU loop runs the same code with
  `precision="fp32"`; the NF4 path (`load_base`, `prepare_model_for_kbit_training`,
  `paged_adamw_8bit`) needs CUDA.
- **450 training questions is small.** It is all the in-distribution, three-label data there is;
  `pqa_artificial` would need class rebalancing and a final pass on `pqa_labeled` to help rather
  than hurt (that is the paper's recipe), and is left out deliberately.
- **Contamination cannot be ruled out.** PubMedQA has been public since 2019; a base model's
  absolute score may be inflated. The fine-tuning delta is measured on held-out questions.
- **int8 serving** is not offered: vLLM has no in-flight bitsandbytes 8-bit mode; the 8-bit arm is
  FP8 from `sovereign compress`.
- **The compress step has not run against llm-compressor.** Its recipes follow llm-compressor's
  documented FP8 and GPTQ examples, and the tests check what is passed to them, through stand-ins.
  W4A16 calibrates on the 450 training prompts (fewer than the usual 512). FP8 *compute* needs
  compute capability 8.9+ (Ada, Hopper, Blackwell); older GPUs run FP8 checkpoints weight-only.
- **Constrained decoding uses vLLM's `structured_outputs`** (0.12+); older servers ignore or
  reject the field.

## More projects

One of five enterprise AI builds, each a standalone package with its own tests and CI:

- [guarded-support-agent](https://github.com/madhusudanpatnaik/guarded-support-agent): a tool-using agent with a gate on every call, human review of irreversible actions and rollback
- [document-intelligence](https://github.com/madhusudanpatnaik/document-intelligence): contracts and invoices to validated JSON: OCR vs. layout parsing vs. a vision model
- [enterprise-search-rag](https://github.com/madhusudanpatnaik/enterprise-search-rag): hybrid retrieval with re-ranking, verified citations and calibrated abstention
- [meeting-intelligence](https://github.com/madhusudanpatnaik/meeting-intelligence): transcription, diarization and summaries whose action items are checked against the transcript
- [sovereign-llm](https://github.com/madhusudanpatnaik/sovereign-llm) (this repository): fine-tune, quantize and serve a small open model, measured against a frontier API

MIT licensed: see [LICENSE](LICENSE).
