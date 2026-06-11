# Local inference landscape — verified research report (2026-06)

Deep-research run for chord's roadmap: hardware requirements of trending
local engines, the physics of LLM inference, vLLM architecture lessons, and
local+remote hybrid pipelines. Method: 5 parallel search angles, 24 sources
fetched, 120 claims extracted, 25 survived 3-vote adversarial verification
(0 refuted), merged to 13 findings. Confidence labels reflect that process.
Companion to [`../theory/THEORY.md`](../theory/THEORY.md) — this report puts
numbers behind rules R3–R7.

## 1. Hardware requirements of the trending engines

All high-confidence, verified against primary repos (fetched 2026-06-11):

- **llama.cpp covers chord's whole CPU target**: AVX, AVX2, AVX512, AMX on
  x86; ARM NEON/Accelerate/Metal with Apple silicon first-class. No SIMD
  feature-gating needed in chord — the engine autodetects. AVX2 is the
  practical consumer baseline.
- **GGUF quantization is the RAM lever**: 1.5–8-bit integer quants; Q4_0 ≈
  4.5 effective bits/weight (block format: 18 bytes per 32 weights), Q8_0 ≈
  8.5. A 7B model ≈ **3.5 GB at Q4_0** vs ≈ 7 GB at Q8_0. Caveat: weight
  quantization does not shrink KV cache or activations.
- **GPU path**: CUDA, AMD HIP, MUSA, Vulkan, SYCL (Intel-only since
  Feb 2026), Metal — plus **CPU+GPU hybrid offload** for models larger than
  VRAM, so chord's future GPU support is partial-offload planning, not an
  all-or-nothing VRAM gate. Vulkan is the most portable consumer backend.
- **stable-diffusion.cpp**: CPU AVX/AVX2/AVX512 + the same GPU backends.
  SD 1.x @512×512: ~2.8 GB f32 → ~2.0 GB quantized → **~1.5 GB with Flash
  Attention**. ~32 s/image on a 10-core laptop CPU vs ~2.5 s on an RTX 3070.
  `--diffusion-fa` saves real memory (~600 MB FLUX@768, ~1.4 GB SD2@768) but
  *slows down* most non-CUDA backends — default it per backend (on for
  CPU-when-RAM-tight and CUDA, off for Vulkan/SYCL).
- **sherpa-onnx is the lowest-risk stack**: fully offline on x86 through
  ARM/RISC-V, onnxruntime defaults to the CPU provider; STT (Whisper,
  Moonshine, Paraformer, SenseVoice, Zipformer), TTS (Piper, Matcha, VITS,
  Kokoro-82M, ZipVoice), diarization, VAD, more — models sized in the
  tens-to-hundreds of MB (Kokoro: 82M params).

**Implications for chord.** The `mem_hint` admission-control roadmap item
can *compute* hints instead of guessing: `params × bpw / 8 + KV budget`.
chord-draw's defaults should be quantized GGUF + per-backend FA. The
sherpa-onnx stages are safe anywhere; the chat stage is the budget driver.

## 2. The physics of LLM inference

Verified against Pope et al. (MLSys 2023), the DeepMind scaling book, and
the LMCache calculator; all 3-0 votes:

- **Two phases, two regimes.** Prefill is parallel and compute-bound;
  decode is sequential and **memory-bandwidth-bound** — at T=1 attention
  arithmetic intensity is O(1): a trickle of FLOPs while streaming the
  entire KV cache. (PaLM-540B-era anchor: 3 TB of KV at batch 512 — 3× the
  parameters — while compute sat idle. MHA-era; GQA/MLA models are 4–8×
  smaller.)
- **The throughput law**: max decode tokens/s = B × bandwidth /
  (B × KV-bytes + param-bytes); at batch 1 with small KV this is
  **tokens/s ≈ bandwidth / model-bytes**. Worked example: Q4_0 8B (~4.5 GB)
  on a 60 GB/s dual-channel DDR5 desktop tops out around **13 tok/s**; Q8
  halves that. On CPU, buy quantization, not threads.
- **KV-cache math**: bytes/token = 2 × layers × kv_heads × head_dim ×
  dtype_bytes. Llama-3.1-8B (32 layers, 8 KV heads, head_dim 128, fp16) =
  **128 KiB/token = 1 GiB per 8K context** (half with q8_0 KV, ~6% scale
  overhead). Formula covers MHA/GQA; not MLA or SSM hybrids.
- **Batching becomes compute-bound only above** B_crit = (FLOPs/s ÷
  bandwidth) × (weight bits ÷ activation bits) — ~240–280 tokens on
  TPU v5e/H100, halved by int8 weights. Decode *attention* stays
  memory-bound at any batch. Constants are accelerator-derived; the
  qualitative conclusion transfers to CPU.
- Empirical anchor: PaLM 540B, int8, 64 TPU v4: 29 ms/token decode at low
  batch; 76% MFU on batched prefill.

**Implications for chord.** Single-stream chat speed on the CPU box is set
by DRAM bandwidth (~50–100 GB/s dual-channel), full stop. The Little's-Law
event log (R7) can now be compared against a *predicted* W = model-bytes /
bandwidth. Per-conversation KV is a first-class RAM line item for the
daemon: weights + 1 GiB per 8K fp16 context.

## 3. vLLM architecture lessons

- **PagedAttention** (SOSP 2023): pre-vLLM systems wasted 60–80% of KV
  memory to fragmentation; paging KV into fixed-size non-contiguous blocks
  (blocks=pages, tokens=bytes, sequences=processes) cut waste to <4% and
  gave 2–4× throughput vs 2023 baselines (FasterTransformer, Orca).
  Qualifier: vAttention (2024) disputes software-paging overhead (~10%),
  and the comparison baselines are deprecated — but the *framing* is now
  universal (even llama-server ships paged/continuous batching).
- **vLLM V1 scheduler** (2025, verified still current in 2026): erased the
  prefill/decode special-casing — each scheduling step is a plain
  `{request_id: num_tokens}` budget, general enough for chunked prefill,
  prefix caching, and speculative decoding with no special cases.

**Implications for chord.** The transferable lesson is Denning restated for
2026: treat resident models *and per-conversation KV* as paged working sets
that are admitted, evicted, and shared — never contiguous all-or-nothing
allocations. And if the batch/daemon mode ever needs a scheduler, copy V1's
minimalism: one token-budget dictionary, no phase special-cases (chord's
MTP speculative decoding folds into the same abstraction).

## 4. Local + remote hybrid pipelines

- **The OpenAI chat/completions API is the de-facto interface** on both
  sides of the wire: llama.cpp ships `llama-server` ("lightweight, OpenAI
  API compatible") with parallel decoding and continuous batching; Ollama,
  vLLM, LM Studio, LocalAI expose the same `/v1/chat/completions`,
  `/v1/completions`, `/v1/embeddings`. Qualifier from llama.cpp's own docs:
  "no strong claims of compatibility with OpenAI API spec is being made" —
  compatibility is practical, not certified. Integration-test against real
  endpoints (SSE edge cases, multimodal payloads); don't assume parity.

**Implications for chord.** A remote stage should be *just another engine
binary*: `chord-chat-openai` answers `--chord-manifest` like any plug-in
(`name: chat, backend: "openai"`), and its `apply` is an HTTPS client
instead of a local model. chord's existing `--backend` alternate mechanism
already selects it; the Kahn pipe discipline, kind checking, and event
instrumentation all apply unchanged. `pull::ensure` becomes "check API key
present" for remote backends. Mixing `stt (local) :: chat --backend openai
:: tts (local)` then requires zero kernel changes.

## Verified-coverage caveats (read before relying on this)

- **Topic 4 is one strong anchor plus inference**: claims about
  Gemini/Imagen request shapes, LiteLLM/OpenRouter/aisuite gateway
  normalization, SSE specifics, and latency/cost/privacy tradeoffs did
  **not** survive verification and remain open.
- **whisper.cpp and mistral.rs — chord's actual STT/chat engines — have no
  surviving dedicated claims**; llama.cpp/sd.cpp/sherpa-onnx findings are
  extrapolated to them at the reader's risk. mmap residency behavior (key
  for cross-process weight sharing) was not directly evidenced.
- No verified speculative-decoding speedup numbers (bandwidth-bound theory
  predicts gains ∝ acceptance rate; measure chord's own MTP path).
- Time-sensitive: vLLM 2–4× is vs deprecated baselines; engine READMEs and
  sd.cpp memory tables drift with releases.

## Open questions (next research / measurement targets)

1. Gemini/Imagen + gateway (LiteLLM/OpenRouter/aisuite) API normalization
   patterns — the missing half of the remote-stage design.
2. Measured speculative-decoding gains on CPU (chord ships MTP — measure
   it with the new jsonl instrumentation).
3. whisper.cpp / mistral.rs footprints, SIMD floors, and GGUF mmap
   residency across one-shot processes vs a daemon.
4. Daemon eviction policy when Σ model bytes > RAM: LRU over mmap'd GGUF
   vs explicit unload (ties to Denning τ≈2T and Sleator–Tarjan
   competitiveness in THEORY.md).
