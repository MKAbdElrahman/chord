# The theory behind chord's execution model

chord composes AI models as Unix filters. That design is not just an
aesthetic; it is an instance of five classical results in computer science.
This document states each result, cites the primary source (downloaded into
[`papers/`](papers/)), maps it onto chord's architecture, and derives the
design rules the code is expected to honor. Each rule is labeled **R1–R6**
and referenced from the code where it is enforced.

## 1. Kahn process networks — why the pipeline is correct

**Source:** G. Kahn, *The Semantics of a Simple Language for Parallel
Programming*, IFIP Congress 74, 1974.
([papers/kahn-1974-process-networks.pdf](papers/kahn-1974-process-networks.pdf))

**The result.** A network of sequential processes that communicate *only*
through unbounded FIFO channels, where reads **block** until data arrives,
computes a continuous function on streams. Kahn's theorem: the network's
observable behavior (the content of every channel) is the least fixed point
of that function — and is therefore **deterministic regardless of process
scheduling, speed, or spawn order**.

**In chord.** `chord pipeline a :: b :: c` is a textbook Kahn network: each
stage is a sequential OS process, the channels are OS pipes, and every stage
does blocking reads on stdin (`crates/chord-runner/src/lib.rs`). Kahn's
theorem is the license behind `run_pipeline`'s spawn-everything-at-once
strategy (`crates/chord-cli/src/main.rs`): the OS may schedule stages
however it likes; the output bytes are the same.

**Derived rules.**
- **R1 — pipes are the only channel.** Stages must not communicate out of
  band (shared files, sockets, environment) or determinism is lost. The
  `Transform` contract's statelessness rule is the per-process half of this.
- **R2 — the glue must not buffer what the kernel doesn't need.** A Kahn
  process is free to stream; buffering belongs to a *stage that needs it*
  (e.g. whisper needs the whole utterance), never to the plumbing. This is
  why `Transform::apply_raw` exists and why `Unary` passes streams straight
  through (`crates/chord-core/src/transform.rs`).

## 2. Pipelining — when overlap pays

**Source:** J. L. Hennessy & D. A. Patterson, *Computer Architecture: A
Quantitative Approach*, Appendix C ("Pipelining"). (Book; not downloadable —
the canonical treatment of instruction pipelining.)

**The result.** A k-stage pipeline improves **throughput** toward one item
per slowest-stage time, but never improves the **latency** of a single item
(it is the sum of stage times plus handoff overhead). The benefit exists
only with multiple items in flight; steady-state throughput is bounded by
the bottleneck stage; the first item pays the full pipeline-fill cost.

**In chord.** A single-message `chord pipeline` run is a pipeline with one
item in flight: no overlap is possible (until intra-message streaming lands,
§6), so nothing is gained by holding every stage's resources at once — and
nothing is lost by acquiring them one stage at a time.

**Derived rule.**
- **R3 — resource residency must follow data availability, not process
  lifetime.** Spawning all stages up front is fine (processes are cheap and
  Kahn-safe); *loading models* up front is not. See R4.

## 3. The working-set model — what should be resident

**Source:** P. J. Denning, *The Working Set Model for Program Behavior*,
Communications of the ACM 11(5), 1968.
([papers/denning-1968-working-set.pdf](papers/denning-1968-working-set.pdf))

**The result.** Keep resident exactly the set of pages a computation
references in its current phase (its *working set*); evicting and reloading
something that will be referenced again immediately is thrashing, and
residency beyond the working set is waste.

**In chord.** Models are the "pages." For a single buffered message, the
working set at any instant is **one stage's model** — the predecessor has
exited and the successor is still blocked on its pipe. For a batch of
messages flowing continuously, every stage's model is in the working set.
Reloading a model per item in a batch is thrashing by Denning's definition.

**Derived rule.**
- **R4 — lazy model load.** An engine loads its model inside `apply`, on
  first demand, never at process start. This single rule yields the optimal
  residency in *both* regimes: batch semantics serialize the loads
  (peak ≈ one model); streaming semantics overlap a load with the upstream
  stage's compute (prefetch for free). Stated as contract rule 4 on
  [`Transform`](../../crates/chord-core/src/transform.rs).

## 4. Amortized analysis — why batches must reuse loaded models

**Source:** R. E. Tarjan, *Amortized Computational Complexity*, SIAM
J. Alg. Disc. Meth. 6(2), 1985.
([papers/tarjan-1985-amortized-complexity.pdf](papers/tarjan-1985-amortized-complexity.pdf))

**The result.** Charge expensive, infrequent operations against the cheap
operations they enable; an O(L) setup amortizes to O(L/n) per item over n
items.

**In chord.** With model-load cost L and per-item inference cost c, a batch
of n inputs costs `n(L + c)` if every invocation reloads, vs `L + nc` if
engines persist. For chord, L (gigabytes from disk) dwarfs c, so the
break-even is n = 2.

**Derived rule.**
- **R5 — hoist model loads out of the per-item loop.** Implemented as
  `chord pipeline --each`: a delimited multi-message wire format (an
  end-of-message sentinel — git filter-process's flush-packet pattern), a
  per-message loop in `chord-runner` (`filter_each`), capability-gated via
  the manifest (`caps: ["each"]`, git-handshake style), with engines
  memoizing loaded models across `apply` calls (stt: whisper context per
  model path; tts: ONNX sessions per assets+voice). Measured on a 3-item
  tts::stt batch: per-item marginal cost fell from ~4.1 s (process-per-item
  loop) to ~1.8 s, with model loads paid exactly once.

## 5. Lazy evaluation — the mechanism behind R4

**Source:** D. P. Friedman & D. S. Wise, *CONS Should Not Evaluate its
Arguments*, Indiana University CS TR 44, 1976 (published at ICALP 1976).
([papers/friedman-wise-1976-cons-lazy.pdf](papers/friedman-wise-1976-cons-lazy.pdf))

**The result.** Make the constructor build *suspensions* (thunks) that are
forced only when a strict operation demands them. Evaluation order then
follows the data-dependency graph automatically — no scheduler decides what
to compute when; demand does.

**In chord.** A model load is the thunk; the first input byte is the strict
demand that forces it. Because engines block reading stdin before `apply`
runs, laziness is *structural*: under batch semantics demands arrive
sequentially (one model at a time), under streaming semantics they overlap
(loads hide behind upstream compute). The same one-line policy is optimal in
both worlds precisely because lazy evaluation defers the scheduling decision
to the data.

## 6. Little's Law — sizing what's in flight

**Source:** J. D. C. Little, *Little's Law as Viewed on Its 50th
Anniversary*, Operations Research 59(3), 2011.
([papers/little-2011-littles-law-50th.pdf](papers/little-2011-littles-law-50th.pdf))

**The result.** L = λW: the average number of items in a system equals
arrival rate × average time in system, independent of distributions or
scheduling.

**In chord.** In batch mode (R5), the messages in flight between stages
L = λW where W is dominated by the bottleneck engine. Bounded OS pipe
buffers (64 KiB on Linux) provide backpressure for free: a fast producer
blocks on `write` when the consumer lags, capping L without any code.

**Derived rule.**
- **R6 — rely on pipe backpressure; never add unbounded queues between
  stages.** (Kahn assumes unbounded channels for the determinism proof;
  bounded channels preserve determinism and add flow control — the standard
  practical refinement.)

## R8 — the host composes; engines deploy

Kahn's Property 2 (the minimal solution is continuous in the system's
operators) is the license for top-down design: *"we can postpone the
decision to implement a given function by a single process or a set of
interconnected processes: this decision will not introduce perturbations in
the remainder of the system."* chord extends "process or network" to
**deployment substrate**: an engine may be a CPU binary, a GPU binary, a
shell script, or an HTTPS client for a remote API — indistinguishable to
the host.

**The rule.** Everything the host knows about an engine arrives through the
manifest, and the host may only *interpret* composition-relevant fields:
name, kinds, options, version, description. Resource and hardware semantics
are opaque: the host may fetch a declared resource by its URI scheme
(`hf:`, `https:`) and display metadata, but never encode engine-specific or
hardware-specific knowledge. The PR test: *does this change make `chord-cli`
interpret deployment? Then it belongs in an engine.*

Consequences applied:
- `pull.rs`'s hardcoded per-engine model table was the one violation in the
  codebase; engines now declare `ResourceSpec`s in their manifests and the
  host keeps only the scheme handlers and the prompt/progress UX.
- Hardware adaptation (GPU backends, offload splits) happens inside engines
  (e.g. ggml dynamic backend loading) or by user choice (`--backend`,
  config) — never by host-side hardware probing.
- A remote stage (`chord-chat-openai`) is just another engine; its
  "resource" is an API key, its `apply` is a network call, and the host
  needs no concept of "remote".

## Deep-read addenda

A close reading of the primary sources sharpened four points:

- **Memoized forcing (R4 corollary).** Friedman & Wise §III: coercing a
  suspension must produce the value and *store it back* — never re-evaluate,
  never silently yield nothing. Hence `Part::force()` errors loudly on an
  unresolved `Ref` body; eventual ref resolution must materialize-once.
- **Measurement before optimization (R7).** Little's finite-window theorem
  (LL.1/LL.2) is *numerically exact on a sample path* — no stationarity, no
  probability. Timestamped, byte-counted `--format jsonl` events are
  therefore sufficient to compute each stage's λ, W, L exactly from one
  run's log. The event schema carries `ts_ms`, `pid`, `duration_ms`,
  `bytes_in`, `bytes_out` for precisely this.
- **Prefetch is justified here (R3 refinement).** Denning argues against
  look-ahead because "there is no reliable advance source of allocation
  information" — but a pipeline plan IS one. Loading stage N+1's model while
  stage N computes is sound prepaging, gated on an admission check
  (Denning's balance policy, eqs. 10–13) once manifests carry memory hints.
- **Control-plane determinism (R1 corollary).** Kahn's restriction (ii) —
  lines transmit in *finite* time — motivates the manifest probe timeout;
  his determinism theorem motivates sorted discovery and an
  enumeration-order-free default-backend choice. His "2-plicator" shows
  deterministic fan-out (tee) is sound while fan-in/merge is not — the guard
  rail for any future pipeline DAG syntax.

## Rule-to-code map

| Rule | Statement | Where | Status |
|------|-----------|-------|--------|
| R1 | Pipes are the only inter-stage channel | `run_pipeline`, `Transform` statelessness | implemented |
| R2 | Glue never buffers what the kernel doesn't need | `Transform::apply_raw`, `Unary` override, `chord_runner::filter` | implemented |
| R3 | Residency follows data availability | consequence of R4 under both regimes | implemented |
| R4 | Models load lazily, inside `apply` | `Transform` contract rule 4; all engines | implemented (now contractual) |
| R5 | Batch mode amortizes loads over items | `chord pipeline --each`, `filter_each`, `read/write_delimited`, engine model memoization | implemented |
| R6 | Backpressure via bounded pipes only | OS pipes in `run_pipeline` | implemented |
| R7 | Measure before optimizing (Little, exact on sample paths) | `events.rs` ts/pid/duration/bytes; `chord_runner::filter` counting | implemented |
| R8 | The host composes; engines deploy | `Transform::resources`, manifest `resources`, generic `pull.rs` | implemented |

## Honest limits

- **Streaming is plumbing-deep only (R2).** `apply_raw` keeps the *glue*
  from buffering raw singleton streams, but most engines still buffer
  internally (whisper genuinely needs the whole utterance; chat/tts can and
  should become incremental). Until engines stream, a single-message
  pipeline still computes sequentially — R2 removes the architectural
  obstacle, not the engine-level one.
- **Framed (multi-part) messages are still buffered.** Chunked part framing
  is the eventual fix; it is deliberately sequenced after R5 since both
  touch the wire format.
- **The host's single-transform path (`chord stt file.wav`) buffers in the
  host** (`run_filter` decodes, then the exec-proxy re-encodes to the
  child). Streaming that path is a separate, smaller refactor.
