# Option conventions — cross-engine profiles

The kernel never interprets engine options (rule R8 in
[theory/THEORY.md](theory/THEORY.md)); portability across backends comes
from **conventions**, not enforced interfaces — the same way the
OpenAI-compatible API won as a convention adapters implement, with quirks
absorbed inside each adapter. An engine maps these keys to whatever its
backend actually means by them; the engine is the adapter.

## Universal

| key | meaning |
|---|---|
| `model` | the engine's weights: a path, directory, or `hf:` reference. Also the binding key for manifest-declared resources (`chord pull`). |

## `text -> audio` (tts profile)

| key | meaning |
|---|---|
| `model` | the model bundle (Supertonic: the assets dir; a Piper-style backend would map `voice` to model files internally) |
| `voice` | named speaker preset or path — whatever a "voice" is for that backend (style vector, embedding, per-voice model, reference clip) |
| `lang`  | language code |
| `speed` | speaking rate, 1.0 = natural |
| `chunk` | max characters per synthesis chunk — the latency knob |

Backends MAY declare extras; consumers should not assume them. Deprecated
aliases (e.g. tts `assets` for `model`) stay declared until removed.

## Why not a kernel interface

Voice semantics differ *structurally* across backends (Piper: voice IS the
model; XTTS: voice is a reference recording). A kernel-level TTS API would
either be the lowest common denominator or leak every quirk. Conventions
keep the host ignorant (R8), let `--backend X` swap engines with the same
flags (the host already unions alternates' declared options), and put the
quirk-absorbing code where the quirk lives.
