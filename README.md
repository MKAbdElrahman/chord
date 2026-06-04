# chord

Compose local AI models as Unix filters. Each transform reads an ordered list of
typed parts (a *message*) on stdin and writes one on stdout, so you connect them
with the shell pipe. In the common case a message is a single part of one
modality, and the pipe carries it as raw bytes — so a stage stays a plain filter.

```sh
cat question.wav | chord stt | chord text | chord tts | aplay
#                  audio→text   text→text    text→audio
```

A transform is, in full generality, a function `message -> message`: an ordered
list of typed parts in, an ordered list out. The mono-modal verbs (`stt`, `tts`,
`text`, `see`, `draw`) are the one-part-in/one-part-out special case; `chat` is
the general multimodal case — text, images, and audio in one prompt. Everything
runs locally.

## Transforms

| Verb   | Kinds         | Backend |
|--------|---------------|--------|
| `stt`  | audio → text  | whisper.cpp (also `--backend llama.cpp` / `sherpa-onnx`) |
| `tts`  | text → audio  | Supertonic / ONNX |
| `text` | text → text   | llama.cpp (text generation / chat) |
| `chat` | text · image · audio → text | mistral.rs (multimodal) |
| `see`  | image → text  | llama.cpp (vision) |
| `draw` | text → image  | stable-diffusion.cpp |
| `redact` | text → text | ONNX (PII filter) |
| `pack` · `unpack` | message ↔ parts | — (assemble / inspect multimodal messages) |
| `vad` · `langid` · `diarize` | audio → text | sherpa-onnx |

`text` is the lightweight text-only LLM (llama.cpp, GGUF); `chat` is the general
multimodal model (mistral.rs) that takes text, images, and audio together.

`chord ls` lists whatever engines are installed (engines are discovered, not
hardcoded). `chord <verb> --help` shows a verb's flags; `--backend <id>` picks an
alternate engine where one exists.

## Install

Requires a Rust toolchain. Every engine is a separate binary and the `chord`
host *discovers* them by name, so install the host plus each engine you want —
they land side-by-side in `~/.cargo/bin`, where `chord` finds them automatically:

```sh
cargo install --path crates/chord-cli            # the `chord` host
cargo install --path crates/transforms/chord-stt
cargo install --path crates/transforms/chord-tts
cargo install --path crates/transforms/chord-text   # text-only LLM (llama.cpp)
cargo install --path crates/transforms/chord-chat   # multimodal (mistral.rs)
cargo install --path crates/transforms/chord-see
cargo install --path crates/transforms/chord-draw
cargo install --path crates/transforms/chord-redact
cargo install --path crates/transforms/chord-pack
cargo install --path crates/transforms/chord-unpack
# audio building blocks + alternate stt backends (sherpa-onnx):
cargo install --path crates/transforms/chord-vad
cargo install --path crates/transforms/chord-langid
cargo install --path crates/transforms/chord-diarize
cargo install --path crates/transforms/chord-stt-sherpa
cargo install --path crates/transforms/chord-stt-llama
```

Adding a new engine needs **no host change** — `cargo install` it next to `chord`
and it shows up in `chord ls`. The sherpa-onnx engines also need their shared
libraries beside the binary (the loader must not pick up a system ORT):

```sh
cp target/release/libsherpa-onnx-c-api.so target/release/libonnxruntime.so ~/.cargo/bin/
```

## Models

Each engine finds its model via config, `--model`, or a default path; if it's
missing it exits with a hint. Models live under the XDG data dir
(`$XDG_DATA_HOME/chord/models`, i.e. `~/.local/share/chord/models`).
`chord pull <transform>` fetches a default where there's a canonical public
download:

```sh
chord pull stt     # whisper large-v3-turbo (~1.5 GB) -> ~/.local/share/chord/models/
```

A `--model hf:org/repo[:quant-or-file]` reference (or `chord pull --model hf:…`)
pulls from the Hugging Face Hub into its cache. `draw` downloads its weights
automatically on first use. For `text`/`see` (large GGUFs), `chat` (mistral.rs
models, e.g. a multimodal Gemma), and `tts` (the Supertonic asset bundle), point
the engine at a local model with `--model`/config (or `--assets` for tts). Each
engine also honors a `$CHORD_<NAME>_MODEL` env override.

## Syntax

Every transform is a filter with the same shape:

```sh
chord <verb> [input] [--flags]
```

- **input** — a file path, or stdin if omitted (or given as `-`).
- **flags** — named options like `--prompt`, `--system`, `--steps`; see `chord <verb> --help`.
- **output** — always stdout.

```sh
# Describe an image, translate the answer, speak it
chord see photo.jpg --prompt "what is this?" \
  | chord text --system "translate to German" \
  | chord tts | aplay

# Text to image
echo "a lighthouse at sunset" | chord draw --steps 6 > out.png
```

### Multimodal messages

`chat` takes several modalities in one prompt. Give it files directly, or build a
message with `pack` and pipe it in:

```sh
# Files straight on the chat command
chord chat --image cat.png --audio question.wav "Describe the image and answer the question"

# Or assemble a message explicitly, then inspect or send it
chord pack --image cat.png --text "what is this?" --audio q.wav | chord unpack --manifest
chord pack --image cat.png --text "what is this?" --audio q.wav | chord chat
```

A single-part message is carried raw (so `chord draw > x.png` still works);
multi-part messages are framed. `unpack --part N` extracts one part's bytes.

## Pipelines

Because every transform reads stdin and writes stdout, you compose them with the
shell pipe — each stage is its own process:

```sh
chord see photo.jpg --prompt "what is this?" | chord text --system "translate to German" | chord tts
```

### The `pipeline` shortcut

`chord pipeline` runs a whole chain as a single command, with stages separated by
`::`:

```sh
chord pipeline see photo.jpg --prompt "what is this?" :: text --system "translate to German" :: tts | aplay
```

Each stage is `verb [flags]`, written exactly as you would on its own. The first
stage reads the file argument, literal text, or stdin; the last writes stdout.
It's a **true streaming pipeline**: every stage's engine is spawned at once and
wired with OS pipes (stage N's stdout *is* stage N+1's stdin), so the stages run
concurrently and bytes flow between them as they're produced — exactly like the
shell pipe above, just one command with unified flag parsing and an up-front
model check. If a stage fails, the pipeline exits with its categorized code.

## Configuration

Per-transform defaults live in a YAML file, found in order: `--config <file>`,
`$CHORD_CONFIG`, `./chord.yaml`, `~/.config/chord/config.yaml`. CLI flags
override the file. `chord config` shows what was resolved.

```yaml
text:
  model: ~/models/qwen3-8b.gguf
  system: "Be concise."
```

## Scripting

Every transform exits with a categorized code — `0` ok, `1` engine error,
`2` bad input, `3` missing model — so callers can branch on the failure. Add
`--format jsonl` for machine-readable lifecycle/error events on stderr (the data
still flows on stdout):

```sh
$ printf '' | chord tts --format jsonl
{"event":"start","transform":"tts"}
{"event":"error","transform":"tts","code":2,"kind":"bad_input","message":"no input text"}
```

## Architecture

The core (`crates/chord-core`) defines `Kind`, `Message`/`Part` (the data plane
and its wire codec), the `Transform` trait (`message -> message`) with the
`Unary` convenience for 1→1 engines, a `Registry`, the plug-in `Manifest`, and
the XDG path helpers — and nothing else. The wire codec writes a lone inline part
as bare bytes (so mono-modal pipes stay raw) and frames multi-part messages.

Every engine runs **out-of-process**: each is its own `chord-<name>` binary (a
plain stdin → stdout filter) that links only its own native library. The host
(`crates/chord-cli`) links *no* engine code.

Engines are **self-describing plug-ins**, discovered the way `git`/`cargo` find
their subcommands. Each binary answers `--chord-manifest` with a JSON description
of itself (name, kinds, backend, flags); the host scans for `chord-*` binaries
beside it, queries each, and builds its registry from what's installed — caching
manifests so steady-state startup is just a few `stat`s. There is no hardcoded
engine list and no option tables duplicated in the host: an engine's own
`Transform` impl is the single source of truth.

```
chord (host) ── discovers + queries ──┐
  ├─ chord-stt --chord-manifest        │   {"name":"stt","accepts":["audio"],"emits":["text"],…}
  ├─ chord-chat --chord-manifest       │   {"name":"chat","backend":"mistral.rs",…}
  └─ …                                 ┘
        │
        └─ run: spawn the sibling binary, pipe bytes through it
```

To run a transform the host spawns the matching binary and streams bytes through
it; `chord pipeline a :: b :: c` spawns every stage at once and connects them
with OS pipes (a real concurrent streaming chain). Errors are categorized
(`ChordError` → exit codes `1`/`2`/`3`), logging is `tracing` on stderr filtered
by `RUST_LOG`, and paths follow XDG via the `directories` crate.

This keeps incompatible native libraries (e.g. the two copies of `ggml` in
llama.cpp and stable-diffusion.cpp) out of one address space, lets engines stream
and crash independently, and means **adding a model is just a new binary** —
install it next to `chord` and it's discovered. The core never changes, and the
host never grows an engine dependency.
