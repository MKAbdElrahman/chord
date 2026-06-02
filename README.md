# chord

Compose local AI models as Unix filters — format in, format out. Each model
reads one format on stdin and writes another on stdout, so you connect them with
the shell pipe.

```sh
cat question.wav | chord stt | chord chat | chord tts | aplay
#                  audio→text   text→text    text→audio
```

Everything runs locally.

## Transforms

| Verb   | Kinds         | Engine |
|--------|---------------|--------|
| `stt`  | audio → text  | whisper.cpp |
| `tts`  | text → audio  | Supertonic / ONNX |
| `chat` | text → text   | llama.cpp |
| `see`  | image → text  | llama.cpp (vision) |
| `draw` | text → image  | stable-diffusion.cpp |

`chord ls` lists them. `chord <verb> --help` shows a verb's flags.

## Install

Requires a Rust toolchain. Every engine is a separate binary and the `chord`
host spawns them, so install the host plus each engine — they land side-by-side
in `~/.cargo/bin`, where `chord` finds them automatically:

```sh
cargo install --path crates/chord-cli            # the `chord` host
cargo install --path crates/transforms/chord-stt
cargo install --path crates/transforms/chord-tts
cargo install --path crates/transforms/chord-chat
cargo install --path crates/transforms/chord-see
cargo install --path crates/transforms/chord-draw
```

## Models

Each engine finds its model via config, `--model`, or a default path; if it's
missing it exits with a hint. `chord pull <transform>` fetches a default where
there's a canonical public download:

```sh
chord pull stt     # whisper large-v3-turbo (~1.5 GB) -> ~/models/
```

`draw` downloads its weights automatically on first use. For `chat`/`see` (large
GGUFs) and `tts` (the Supertonic asset bundle), point the engine at a local model
with `--model`/config (or `--assets` for tts).

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
  | chord chat --system "translate to German" \
  | chord tts | aplay

# Text to image
echo "a lighthouse at sunset" | chord draw --steps 6 > out.png
```

## Pipelines

Because every transform reads stdin and writes stdout, you compose them with the
shell pipe — each stage is its own process:

```sh
chord see photo.jpg --prompt "what is this?" | chord chat --system "translate to German" | chord tts
```

### The `pipeline` shortcut

`chord pipeline` runs a whole chain in **one process**, with stages separated by
`::` (so models load once instead of per stage):

```sh
chord pipeline see photo.jpg --prompt "what is this?" :: chat --system "translate to German" :: tts | aplay
```

Each stage is `verb [flags]`, written exactly as you would on its own. The first
stage reads the file argument or stdin, each later stage reads the previous
stage's output, and the last writes stdout.

## Configuration

Per-transform defaults live in a YAML file, found in order: `--config <file>`,
`$CHORD_CONFIG`, `./chord.yaml`, `~/.config/chord/config.yaml`. CLI flags
override the file. `chord config` shows what was resolved.

```yaml
chat:
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

The core (`crates/chord-core`) defines `Kind`, the `Transform` trait, and a
`Registry` — and nothing else.

Every engine runs **out-of-process**: each is its own `chord-<name>` binary (a
plain stdin → stdout filter) that links only its own native library. The host
(`crates/chord-cli`) links *no* engine code; for each one it registers an
exec-proxy that spawns the sibling binary and pipes bytes through it. So
`chord stt | chord chat | chord tts` and `chord pipeline …` work the same, while
each engine stays isolated.

```
chord (host)
  └─ proxy ──spawns──> chord-stt    whisper.cpp
  └─ proxy ──spawns──> chord-tts    Supertonic / ONNX
  └─ proxy ──spawns──> chord-chat   llama.cpp
  └─ proxy ──spawns──> chord-see    llama.cpp (vision)
  └─ proxy ──spawns──> chord-draw   stable-diffusion.cpp
```

This keeps incompatible native libraries (e.g. the two copies of `ggml` in
llama.cpp and stable-diffusion.cpp) out of one address space, lets engines
stream and crash independently, and means adding a model is a new binary plus
one line in the host — the core never changes.
