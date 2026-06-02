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

Requires a Rust toolchain. `draw` is a separate helper binary; install both
(they land side-by-side in `~/.cargo/bin`):

```sh
cargo install --path crates/chord-cli
cargo install --path crates/transforms/chord-draw
```

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

## Architecture

The core (`crates/chord-core`) defines `Kind`, the `Transform` trait, and a
`Registry`. Engines are plug-in crates under `crates/transforms/`; the host
(`crates/chord-cli`) wires them up in `build_registry()`. Adding a model means
writing a plug-in and registering it — the core stays unchanged.
