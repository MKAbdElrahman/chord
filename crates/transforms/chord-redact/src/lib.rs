//! chord-redact — privacy filter (`text -> text`): detect and redact PII.
//!
//! Backed by `openai/privacy-filter`, a bidirectional token classifier (gpt-oss
//! lineage) that labels every token with a BIOES tag over 8 privacy categories
//! — `private_person`, `private_email`, `private_phone`, `private_address`,
//! `private_url`, `private_date`, `account_number`, `secret` — in a single
//! forward pass. We run its ONNX export through ONNX Runtime (`ort`) with the
//! repo's HF tokenizer, take the per-token argmax (the official decode; the
//! upstream Viterbi pass is a span-coherence refinement), merge same-category
//! spans, and replpleaace each with a `[CATEGORY]` placeholder.
//!
//! Options (via `-o key=value`):
//!   - `model` — PII model: a local `.onnx` path, or an `hf:` reference to the
//!     ONNX file (default `hf:openai/privacy-filter:onnx/model_q4f16.onnx`).
//!     The tokenizer/config and the `.onnx_data` weights are resolved alongside.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use chord_core::{ChordError, Kind, OptionSpec, Options, Result, Unary};
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

/// Default PII model: the smallest (4-bit) ONNX export of openai/privacy-filter.
const DEFAULT_MODEL: &str = "hf:openai/privacy-filter:onnx/model_q4f16.onnx";

const OPTS: &[OptionSpec] = &[OptionSpec {
    key: "model",
    help: "PII model: .onnx path or hf: ref (default openai/privacy-filter q4f16)",
    takes_value: true,
}];

pub struct Redact;

impl Unary for Redact {
    fn name(&self) -> &str {
        "redact"
    }
    fn from(&self) -> Kind {
        Kind::Text
    }
    fn to(&self) -> Kind {
        Kind::Text
    }
    fn describe(&self) -> &str {
        "privacy filter: detect & redact PII (openai/privacy-filter)"
    }
    fn backend(&self) -> &str {
        "onnxruntime"
    }
    fn options(&self) -> &'static [OptionSpec] {
        OPTS
    }

    fn apply(&self, input: &mut dyn Read, output: &mut dyn Write, opts: &Options) -> Result<()> {
        let assets = resolve_assets(opts.get("model"))?;

        let tokenizer = Tokenizer::from_file(&assets.tokenizer)
            .map_err(|e| ChordError::Engine(format!("load tokenizer: {e}")))?;
        let id2label = load_labels(&assets.config)?;
        let mut session = Session::builder()?.commit_from_file(&assets.onnx)?;

        let mut text = String::new();
        input.read_to_string(&mut text)?;
        // Preserve a trailing newline if there was one; redact the rest.
        let (body, trailing_nl) = match text.strip_suffix('\n') {
            Some(b) => (b, true),
            None => (text.as_str(), false),
        };

        let redacted = redact_text(body, &tokenizer, &mut session, &id2label)?;
        output.write_all(redacted.as_bytes())?;
        if trailing_nl {
            output.write_all(b"\n")?;
        }
        Ok(())
    }
}

/// Run the token classifier over `text` and return it with PII spans replaced by
/// `[CATEGORY]` placeholders.
fn redact_text(
    text: &str,
    tokenizer: &Tokenizer,
    session: &mut Session,
    id2label: &[String],
) -> Result<String> {
    if text.trim().is_empty() {
        return Ok(text.to_string());
    }

    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| ChordError::Engine(format!("tokenize: {e}")))?;
    let ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
    let offsets = encoding.get_offsets();
    let len = ids.len() as i64;
    if len == 0 {
        return Ok(text.to_string());
    }

    let ids_t = Tensor::from_array((vec![1_i64, len], ids))?;
    let mask_t = Tensor::from_array((vec![1_i64, len], vec![1_i64; len as usize]))?;
    let outputs = session.run(ort::inputs![
        "input_ids" => ids_t,
        "attention_mask" => mask_t,
    ])?;

    // Output is the logits tensor; the model has a single output.
    let (shape, logits) = outputs[0].try_extract_tensor::<f32>()?;
    let num_labels = *shape.last().unwrap_or(&0) as usize;
    if num_labels == 0 {
        return Err(ChordError::Engine("model produced no labels".into()).into());
    }

    // Per token: argmax over the label axis, then keep non-`O` spans with offsets.
    let mut spans: Vec<(usize, usize, String)> = Vec::new();
    for (t, &(start, end)) in offsets.iter().enumerate() {
        if end <= start {
            continue; // special tokens / empty pieces
        }
        let row = &logits[t * num_labels..(t + 1) * num_labels];
        let best = argmax(row);
        let label = id2label.get(best).map(String::as_str).unwrap_or("O");
        if label == "O" {
            continue;
        }
        spans.push((start, end, category_of(label)));
    }

    Ok(apply_redactions(text, &merge_spans(text, spans)))
}

/// Coalesce token spans into coherent entity spans, a lightweight stand-in for
/// the upstream Viterbi decoder (which we approximate with argmax):
///   - same category over a short gap with no clause punctuation -> one span
///     (multi-word names/addresses, e.g. "Mountain View CA 94043"), and
///   - directly touching pieces of different categories (fragmentation inside
///     one run) -> one span, the longer piece's category winning.
fn merge_spans(text: &str, mut spans: Vec<(usize, usize, String)>) -> Vec<(usize, usize, String)> {
    let bytes = text.as_bytes();
    spans.sort_by_key(|s| s.0);
    let mut merged: Vec<(usize, usize, String)> = Vec::new();
    for (start, end, cat) in spans {
        if let Some(last) = merged.last_mut() {
            let gap = start.saturating_sub(last.1);
            let same = last.2 == cat;
            let bridgeable = same && gap <= 15 && !clause_break(&bytes[last.1..start]);
            if bridgeable || (!same && gap == 0) {
                if !same && (end - start) > (last.1 - last.0) {
                    last.2 = cat;
                }
                last.1 = last.1.max(end);
                continue;
            }
        }
        merged.push((start, end, cat));
    }
    merged
}

/// Rebuild `text` with each byte-range span replaced by its `[CATEGORY]` tag.
fn apply_redactions(text: &str, spans: &[(usize, usize, String)]) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    for (start, end, cat) in spans {
        let (mut start, mut end) = (*start.min(&bytes.len()), *end.min(&bytes.len()));
        // ByteLevel BPE folds the leading space into the token; trim whitespace
        // at the span edges so we don't swallow the spacing around the entity.
        while start < end && bytes[start].is_ascii_whitespace() {
            start += 1;
        }
        while end > start && bytes[end - 1].is_ascii_whitespace() {
            end -= 1;
        }
        if start < cursor {
            continue; // overlap guard
        }
        out.push_str(&String::from_utf8_lossy(&bytes[cursor..start]));
        out.push('[');
        out.push_str(cat);
        out.push(']');
        cursor = end;
    }
    out.push_str(&String::from_utf8_lossy(&bytes[cursor..]));
    out
}

/// Map a BIOES label (`B-private_email`, `S-secret`, …) to a redaction tag
/// (`EMAIL`, `SECRET`). The `private_` prefix is cosmetic and dropped.
fn category_of(label: &str) -> String {
    let cat = label.split_once('-').map(|(_, c)| c).unwrap_or(label);
    cat.strip_prefix("private_")
        .unwrap_or(cat)
        .to_ascii_uppercase()
}

/// True if a between-spans gap holds punctuation that ends a clause — a signal
/// not to bridge two same-category spans across it.
fn clause_break(gap: &[u8]) -> bool {
    gap.iter()
        .any(|b| matches!(b, b'.' | b',' | b';' | b':' | b'!' | b'?' | b'\n' | b'"' | b'(' | b')'))
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

struct Assets {
    onnx: PathBuf,
    tokenizer: PathBuf,
    config: PathBuf,
}

/// Resolve the ONNX graph, its external-data weights, the tokenizer, and the
/// config. For `hf:` references everything is fetched from the same repo (and
/// must be present — see `chord pull redact`); for a local `.onnx` path the
/// siblings are looked up next to it.
fn resolve_assets(model: Option<&str>) -> Result<Assets> {
    let spec = model.unwrap_or(DEFAULT_MODEL);

    if chord_hf::is_hf(spec) {
        // `hf:<repo>[@rev]:<file>` -> the bare `hf:<repo>[@rev]` ref for siblings.
        let after = spec.strip_prefix("hf:").unwrap_or(spec);
        let repo_ref = match after.split_once(':') {
            Some((r, _)) => format!("hf:{r}"),
            None => spec.to_string(),
        };
        let onnx = cached(spec)?;
        // The weights live in a sibling `.onnx_data`; ensure it's downloaded so
        // ONNX Runtime finds it next to the graph (it loads it by relative name).
        let _weights = cached(&format!("{spec}_data"))?;
        let tokenizer = cached(&format!("{repo_ref}:tokenizer.json"))?;
        let config = cached(&format!("{repo_ref}:config.json"))?;
        Ok(Assets {
            onnx,
            tokenizer,
            config,
        })
    } else {
        let onnx = PathBuf::from(spec);
        if !onnx.exists() {
            return Err(ChordError::ModelMissing {
                what: format!("PII model {spec:?}"),
                hint: "run `chord pull redact`, or set --model to an .onnx path".to_string(),
            }
            .into());
        }
        Ok(Assets {
            tokenizer: find_sibling(&onnx, "tokenizer.json")?,
            config: find_sibling(&onnx, "config.json")?,
            onnx,
        })
    }
}

/// Cached path for an `hf:` spec, or a `chord pull redact` hint if not present.
fn cached(spec: &str) -> Result<PathBuf> {
    chord_hf::cached_path(spec)?.ok_or_else(|| {
        ChordError::ModelMissing {
            what: format!("PII model file {spec}"),
            hint: "run `chord pull redact` to download openai/privacy-filter".to_string(),
        }
        .into()
    })
}

/// Look for `name` next to a local model file, or one directory up (HF lays the
/// ONNX out under an `onnx/` subdir with tokenizer/config at the repo root).
fn find_sibling(onnx: &Path, name: &str) -> Result<PathBuf> {
    let dir = onnx.parent().unwrap_or(Path::new("."));
    for cand in [dir.join(name), dir.join("..").join(name)] {
        if cand.exists() {
            return Ok(cand);
        }
    }
    Err(ChordError::ModelMissing {
        what: format!("{name} for {}", onnx.display()),
        hint: format!("place {name} next to the .onnx (or its parent dir)"),
    }
    .into())
}

/// Read `id2label` from the model `config.json` into an index-ordered vec.
fn load_labels(config: &Path) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(config)?;
    let v: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| ChordError::Engine(format!("parse config.json: {e}")))?;
    let map = v
        .get("id2label")
        .and_then(|m| m.as_object())
        .ok_or_else(|| ChordError::Engine("config.json has no id2label".into()))?;
    let mut labels = vec![String::from("O"); map.len()];
    for (k, val) in map {
        let idx: usize = k
            .parse()
            .map_err(|_| ChordError::Engine(format!("bad label id {k:?}")))?;
        if let Some(s) = val.as_str() {
            if idx < labels.len() {
                labels[idx] = s.to_string();
            }
        }
    }
    Ok(labels)
}
