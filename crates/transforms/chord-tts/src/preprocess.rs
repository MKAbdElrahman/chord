//! Text preprocessing for the Supertonic model: NFKD normalize,
//! symbol/quote cleanup, ensure terminal punctuation, wrap in `<lang>…</lang>`,
//! then map each Unicode codepoint through the model's indexer.

use unicode_normalization::UnicodeNormalization;

/// Convert text to model token ids: preprocess, then look up each codepoint in
/// the indexer (out-of-range codepoints map to -1, matching the reference).
pub fn text_to_ids(text: &str, lang: &str, indexer: &[i64]) -> Vec<i64> {
    let processed = preprocess(text, lang);
    processed
        .chars()
        .map(|c| {
            let cp = c as usize;
            if cp < indexer.len() {
                indexer[cp]
            } else {
                -1
            }
        })
        .collect()
}

fn preprocess(text: &str, lang: &str) -> String {
    // NFKD normalization (e.g. decomposes accented letters), as the reference does.
    let mut s: String = text.nfkd().collect();

    // Dash/quote/symbol replacements.
    for (from, to) in [
        ("\u{2013}", "-"), // en dash
        ("\u{2011}", "-"), // non-breaking hyphen
        ("\u{2014}", "-"), // em dash
        ("_", " "),
        ("\u{201C}", "\""),
        ("\u{201D}", "\""),
        ("\u{2018}", "'"),
        ("\u{2019}", "'"),
        ("\u{00B4}", "'"),
        ("`", "'"),
        ("[", " "),
        ("]", " "),
        ("|", " "),
        ("/", " "),
        ("#", " "),
        ("\u{2192}", " "),
        ("\u{2190}", " "),
    ] {
        s = s.replace(from, to);
    }
    for sym in ["\u{2665}", "\u{2606}", "\u{2661}", "\u{00A9}", "\\"] {
        s = s.replace(sym, "");
    }
    for (from, to) in [
        ("@", " at "),
        ("e.g.,", "for example, "),
        ("i.e.,", "that is, "),
    ] {
        s = s.replace(from, to);
    }

    // Fix spacing before punctuation.
    for (from, to) in [
        (" ,", ","),
        (" .", "."),
        (" !", "!"),
        (" ?", "?"),
        (" ;", ";"),
        (" :", ":"),
        (" '", "'"),
    ] {
        s = s.replace(from, to);
    }

    // Collapse duplicate quotes.
    while s.contains("\"\"") {
        s = s.replace("\"\"", "\"");
    }
    while s.contains("''") {
        s = s.replace("''", "'");
    }

    // Collapse whitespace, trim.
    s = s.split_whitespace().collect::<Vec<_>>().join(" ");

    // Ensure terminal punctuation.
    if let Some(last) = s.chars().last() {
        const ENDERS: &str = ".!?;:,'\"\u{201C}\u{201D}\u{2018}\u{2019})]}\u{2026}\u{3002}\u{300D}\u{300F}\u{3011}\u{3009}\u{300B}\u{203A}\u{00BB}";
        if !ENDERS.contains(last) {
            s.push('.');
        }
    }

    // Wrap with language tags.
    format!("<{lang}>{s}</{lang}>")
}

/// Split text into chunks no longer than `max_len` characters, on paragraph and
/// then sentence boundaries (a simplified form of the reference chunker).
pub fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }

    let mut chunks = Vec::new();
    for para in text.split("\n\n") {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }
        if para.chars().count() <= max_len {
            chunks.push(para.to_string());
            continue;
        }
        // Pack sentences up to max_len.
        let mut current = String::new();
        for sentence in split_sentences(para) {
            if current.chars().count() + sentence.chars().count() + 1 > max_len
                && !current.is_empty()
            {
                chunks.push(current.trim().to_string());
                current.clear();
            }
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(sentence.trim());
        }
        if !current.trim().is_empty() {
            chunks.push(current.trim().to_string());
        }
    }
    if chunks.is_empty() {
        chunks.push(text.to_string());
    }
    chunks
}

/// Split on sentence-ending punctuation, keeping the punctuation with the
/// sentence.
fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        cur.push(c);
        if matches!(c, '.' | '!' | '?') {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

/// Pop the next speakable chunk off a growing stream buffer, or `None` if no
/// complete sentence has arrived yet. Used to synthesize while an upstream
/// token stream (chord-chat) is still generating: a chunk is ready when a
/// sentence terminator is followed by whitespace inside `max_len`, or when
/// the buffer exceeds `max_len` (split at the last whitespace). The caller
/// flushes whatever remains at EOF via [`chunk_text`].
pub fn take_stream_chunk(pending: &mut String, max_len: usize) -> Option<String> {
    let bytes = pending.as_bytes();
    let mut boundary = None;
    for (i, &b) in bytes.iter().enumerate() {
        if matches!(b, b'.' | b'!' | b'?')
            && bytes.get(i + 1).is_some_and(|c| c.is_ascii_whitespace())
        {
            boundary = Some(i + 1);
            if i + 1 >= max_len / 2 {
                break; // long enough — don't wait for more
            }
        }
        if i >= max_len {
            break;
        }
    }
    let cut = match boundary {
        Some(b) => b,
        None if pending.len() > max_len => {
            // No sentence boundary in sight: split at the last whitespace
            // inside the window so we never cut a word (or a UTF-8 char).
            pending[..max_len]
                .char_indices()
                .rev()
                .find(|(_, c)| c.is_whitespace())
                .map(|(i, _)| i + 1)?
        }
        None => return None,
    };
    let rest = pending.split_off(cut);
    let chunk = std::mem::replace(pending, rest.trim_start().to_string());
    let chunk = chunk.trim().to_string();
    if chunk.is_empty() {
        None
    } else {
        Some(chunk)
    }
}

#[cfg(test)]
mod stream_tests {
    use super::*;

    #[test]
    fn no_chunk_until_a_sentence_completes() {
        let mut buf = "An unfinished thought".to_string();
        assert!(take_stream_chunk(&mut buf, 300).is_none());
        assert_eq!(buf, "An unfinished thought"); // untouched
    }

    #[test]
    fn complete_sentence_pops_and_leaves_the_rest() {
        let mut buf = "First sentence done. And then".to_string();
        let c = take_stream_chunk(&mut buf, 300).unwrap();
        assert_eq!(c, "First sentence done.");
        assert_eq!(buf, "And then");
    }

    #[test]
    fn overlong_buffer_splits_at_whitespace_without_a_boundary() {
        let mut buf = "word ".repeat(40); // 200 chars, no terminator
        let c = take_stream_chunk(&mut buf, 100).unwrap();
        assert!(c.len() <= 100);
        assert!(c.ends_with("word"));
        assert!(!buf.starts_with(' '));
    }
}
