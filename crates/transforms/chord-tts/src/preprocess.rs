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
