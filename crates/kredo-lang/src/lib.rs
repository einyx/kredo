//! Script and language detection for routers.
//!
//! Routers like the `jeff` model detect the script/language of a request and
//! delegate to a concrete tag such as `kredo:en` or `kredo:multilingual`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Script {
    Latn,
    Cyrl,
    Grek,
    Hani,
    Hira,
    Hang,
    Arab,
    Hebr,
    Thai,
    Other,
}

/// The resolved route of a request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route {
    pub script: Script,
    /// Best-effort BCP-47 hint, e.g. `en`, `de`, `zh`.
    pub language: String,
    /// The concrete tag the router picked.
    pub tag: String,
}

/// Count how many letters of `text` fall into each Unicode script range.
fn letter_histogram(text: &str) -> [(Script, usize); 9] {
    let mut latn = 0usize;
    let mut cyrl = 0;
    let mut grek = 0;
    let mut hani = 0;
    let mut hira = 0;
    let mut hang = 0;
    let mut arab = 0;
    let mut hebr = 0;
    let mut thai = 0;
    for c in text.chars() {
        if c.is_ascii_alphabetic()
            || ('À'..='ž').contains(&c)
            || ('Ă'..='ǿ').contains(&c)
            || ('Ā'..='ɏ').contains(&c)
        {
            latn += 1;
        } else if ('\u{0400}'..='\u{052F}').contains(&c) {
            cyrl += 1;
        } else if ('\u{0370}'..='\u{03FF}').contains(&c) {
            grek += 1;
        } else if ('\u{3400}'..='\u{9FFF}').contains(&c) {
            hani += 1;
        } else if ('\u{3040}'..='\u{309F}').contains(&c) {
            hira += 1;
        } else if ('\u{AC00}'..='\u{D7AF}').contains(&c) {
            hang += 1;
        } else if ('\u{0600}'..='\u{06FF}').contains(&c) {
            arab += 1;
        } else if ('\u{0590}'..='\u{05FF}').contains(&c) {
            hebr += 1;
        } else if ('\u{0E00}'..='\u{0E7F}').contains(&c) {
            thai += 1;
        }
    }
    [
        (Script::Latn, latn),
        (Script::Cyrl, cyrl),
        (Script::Grek, grek),
        (Script::Hani, hani),
        (Script::Hira, hira),
        (Script::Hang, hang),
        (Script::Arab, arab),
        (Script::Hebr, hebr),
        (Script::Thai, thai),
    ]
}

/// Detect the dominant script of `text`.
pub fn detect_script(text: &str) -> Script {
    let hist = letter_histogram(text);
    hist.into_iter()
        .max_by_key(|&(_, n)| n)
        .filter(|&(_, n)| n > 0)
        .map(|(s, _)| s)
        .unwrap_or(Script::Other)
}

/// Best-effort language hint within the Latin script.
fn latin_language(text: &str) -> String {
    let lower = text.to_lowercase();
    let has = |subs: &[&str]| subs.iter().any(|s| lower.contains(s));
    // Tiny frequency cue table; good enough for routing hints.
    if has(&[" the ", " and ", "ing "]) || lower.split_whitespace().count() <= 2 {
        "en".into()
    } else if has(&[" der ", " die ", " das ", " und ", " nicht "]) {
        "de".into()
    } else if has(&[" le ", " la ", " les ", " et ", " est ", " une "]) {
        "fr".into()
    } else if has(&[" el ", " la ", " los ", " que ", " una "]) {
        "es".into()
    } else if has(&[" il ", " lo ", " che ", " non ", " una "]) {
        "it".into()
    } else if has(&[" não ", " com ", " uma ", " para "]) {
        "pt".into()
    } else {
        "en".into()
    }
}

/// Route a state text to a concrete model tag.
///
/// `english_tag` / `multilingual_tag` are full names such as `kredo:en` and
/// `kredo:multilingual`. English and short CJK-free Latin text go to the
/// English model; everything else goes multilingual.
pub fn route(text: &str, english_tag: &str, multilingual_tag: &str) -> Route {
    let script = detect_script(text);
    let (language, tag) = match script {
        Script::Latn => {
            let lang = latin_language(text);
            let tag = if lang == "en" {
                english_tag
            } else {
                multilingual_tag
            };
            (lang, tag)
        }
        Script::Cyrl => ("ru".into(), multilingual_tag),
        Script::Grek => ("el".into(), multilingual_tag),
        Script::Hani => ("zh".into(), multilingual_tag),
        Script::Hira => ("ja".into(), multilingual_tag),
        Script::Hang => ("ko".into(), multilingual_tag),
        Script::Arab => ("ar".into(), multilingual_tag),
        Script::Hebr => ("he".into(), multilingual_tag),
        Script::Thai => ("th".into(), multilingual_tag),
        Script::Other => ("und".into(), multilingual_tag),
    };
    Route {
        script,
        language,
        tag: tag.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_english_to_en() {
        let r = route(
            "I was charged twice for my subscription.",
            "kredo:en",
            "jeff:multi",
        );
        assert_eq!(r.tag, "kredo:en");
        assert_eq!(r.language, "en");
    }

    #[test]
    fn routes_cyrillic_to_multilingual() {
        let r = route("Меня дважды списали за подписку.", "kredo:en", "jeff:multi");
        assert_eq!(r.script, Script::Cyrl);
        assert_eq!(r.tag, "jeff:multi");
        assert_eq!(r.language, "ru");
    }

    #[test]
    fn routes_hiragana_to_japanese() {
        let r = route("ひらがなのテキストです。", "kredo:en", "jeff:multi");
        assert_eq!(r.script, Script::Hira);
        assert_eq!(r.language, "ja");
    }

    #[test]
    fn detects_script() {
        assert_eq!(detect_script("hello"), Script::Latn);
        assert_eq!(detect_script("你好世界"), Script::Hani);
        assert_eq!(detect_script("12345 !!"), Script::Other);
    }
}
