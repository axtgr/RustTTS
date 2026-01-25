//! Normalization rules.

use crate::num2words;
use lazy_static::lazy_static;
use regex::Regex;
use tts_core::{Lang, TtsResult};

/// A text normalization rule.
pub trait Rule: Send + Sync + std::fmt::Debug {
    /// Get the rule name.
    fn name(&self) -> &str;

    /// Check if this rule applies to the given language.
    fn applies_to(&self, lang: Lang) -> bool;

    /// Apply the rule to the input text.
    fn apply(&self, input: &str, lang: Lang) -> TtsResult<String>;
}

/// Create the default set of normalization rules.
///
/// Order matters! More specific patterns should be matched first.
pub fn default_rules() -> Vec<Box<dyn Rule>> {
    vec![
        Box::new(WhitespaceRule),
        Box::new(UnicodeNormalizationRule),
        // Process structured patterns first (before plain numbers)
        Box::new(DateRule),
        Box::new(TimeRule),
        Box::new(CurrencyRule),
        Box::new(UnitRule), // includes %, °C, etc.
        // Abbreviations before plain numbers
        Box::new(AbbreviationRule),
        // Plain numbers last
        Box::new(NumberRule),
        // Symbol replacement at the end
        Box::new(SymbolRule),
    ]
}

/// Normalize whitespace (collapse multiple spaces, trim).
#[derive(Debug)]
pub struct WhitespaceRule;

impl Rule for WhitespaceRule {
    fn name(&self) -> &str {
        "whitespace"
    }

    fn applies_to(&self, _lang: Lang) -> bool {
        true
    }

    fn apply(&self, input: &str, _lang: Lang) -> TtsResult<String> {
        // Collapse multiple whitespace into single spaces and trim
        let result: String = input.split_whitespace().collect::<Vec<_>>().join(" ");
        Ok(result)
    }
}

/// Unicode normalization (NFC form).
#[derive(Debug)]
pub struct UnicodeNormalizationRule;

impl Rule for UnicodeNormalizationRule {
    fn name(&self) -> &str {
        "unicode_normalization"
    }

    fn applies_to(&self, _lang: Lang) -> bool {
        true
    }

    fn apply(&self, input: &str, _lang: Lang) -> TtsResult<String> {
        // Basic Unicode cleanup
        // Note: Full NFC normalization would require the `unicode-normalization` crate
        // For now, just handle common cases
        let result = input
            .replace('\u{00A0}', " ") // Non-breaking space -> regular space
            .replace(['\u{2019}', '\u{2018}'], "'") // Left single quote -> apostrophe
            .replace(['\u{201C}', '\u{201D}'], "\"") // Right double quote -> straight quote
            .replace('\u{2014}', " - ") // Em dash -> hyphen with spaces
            .replace('\u{2013}', "-") // En dash -> hyphen
            .replace('\u{2026}', "..."); // Ellipsis -> three dots

        Ok(result)
    }
}

/// Symbol normalization (replace or remove special symbols).
#[derive(Debug)]
pub struct SymbolRule;

impl Rule for SymbolRule {
    fn name(&self) -> &str {
        "symbol"
    }

    fn applies_to(&self, _lang: Lang) -> bool {
        true
    }

    fn apply(&self, input: &str, _lang: Lang) -> TtsResult<String> {
        let mut result = String::with_capacity(input.len());

        for c in input.chars() {
            match c {
                // Keep alphanumeric, whitespace, and basic punctuation
                _ if c.is_alphanumeric() => result.push(c),
                ' ' | '\t' | '\n' | '\r' => result.push(c),
                '.' | ',' | '!' | '?' | ':' | ';' | '-' | '\'' | '"' | '(' | ')' => result.push(c),
                // Replace common symbols
                '@' => result.push_str(" at "),
                '&' => result.push_str(" and "),
                '%' => result.push_str(" percent "),
                '+' => result.push_str(" plus "),
                '=' => result.push_str(" equals "),
                '#' => result.push_str(" number "),
                // Skip other symbols
                _ => {}
            }
        }

        Ok(result)
    }
}

// ============================================================================
// NumberRule - Convert numbers to words
// ============================================================================

lazy_static! {
    // Match integers (with optional sign) - using word boundaries instead of lookaround
    static ref NUMBER_RE: Regex = Regex::new(r"\b(-?\d{1,15})\b").unwrap();
    // Match ordinals like "1st", "2nd", "3rd", "4th" or "1-й", "2-я"
    static ref ORDINAL_EN_RE: Regex = Regex::new(r"\b(\d+)(st|nd|rd|th)\b").unwrap();
    static ref ORDINAL_RU_RE: Regex = Regex::new(r"\b(\d+)[-]?(й|я|е|го|му|ом|ый|ая|ое|ые)\b").unwrap();
    // Match decimal numbers
    static ref DECIMAL_RE: Regex = Regex::new(r"(\d+)[.,](\d+)").unwrap();
}

/// Number normalization rule.
#[derive(Debug)]
pub struct NumberRule;

impl Rule for NumberRule {
    fn name(&self) -> &str {
        "number"
    }

    fn applies_to(&self, _lang: Lang) -> bool {
        true
    }

    fn apply(&self, input: &str, lang: Lang) -> TtsResult<String> {
        let mut result = input.to_string();

        // Process ordinals first (before plain numbers)
        match lang {
            Lang::En => {
                result = ORDINAL_EN_RE
                    .replace_all(&result, |caps: &regex::Captures| {
                        let num: i64 = caps[1].parse().unwrap_or(0);
                        num2words::ordinal_to_words(num, lang)
                    })
                    .to_string();
            }
            Lang::Ru | Lang::Mixed => {
                result = ORDINAL_RU_RE
                    .replace_all(&result, |caps: &regex::Captures| {
                        let num: i64 = caps[1].parse().unwrap_or(0);
                        num2words::ordinal_to_words(num, lang)
                    })
                    .to_string();
            }
        }

        // Process decimal numbers
        result = DECIMAL_RE
            .replace_all(&result, |caps: &regex::Captures| {
                let integer: i64 = caps[1].parse().unwrap_or(0);
                let decimal = &caps[2];
                let int_words = num2words::num_to_words(integer, lang);

                // Read decimal part digit by digit
                let dec_words: Vec<String> = decimal
                    .chars()
                    .map(|c| {
                        let d = c.to_digit(10).unwrap_or(0) as i64;
                        num2words::num_to_words(d, lang)
                    })
                    .collect();

                let point_word = match lang {
                    Lang::Ru | Lang::Mixed => "целых",
                    Lang::En => "point",
                };

                format!("{} {} {}", int_words, point_word, dec_words.join(" "))
            })
            .to_string();

        // Process plain integers
        result = NUMBER_RE
            .replace_all(&result, |caps: &regex::Captures| {
                let num: i64 = caps[1].parse().unwrap_or(0);
                num2words::num_to_words(num, lang)
            })
            .to_string();

        Ok(result)
    }
}

// ============================================================================
// DateRule - Normalize dates
// ============================================================================

lazy_static! {
    // DD.MM.YYYY or DD/MM/YYYY
    static ref DATE_DMY_RE: Regex = Regex::new(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b").unwrap();
    // YYYY-MM-DD (ISO format)
    static ref DATE_ISO_RE: Regex = Regex::new(r"\b(\d{4})-(\d{2})-(\d{2})\b").unwrap();
}

const RU_MONTHS: [&str; 12] = [
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
];

const EN_MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

/// Date normalization rule.
#[derive(Debug)]
pub struct DateRule;

impl Rule for DateRule {
    fn name(&self) -> &str {
        "date"
    }

    fn applies_to(&self, _lang: Lang) -> bool {
        true
    }

    fn apply(&self, input: &str, lang: Lang) -> TtsResult<String> {
        let mut result = input.to_string();

        // Process DD.MM.YYYY format
        result = DATE_DMY_RE
            .replace_all(&result, |caps: &regex::Captures| {
                let day: usize = caps[1].parse().unwrap_or(1);
                let month: usize = caps[2].parse().unwrap_or(1);
                let year: i64 = caps[3].parse().unwrap_or(2000);

                let month_name = match lang {
                    Lang::Ru | Lang::Mixed => RU_MONTHS.get(month.saturating_sub(1)).unwrap_or(&""),
                    Lang::En => EN_MONTHS.get(month.saturating_sub(1)).unwrap_or(&""),
                };

                let day_words = num2words::num_to_words(day as i64, lang);
                let year_words = num2words::num_to_words(year, lang);

                match lang {
                    Lang::Ru | Lang::Mixed => {
                        format!("{} {} {} года", day_words, month_name, year_words)
                    }
                    Lang::En => format!("{} {}, {}", month_name, day_words, year_words),
                }
            })
            .to_string();

        // Process ISO format YYYY-MM-DD
        result = DATE_ISO_RE
            .replace_all(&result, |caps: &regex::Captures| {
                let year: i64 = caps[1].parse().unwrap_or(2000);
                let month: usize = caps[2].parse().unwrap_or(1);
                let day: usize = caps[3].parse().unwrap_or(1);

                let month_name = match lang {
                    Lang::Ru | Lang::Mixed => RU_MONTHS.get(month.saturating_sub(1)).unwrap_or(&""),
                    Lang::En => EN_MONTHS.get(month.saturating_sub(1)).unwrap_or(&""),
                };

                let day_words = num2words::num_to_words(day as i64, lang);
                let year_words = num2words::num_to_words(year, lang);

                match lang {
                    Lang::Ru | Lang::Mixed => {
                        format!("{} {} {} года", day_words, month_name, year_words)
                    }
                    Lang::En => format!("{} {}, {}", month_name, day_words, year_words),
                }
            })
            .to_string();

        Ok(result)
    }
}

// ============================================================================
// TimeRule - Normalize time expressions
// ============================================================================

lazy_static! {
    // HH:MM or HH:MM:SS
    static ref TIME_RE: Regex = Regex::new(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b").unwrap();
}

/// Time normalization rule.
#[derive(Debug)]
pub struct TimeRule;

impl Rule for TimeRule {
    fn name(&self) -> &str {
        "time"
    }

    fn applies_to(&self, _lang: Lang) -> bool {
        true
    }

    fn apply(&self, input: &str, lang: Lang) -> TtsResult<String> {
        let result = TIME_RE
            .replace_all(input, |caps: &regex::Captures| {
                let hours: i64 = caps[1].parse().unwrap_or(0);
                let minutes: i64 = caps[2].parse().unwrap_or(0);
                let seconds: Option<i64> = caps.get(3).and_then(|m| m.as_str().parse().ok());

                let hours_words = num2words::num_to_words(hours, lang);
                let minutes_words = num2words::num_to_words(minutes, lang);

                match lang {
                    Lang::Ru | Lang::Mixed => {
                        let hour_suffix = ru_hour_suffix(hours);
                        let minute_suffix = ru_minute_suffix(minutes);

                        if let Some(secs) = seconds {
                            let secs_words = num2words::num_to_words(secs, lang);
                            let sec_suffix = ru_second_suffix(secs);
                            format!(
                                "{} {} {} {} {} {}",
                                hours_words,
                                hour_suffix,
                                minutes_words,
                                minute_suffix,
                                secs_words,
                                sec_suffix
                            )
                        } else {
                            format!(
                                "{} {} {} {}",
                                hours_words, hour_suffix, minutes_words, minute_suffix
                            )
                        }
                    }
                    Lang::En => {
                        if let Some(secs) = seconds {
                            let secs_words = num2words::num_to_words(secs, lang);
                            format!(
                                "{} hours {} minutes {} seconds",
                                hours_words, minutes_words, secs_words
                            )
                        } else {
                            format!("{} hours {} minutes", hours_words, minutes_words)
                        }
                    }
                }
            })
            .to_string();

        Ok(result)
    }
}

fn ru_hour_suffix(n: i64) -> &'static str {
    let n = n.abs() % 100;
    if (11..=19).contains(&n) {
        return "часов";
    }
    match n % 10 {
        1 => "час",
        2..=4 => "часа",
        _ => "часов",
    }
}

fn ru_minute_suffix(n: i64) -> &'static str {
    let n = n.abs() % 100;
    if (11..=19).contains(&n) {
        return "минут";
    }
    match n % 10 {
        1 => "минута",
        2..=4 => "минуты",
        _ => "минут",
    }
}

fn ru_second_suffix(n: i64) -> &'static str {
    let n = n.abs() % 100;
    if (11..=19).contains(&n) {
        return "секунд";
    }
    match n % 10 {
        1 => "секунда",
        2..=4 => "секунды",
        _ => "секунд",
    }
}

// ============================================================================
// CurrencyRule - Normalize currency expressions
// ============================================================================

lazy_static! {
    // Russian: 100 руб, 100р, 100 рублей
    // NOTE: Longer alternatives must come FIRST (рублей before руб)
    static ref RU_RUB_RE: Regex = Regex::new(r"(\d+)\s*(?:рублей|рубля|рубль|руб\.?|р\.)").unwrap();
    // Russian: 100$, $100
    static ref RU_USD_RE: Regex = Regex::new(r"(\d+)\s*\$|\$\s*(\d+)").unwrap();
    // Russian: 100€, €100
    static ref RU_EUR_RE: Regex = Regex::new(r"(\d+)\s*€|€\s*(\d+)").unwrap();
    // English: $100
    static ref EN_USD_RE: Regex = Regex::new(r"\$\s*(\d+(?:\.\d{2})?)").unwrap();
    // English: €100
    static ref EN_EUR_RE: Regex = Regex::new(r"€\s*(\d+(?:\.\d{2})?)").unwrap();
}

/// Currency normalization rule.
#[derive(Debug)]
pub struct CurrencyRule;

impl Rule for CurrencyRule {
    fn name(&self) -> &str {
        "currency"
    }

    fn applies_to(&self, _lang: Lang) -> bool {
        true
    }

    fn apply(&self, input: &str, lang: Lang) -> TtsResult<String> {
        let mut result = input.to_string();

        match lang {
            Lang::Ru | Lang::Mixed => {
                // Russian rubles
                result = RU_RUB_RE
                    .replace_all(&result, |caps: &regex::Captures| {
                        let num: i64 = caps[1].parse().unwrap_or(0);
                        let words = num2words::num_to_words(num, lang);
                        let suffix = ru_ruble_suffix(num);
                        format!("{} {}", words, suffix)
                    })
                    .to_string();

                // USD with $ symbol
                result = RU_USD_RE
                    .replace_all(&result, |caps: &regex::Captures| {
                        let num: i64 = caps
                            .get(1)
                            .or_else(|| caps.get(2))
                            .and_then(|m| m.as_str().parse().ok())
                            .unwrap_or(0);
                        let words = num2words::num_to_words(num, lang);
                        let suffix = ru_dollar_suffix(num);
                        format!("{} {}", words, suffix)
                    })
                    .to_string();

                // EUR with € symbol
                result = RU_EUR_RE
                    .replace_all(&result, |caps: &regex::Captures| {
                        let num: i64 = caps
                            .get(1)
                            .or_else(|| caps.get(2))
                            .and_then(|m| m.as_str().parse().ok())
                            .unwrap_or(0);
                        let words = num2words::num_to_words(num, lang);
                        format!("{} евро", words)
                    })
                    .to_string();
            }
            Lang::En => {
                // USD
                result = EN_USD_RE
                    .replace_all(&result, |caps: &regex::Captures| {
                        let num_str = &caps[1];
                        if num_str.contains('.') {
                            let parts: Vec<&str> = num_str.split('.').collect();
                            let dollars: i64 = parts[0].parse().unwrap_or(0);
                            let cents: i64 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
                            let d_words = num2words::num_to_words(dollars, lang);
                            let c_words = num2words::num_to_words(cents, lang);
                            format!("{} dollars {} cents", d_words, c_words)
                        } else {
                            let num: i64 = num_str.parse().unwrap_or(0);
                            let words = num2words::num_to_words(num, lang);
                            if num == 1 {
                                format!("{} dollar", words)
                            } else {
                                format!("{} dollars", words)
                            }
                        }
                    })
                    .to_string();

                // EUR
                result = EN_EUR_RE
                    .replace_all(&result, |caps: &regex::Captures| {
                        let num: i64 = caps[1].parse().unwrap_or(0);
                        let words = num2words::num_to_words(num, lang);
                        if num == 1 {
                            format!("{} euro", words)
                        } else {
                            format!("{} euros", words)
                        }
                    })
                    .to_string();
            }
        }

        Ok(result)
    }
}

fn ru_ruble_suffix(n: i64) -> &'static str {
    let n = n.abs() % 100;
    if (11..=19).contains(&n) {
        return "рублей";
    }
    match n % 10 {
        1 => "рубль",
        2..=4 => "рубля",
        _ => "рублей",
    }
}

fn ru_dollar_suffix(n: i64) -> &'static str {
    let n = n.abs() % 100;
    if (11..=19).contains(&n) {
        return "долларов";
    }
    match n % 10 {
        1 => "доллар",
        2..=4 => "доллара",
        _ => "долларов",
    }
}

// ============================================================================
// UnitRule - Normalize measurement units
// ============================================================================

lazy_static! {
    // Metric units - using word boundaries
    static ref KM_RE: Regex = Regex::new(r"(\d+)\s*(?:км|km)\.?\b").unwrap();
    static ref M_RE: Regex = Regex::new(r"(\d+)\s*(?:м|m)\b").unwrap();
    static ref CM_RE: Regex = Regex::new(r"(\d+)\s*(?:см|cm)\.?\b").unwrap();
    static ref MM_RE: Regex = Regex::new(r"(\d+)\s*(?:мм|mm)\.?\b").unwrap();
    static ref KG_RE: Regex = Regex::new(r"(\d+)\s*(?:кг|kg)\.?\b").unwrap();
    static ref G_RE: Regex = Regex::new(r"(\d+)\s*(?:г|g)\b").unwrap();
    static ref L_RE: Regex = Regex::new(r"(\d+)\s*(?:л|l)\b").unwrap();
    static ref ML_RE: Regex = Regex::new(r"(\d+)\s*(?:мл|ml)\.?\b").unwrap();
    // Temperature
    static ref CELSIUS_RE: Regex = Regex::new(r"([+-]?\d+)\s*°?\s*[CС]\b").unwrap();
    static ref FAHRENHEIT_RE: Regex = Regex::new(r"([+-]?\d+)\s*°?\s*F\b").unwrap();
    // Percentage
    static ref PERCENT_RE: Regex = Regex::new(r"(\d+)\s*%").unwrap();
}

/// Unit normalization rule.
#[derive(Debug)]
pub struct UnitRule;

impl Rule for UnitRule {
    fn name(&self) -> &str {
        "unit"
    }

    fn applies_to(&self, _lang: Lang) -> bool {
        true
    }

    fn apply(&self, input: &str, lang: Lang) -> TtsResult<String> {
        let mut result = input.to_string();

        // Kilometers
        result = KM_RE
            .replace_all(&result, |caps: &regex::Captures| {
                let num: i64 = caps[1].parse().unwrap_or(0);
                let words = num2words::num_to_words(num, lang);
                match lang {
                    Lang::Ru | Lang::Mixed => format!("{} {}", words, ru_km_suffix(num)),
                    Lang::En => format!("{} kilometers", words),
                }
            })
            .to_string();

        // Meters
        result = M_RE
            .replace_all(&result, |caps: &regex::Captures| {
                let num: i64 = caps[1].parse().unwrap_or(0);
                let words = num2words::num_to_words(num, lang);
                match lang {
                    Lang::Ru | Lang::Mixed => format!("{} {}", words, ru_meter_suffix(num)),
                    Lang::En => format!("{} meters", words),
                }
            })
            .to_string();

        // Centimeters
        result = CM_RE
            .replace_all(&result, |caps: &regex::Captures| {
                let num: i64 = caps[1].parse().unwrap_or(0);
                let words = num2words::num_to_words(num, lang);
                match lang {
                    Lang::Ru | Lang::Mixed => format!("{} {}", words, ru_cm_suffix(num)),
                    Lang::En => format!("{} centimeters", words),
                }
            })
            .to_string();

        // Kilograms
        result = KG_RE
            .replace_all(&result, |caps: &regex::Captures| {
                let num: i64 = caps[1].parse().unwrap_or(0);
                let words = num2words::num_to_words(num, lang);
                match lang {
                    Lang::Ru | Lang::Mixed => format!("{} {}", words, ru_kg_suffix(num)),
                    Lang::En => format!("{} kilograms", words),
                }
            })
            .to_string();

        // Liters
        result = L_RE
            .replace_all(&result, |caps: &regex::Captures| {
                let num: i64 = caps[1].parse().unwrap_or(0);
                let words = num2words::num_to_words(num, lang);
                match lang {
                    Lang::Ru | Lang::Mixed => format!("{} {}", words, ru_liter_suffix(num)),
                    Lang::En => format!("{} liters", words),
                }
            })
            .to_string();

        // Celsius
        result = CELSIUS_RE
            .replace_all(&result, |caps: &regex::Captures| {
                let num: i64 = caps[1].parse().unwrap_or(0);
                let words = num2words::num_to_words(num, lang);
                match lang {
                    Lang::Ru | Lang::Mixed => format!("{} {}", words, ru_degree_suffix(num)),
                    Lang::En => format!("{} degrees Celsius", words),
                }
            })
            .to_string();

        // Percentage
        result = PERCENT_RE
            .replace_all(&result, |caps: &regex::Captures| {
                let num: i64 = caps[1].parse().unwrap_or(0);
                let words = num2words::num_to_words(num, lang);
                match lang {
                    Lang::Ru | Lang::Mixed => format!("{} {}", words, ru_percent_suffix(num)),
                    Lang::En => format!("{} percent", words),
                }
            })
            .to_string();

        Ok(result)
    }
}

fn ru_km_suffix(n: i64) -> &'static str {
    let n = n.abs() % 100;
    if (11..=19).contains(&n) {
        return "километров";
    }
    match n % 10 {
        1 => "километр",
        2..=4 => "километра",
        _ => "километров",
    }
}

fn ru_meter_suffix(n: i64) -> &'static str {
    let n = n.abs() % 100;
    if (11..=19).contains(&n) {
        return "метров";
    }
    match n % 10 {
        1 => "метр",
        2..=4 => "метра",
        _ => "метров",
    }
}

fn ru_cm_suffix(n: i64) -> &'static str {
    let n = n.abs() % 100;
    if (11..=19).contains(&n) {
        return "сантиметров";
    }
    match n % 10 {
        1 => "сантиметр",
        2..=4 => "сантиметра",
        _ => "сантиметров",
    }
}

fn ru_kg_suffix(n: i64) -> &'static str {
    let n = n.abs() % 100;
    if (11..=19).contains(&n) {
        return "килограммов";
    }
    match n % 10 {
        1 => "килограмм",
        2..=4 => "килограмма",
        _ => "килограммов",
    }
}

fn ru_liter_suffix(n: i64) -> &'static str {
    let n = n.abs() % 100;
    if (11..=19).contains(&n) {
        return "литров";
    }
    match n % 10 {
        1 => "литр",
        2..=4 => "литра",
        _ => "литров",
    }
}

fn ru_degree_suffix(n: i64) -> &'static str {
    let n = n.abs() % 100;
    if (11..=19).contains(&n) {
        return "градусов Цельсия";
    }
    match n % 10 {
        1 => "градус Цельсия",
        2..=4 => "градуса Цельсия",
        _ => "градусов Цельсия",
    }
}

fn ru_percent_suffix(n: i64) -> &'static str {
    let n = n.abs() % 100;
    if (11..=19).contains(&n) {
        return "процентов";
    }
    match n % 10 {
        1 => "процент",
        2..=4 => "процента",
        _ => "процентов",
    }
}

// ============================================================================
// AbbreviationRule - Expand common abbreviations
// ============================================================================

lazy_static! {
    // Russian abbreviations with word boundaries
    // Address-related
    static ref RU_ABBR_UL: Regex = Regex::new(r"\bул\.\s*").unwrap();
    static ref RU_ABBR_D: Regex = Regex::new(r"\bд\.\s*(\d+)").unwrap();
    static ref RU_ABBR_KV: Regex = Regex::new(r"\bкв\.\s*(\d+)").unwrap();
    static ref RU_ABBR_STR: Regex = Regex::new(r"\bстр\.\s*(\d+)").unwrap();
    static ref RU_ABBR_KORP: Regex = Regex::new(r"\bкорп\.\s*(\d+)").unwrap();
    static ref RU_ABBR_PER: Regex = Regex::new(r"\bпер\.\s*").unwrap();
    static ref RU_ABBR_PR: Regex = Regex::new(r"\bпр\.\s*").unwrap();
    static ref RU_ABBR_PL: Regex = Regex::new(r"\bпл\.\s*").unwrap();

    // Common Russian abbreviations
    // "г." before city name - capture the following capital letter
    static ref RU_ABBR_G: Regex = Regex::new(r"\bг\.\s*([А-ЯЁ])").unwrap();
    static ref RU_ABBR_GG: Regex = Regex::new(r"\bгг\.\s*").unwrap();  // годы
    // "в." before century number - capture the number
    static ref RU_ABBR_V: Regex = Regex::new(r"\bв\.\s*(\d)").unwrap();
    static ref RU_ABBR_N_E: Regex = Regex::new(r"\bн\.э\.").unwrap();  // нашей эры
    static ref RU_ABBR_DO_N_E: Regex = Regex::new(r"\bдо\s*н\.э\.").unwrap();  // до нашей эры
    static ref RU_ABBR_T_D: Regex = Regex::new(r"\bи\s*т\.д\.").unwrap();  // и так далее
    static ref RU_ABBR_T_P: Regex = Regex::new(r"\bи\s*т\.п\.").unwrap();  // и тому подобное
    static ref RU_ABBR_DR: Regex = Regex::new(r"\bи\s*др\.").unwrap();  // и другие
    static ref RU_ABBR_PR2: Regex = Regex::new(r"\bи\s*пр\.").unwrap();  // и прочее
    static ref RU_ABBR_SM: Regex = Regex::new(r"\bсм\.\s*").unwrap();  // смотри
    static ref RU_ABBR_SR: Regex = Regex::new(r"\bср\.\s*").unwrap();  // сравни
    static ref RU_ABBR_T_E: Regex = Regex::new(r"\bт\.е\.").unwrap();  // то есть
    static ref RU_ABBR_T_K: Regex = Regex::new(r"\bт\.к\.").unwrap();  // так как
    static ref RU_ABBR_T_N: Regex = Regex::new(r"\bт\.н\.").unwrap();  // так называемый

    // Titles
    static ref RU_ABBR_MR: Regex = Regex::new(r"\bг-н\b").unwrap();  // господин
    static ref RU_ABBR_MRS: Regex = Regex::new(r"\bг-жа\b").unwrap();  // госпожа

    // English abbreviations
    static ref EN_ABBR_MR: Regex = Regex::new(r"\bMr\.\s*").unwrap();
    static ref EN_ABBR_MRS: Regex = Regex::new(r"\bMrs\.\s*").unwrap();
    static ref EN_ABBR_MS: Regex = Regex::new(r"\bMs\.\s*").unwrap();
    static ref EN_ABBR_DR: Regex = Regex::new(r"\bDr\.\s*").unwrap();
    static ref EN_ABBR_PROF: Regex = Regex::new(r"\bProf\.\s*").unwrap();
    static ref EN_ABBR_ST: Regex = Regex::new(r"\bSt\.\s*").unwrap();
    static ref EN_ABBR_AVE: Regex = Regex::new(r"\bAve\.\s*").unwrap();
    static ref EN_ABBR_BLVD: Regex = Regex::new(r"\bBlvd\.\s*").unwrap();
    static ref EN_ABBR_APT: Regex = Regex::new(r"\bApt\.\s*").unwrap();
    static ref EN_ABBR_ETC: Regex = Regex::new(r"\betc\.\s*").unwrap();
    static ref EN_ABBR_IE: Regex = Regex::new(r"\bi\.e\.\s*").unwrap();
    static ref EN_ABBR_EG: Regex = Regex::new(r"\be\.g\.\s*").unwrap();
    static ref EN_ABBR_VS: Regex = Regex::new(r"\bvs\.\s*").unwrap();
}

/// Abbreviation expansion rule.
#[derive(Debug)]
pub struct AbbreviationRule;

impl Rule for AbbreviationRule {
    fn name(&self) -> &str {
        "abbreviation"
    }

    fn applies_to(&self, _lang: Lang) -> bool {
        true
    }

    fn apply(&self, input: &str, lang: Lang) -> TtsResult<String> {
        let mut result = input.to_string();

        match lang {
            Lang::Ru | Lang::Mixed => {
                // Address abbreviations
                result = RU_ABBR_UL.replace_all(&result, "улица ").to_string();
                result = RU_ABBR_D.replace_all(&result, "дом $1").to_string();
                result = RU_ABBR_KV.replace_all(&result, "квартира $1").to_string();
                result = RU_ABBR_STR.replace_all(&result, "строение $1").to_string();
                result = RU_ABBR_KORP.replace_all(&result, "корпус $1").to_string();
                result = RU_ABBR_PER.replace_all(&result, "переулок ").to_string();
                result = RU_ABBR_PR.replace_all(&result, "проспект ").to_string();
                result = RU_ABBR_PL.replace_all(&result, "площадь ").to_string();

                // Common abbreviations
                // "г. Москва" -> "город Москва" (preserve capital letter)
                result = RU_ABBR_G.replace_all(&result, "город $1").to_string();
                result = RU_ABBR_GG.replace_all(&result, "годы ").to_string();
                // "в. 19" -> "век 19" (preserve number)
                result = RU_ABBR_V.replace_all(&result, "век $1").to_string();
                result = RU_ABBR_DO_N_E
                    .replace_all(&result, "до нашей эры")
                    .to_string();
                result = RU_ABBR_N_E.replace_all(&result, "нашей эры").to_string();
                result = RU_ABBR_T_D.replace_all(&result, "и так далее").to_string();
                result = RU_ABBR_T_P
                    .replace_all(&result, "и тому подобное")
                    .to_string();
                result = RU_ABBR_DR.replace_all(&result, "и другие").to_string();
                result = RU_ABBR_PR2.replace_all(&result, "и прочее").to_string();
                result = RU_ABBR_SM.replace_all(&result, "смотри ").to_string();
                result = RU_ABBR_SR.replace_all(&result, "сравни ").to_string();
                result = RU_ABBR_T_E.replace_all(&result, "то есть").to_string();
                result = RU_ABBR_T_K.replace_all(&result, "так как").to_string();
                result = RU_ABBR_T_N
                    .replace_all(&result, "так называемый")
                    .to_string();

                // Titles
                result = RU_ABBR_MR.replace_all(&result, "господин").to_string();
                result = RU_ABBR_MRS.replace_all(&result, "госпожа").to_string();
            }
            Lang::En => {
                // Titles
                result = EN_ABBR_MR.replace_all(&result, "Mister ").to_string();
                result = EN_ABBR_MRS.replace_all(&result, "Misses ").to_string();
                result = EN_ABBR_MS.replace_all(&result, "Miss ").to_string();
                result = EN_ABBR_DR.replace_all(&result, "Doctor ").to_string();
                result = EN_ABBR_PROF.replace_all(&result, "Professor ").to_string();

                // Address
                result = EN_ABBR_ST.replace_all(&result, "Street ").to_string();
                result = EN_ABBR_AVE.replace_all(&result, "Avenue ").to_string();
                result = EN_ABBR_BLVD.replace_all(&result, "Boulevard ").to_string();
                result = EN_ABBR_APT.replace_all(&result, "Apartment ").to_string();

                // Common
                result = EN_ABBR_ETC.replace_all(&result, "etcetera ").to_string();
                result = EN_ABBR_IE.replace_all(&result, "that is ").to_string();
                result = EN_ABBR_EG.replace_all(&result, "for example ").to_string();
                result = EN_ABBR_VS.replace_all(&result, "versus ").to_string();
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitespace_rule() {
        let rule = WhitespaceRule;
        let result = rule.apply("  hello   world  ", Lang::En).unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_unicode_normalization_rule() {
        let rule = UnicodeNormalizationRule;

        // Test em dash
        let result = rule.apply("hello—world", Lang::En).unwrap();
        assert_eq!(result, "hello - world");

        // Test smart quotes
        let result = rule.apply("\u{201C}hello\u{201D}", Lang::En).unwrap();
        assert_eq!(result, "\"hello\"");
    }

    #[test]
    fn test_symbol_rule() {
        let rule = SymbolRule;

        let result = rule.apply("hello@world", Lang::En).unwrap();
        assert_eq!(result, "hello at world");

        let result = rule.apply("100%", Lang::En).unwrap();
        assert_eq!(result, "100 percent ");
    }

    #[test]
    fn test_number_rule_russian() {
        let rule = NumberRule;

        let result = rule.apply("У меня 5 яблок", Lang::Ru).unwrap();
        assert_eq!(result, "У меня пять яблок");

        let result = rule.apply("Цена 100 рублей", Lang::Ru).unwrap();
        assert!(result.contains("сто"));

        let result = rule.apply("Год 2024", Lang::Ru).unwrap();
        assert!(result.contains("две тысячи двадцать четыре"));
    }

    #[test]
    fn test_number_rule_english() {
        let rule = NumberRule;

        let result = rule.apply("I have 5 apples", Lang::En).unwrap();
        assert_eq!(result, "I have five apples");

        let result = rule.apply("The price is 100 dollars", Lang::En).unwrap();
        assert!(result.contains("one hundred"));
    }

    #[test]
    fn test_date_rule_russian() {
        let rule = DateRule;

        let result = rule.apply("Дата: 15.03.2024", Lang::Ru).unwrap();
        assert!(result.contains("пятнадцать"));
        assert!(result.contains("марта"));
        assert!(result.contains("года"));
    }

    #[test]
    fn test_date_rule_english() {
        let rule = DateRule;

        let result = rule.apply("Date: 2024-03-15", Lang::En).unwrap();
        assert!(result.contains("March"));
        assert!(result.contains("fifteen"));
    }

    #[test]
    fn test_time_rule_russian() {
        let rule = TimeRule;

        let result = rule.apply("Время: 14:30", Lang::Ru).unwrap();
        assert!(result.contains("четырнадцать"));
        assert!(result.contains("час"));
        assert!(result.contains("тридцать"));
        assert!(result.contains("минут"));
    }

    #[test]
    fn test_currency_rule_russian() {
        let rule = CurrencyRule;

        // Test short form "руб"
        let result = rule.apply("Цена: 100 руб", Lang::Ru).unwrap();
        assert!(result.contains("сто"));
        assert!(result.contains("рублей"));

        // Test full form "рублей" - regression test for double suffix bug
        let result = rule.apply("1500 рублей", Lang::Ru).unwrap();
        assert_eq!(result, "одна тысяча пятьсот рублей");
        assert!(!result.contains("рублейлей"), "double suffix bug detected");

        // Test "рубля" form (2-4)
        let result = rule.apply("2 рубля", Lang::Ru).unwrap();
        assert!(result.contains("два"));
        assert!(result.contains("рубля"));

        // Test USD
        let result = rule.apply("Цена: $50", Lang::Ru).unwrap();
        assert!(result.contains("пятьдесят"));
        assert!(result.contains("доллар"));
    }

    #[test]
    fn test_currency_rule_english() {
        let rule = CurrencyRule;

        let result = rule.apply("Price: $100", Lang::En).unwrap();
        assert!(result.contains("one hundred"));
        assert!(result.contains("dollars"));
    }

    #[test]
    fn test_unit_rule_russian() {
        let rule = UnitRule;

        let result = rule.apply("Расстояние: 5 км", Lang::Ru).unwrap();
        assert!(result.contains("пять"));
        assert!(result.contains("километров"));

        let result = rule.apply("Вес: 10 кг", Lang::Ru).unwrap();
        assert!(result.contains("десять"));
        assert!(result.contains("килограмм"));

        let result = rule.apply("Температура: 25°C", Lang::Ru).unwrap();
        assert!(result.contains("двадцать пять"));
        assert!(result.contains("градус"));
    }

    #[test]
    fn test_unit_rule_english() {
        let rule = UnitRule;

        let result = rule.apply("Distance: 5 km", Lang::En).unwrap();
        assert!(result.contains("five"));
        assert!(result.contains("kilometers"));
    }

    #[test]
    fn test_percent_rule() {
        let rule = UnitRule;

        let result = rule.apply("Скидка 50%", Lang::Ru).unwrap();
        assert!(result.contains("пятьдесят"));
        assert!(result.contains("процентов"));

        let result = rule.apply("Discount 50%", Lang::En).unwrap();
        assert!(result.contains("fifty"));
        assert!(result.contains("percent"));
    }

    #[test]
    fn test_abbreviation_rule_russian() {
        let rule = AbbreviationRule;

        // Address abbreviations
        let result = rule.apply("ул. Ленина, д. 5, кв. 10", Lang::Ru).unwrap();
        assert!(result.contains("улица"));
        assert!(result.contains("дом 5"));
        assert!(result.contains("квартира 10"));

        // Common abbreviations
        let result = rule.apply("и т.д.", Lang::Ru).unwrap();
        assert!(result.contains("и так далее"));

        let result = rule.apply("т.е. это важно", Lang::Ru).unwrap();
        assert!(result.contains("то есть"));

        let result = rule.apply("г. Москва", Lang::Ru).unwrap();
        assert!(result.contains("город Москва"));

        // Titles
        let result = rule.apply("г-н Иванов", Lang::Ru).unwrap();
        assert!(result.contains("господин"));
    }

    #[test]
    fn test_abbreviation_rule_english() {
        let rule = AbbreviationRule;

        // Titles
        let result = rule.apply("Mr. Smith", Lang::En).unwrap();
        assert!(result.contains("Mister"));

        let result = rule.apply("Dr. Jones", Lang::En).unwrap();
        assert!(result.contains("Doctor"));

        // Address
        let result = rule.apply("123 Main St.", Lang::En).unwrap();
        assert!(result.contains("Street"));

        // Common
        let result = rule.apply("e.g. examples", Lang::En).unwrap();
        assert!(result.contains("for example"));

        let result = rule.apply("cats, dogs, etc.", Lang::En).unwrap();
        assert!(result.contains("etcetera"));
    }
}
