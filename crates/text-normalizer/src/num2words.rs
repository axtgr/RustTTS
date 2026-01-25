//! Number to words conversion for Russian and English.

use tts_core::Lang;

/// Convert a number to words.
pub fn num_to_words(num: i64, lang: Lang) -> String {
    match lang {
        Lang::Ru | Lang::Mixed => num_to_words_ru(num),
        Lang::En => num_to_words_en(num),
    }
}

/// Convert an ordinal number to words.
pub fn ordinal_to_words(num: i64, lang: Lang) -> String {
    match lang {
        Lang::Ru | Lang::Mixed => ordinal_to_words_ru(num),
        Lang::En => ordinal_to_words_en(num),
    }
}

// ============================================================================
// Russian number conversion
// ============================================================================

const RU_ONES: [&str; 20] = [
    "",
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
];

const RU_ONES_FEM: [&str; 3] = ["", "одна", "две"];

const RU_TENS: [&str; 10] = [
    "",
    "",
    "двадцать",
    "тридцать",
    "сорок",
    "пятьдесят",
    "шестьдесят",
    "семьдесят",
    "восемьдесят",
    "девяносто",
];

const RU_HUNDREDS: [&str; 10] = [
    "",
    "сто",
    "двести",
    "триста",
    "четыреста",
    "пятьсот",
    "шестьсот",
    "семьсот",
    "восемьсот",
    "девятьсот",
];

/// Russian plural forms for thousands.
fn ru_thousands(n: i64) -> &'static str {
    let n = n.abs() % 100;
    if (11..=19).contains(&n) {
        return "тысяч";
    }
    match n % 10 {
        1 => "тысяча",
        2..=4 => "тысячи",
        _ => "тысяч",
    }
}

/// Russian plural forms for millions.
fn ru_millions(n: i64) -> &'static str {
    let n = n.abs() % 100;
    if (11..=19).contains(&n) {
        return "миллионов";
    }
    match n % 10 {
        1 => "миллион",
        2..=4 => "миллиона",
        _ => "миллионов",
    }
}

/// Russian plural forms for billions.
fn ru_billions(n: i64) -> &'static str {
    let n = n.abs() % 100;
    if (11..=19).contains(&n) {
        return "миллиардов";
    }
    match n % 10 {
        1 => "миллиард",
        2..=4 => "миллиарда",
        _ => "миллиардов",
    }
}

/// Convert hundreds part (0-999) to Russian words.
fn hundreds_to_words_ru(n: i64, feminine: bool) -> String {
    let n = n.unsigned_abs() as usize;
    if n == 0 {
        return String::new();
    }

    let mut parts = Vec::new();

    let h = n / 100;
    if h > 0 {
        parts.push(RU_HUNDREDS[h].to_string());
    }

    let remainder = n % 100;
    if remainder > 0 {
        if remainder < 20 {
            if feminine && remainder <= 2 {
                parts.push(RU_ONES_FEM[remainder].to_string());
            } else {
                parts.push(RU_ONES[remainder].to_string());
            }
        } else {
            let tens = remainder / 10;
            let ones = remainder % 10;
            parts.push(RU_TENS[tens].to_string());
            if ones > 0 {
                if feminine && ones <= 2 {
                    parts.push(RU_ONES_FEM[ones].to_string());
                } else {
                    parts.push(RU_ONES[ones].to_string());
                }
            }
        }
    }

    parts.join(" ")
}

/// Convert a number to Russian words.
pub fn num_to_words_ru(num: i64) -> String {
    if num == 0 {
        return "ноль".to_string();
    }

    let mut parts = Vec::new();
    let mut n = num;

    if n < 0 {
        parts.push("минус".to_string());
        n = -n;
    }

    // Billions
    let billions = n / 1_000_000_000;
    if billions > 0 {
        parts.push(hundreds_to_words_ru(billions, false));
        parts.push(ru_billions(billions).to_string());
    }
    n %= 1_000_000_000;

    // Millions
    let millions = n / 1_000_000;
    if millions > 0 {
        parts.push(hundreds_to_words_ru(millions, false));
        parts.push(ru_millions(millions).to_string());
    }
    n %= 1_000_000;

    // Thousands (feminine in Russian)
    let thousands = n / 1_000;
    if thousands > 0 {
        parts.push(hundreds_to_words_ru(thousands, true));
        parts.push(ru_thousands(thousands).to_string());
    }
    n %= 1_000;

    // Ones
    if n > 0 || parts.is_empty() {
        parts.push(hundreds_to_words_ru(n, false));
    }

    parts
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Russian ordinal suffixes.
fn ordinal_to_words_ru(num: i64) -> String {
    // Simplified ordinal conversion
    let base = num_to_words_ru(num);

    // For complex numbers, just add "-й" suffix approximation
    // Full implementation would need grammatical case handling
    if num == 1 {
        return "первый".to_string();
    }
    if num == 2 {
        return "второй".to_string();
    }
    if num == 3 {
        return "третий".to_string();
    }

    // Simple heuristic for other numbers
    let last = base.split_whitespace().last().unwrap_or(&base);
    match last {
        s if s.ends_with("ь") => format!("{}ой", &base[..base.len() - 2]),
        s if s.ends_with("а") => format!("{}ой", &base[..base.len() - 2]),
        s if s.ends_with("е") => format!("{}ой", &base[..base.len() - 2]),
        _ => format!("{}-й", base),
    }
}

// ============================================================================
// English number conversion
// ============================================================================

const EN_ONES: [&str; 20] = [
    "",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];

const EN_TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

/// Convert hundreds part (0-999) to English words.
fn hundreds_to_words_en(n: i64) -> String {
    let n = n.unsigned_abs() as usize;
    if n == 0 {
        return String::new();
    }

    let mut parts = Vec::new();

    let h = n / 100;
    if h > 0 {
        parts.push(format!("{} hundred", EN_ONES[h]));
    }

    let remainder = n % 100;
    if remainder > 0 {
        if remainder < 20 {
            parts.push(EN_ONES[remainder].to_string());
        } else {
            let tens = remainder / 10;
            let ones = remainder % 10;
            if ones > 0 {
                parts.push(format!("{}-{}", EN_TENS[tens], EN_ONES[ones]));
            } else {
                parts.push(EN_TENS[tens].to_string());
            }
        }
    }

    parts.join(" ")
}

/// Convert a number to English words.
pub fn num_to_words_en(num: i64) -> String {
    if num == 0 {
        return "zero".to_string();
    }

    let mut parts = Vec::new();
    let mut n = num;

    if n < 0 {
        parts.push("minus".to_string());
        n = -n;
    }

    // Billions
    let billions = n / 1_000_000_000;
    if billions > 0 {
        parts.push(hundreds_to_words_en(billions));
        parts.push("billion".to_string());
    }
    n %= 1_000_000_000;

    // Millions
    let millions = n / 1_000_000;
    if millions > 0 {
        parts.push(hundreds_to_words_en(millions));
        parts.push("million".to_string());
    }
    n %= 1_000_000;

    // Thousands
    let thousands = n / 1_000;
    if thousands > 0 {
        parts.push(hundreds_to_words_en(thousands));
        parts.push("thousand".to_string());
    }
    n %= 1_000;

    // Ones
    if n > 0 || parts.is_empty() {
        parts.push(hundreds_to_words_en(n));
    }

    parts
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// English ordinal conversion.
fn ordinal_to_words_en(num: i64) -> String {
    // Get the last digit for suffix determination
    let last_digit = (num % 10) as i32;
    let last_two = (num % 100) as i32;

    // Special cases for 11, 12, 13 (all use 'th')
    if (11..=13).contains(&last_two) {
        let base = num_to_words_en(num);
        return format!("{base}th");
    }

    // For compound numbers (21st, 22nd, etc.), convert the base and handle the last word
    let base = num_to_words_en(num);

    // Simple numbers
    if num == 1 {
        return "first".to_string();
    }
    if num == 2 {
        return "second".to_string();
    }
    if num == 3 {
        return "third".to_string();
    }
    if num == 5 {
        return "fifth".to_string();
    }
    if num == 8 {
        return "eighth".to_string();
    }
    if num == 9 {
        return "ninth".to_string();
    }
    if num == 12 {
        return "twelfth".to_string();
    }

    // For compound numbers, replace the last word with its ordinal form
    if base.contains('-') || base.contains(' ') {
        // Find the last word
        let parts: Vec<&str> = if base.contains('-') {
            base.rsplitn(2, '-').collect()
        } else {
            base.rsplitn(2, ' ').collect()
        };

        if parts.len() == 2 {
            let last_word = parts[0];
            let prefix = parts[1];
            let separator = if base.contains('-') && !prefix.contains(' ') {
                "-"
            } else {
                " "
            };

            let ordinal_suffix = match last_digit {
                1 => "first",
                2 => "second",
                3 => "third",
                _ => {
                    // Handle special cases for the last word
                    return match last_word {
                        "five" => format!("{prefix}{separator}fifth"),
                        "eight" => format!("{prefix}{separator}eighth"),
                        "nine" => format!("{prefix}{separator}ninth"),
                        "twelve" => format!("{prefix}{separator}twelfth"),
                        w if w.ends_with('y') => {
                            format!("{prefix}{separator}{}ieth", &w[..w.len() - 1])
                        }
                        w if w.ends_with('e') => {
                            format!("{prefix}{separator}{}th", &w[..w.len() - 1])
                        }
                        _ => format!("{prefix}{separator}{last_word}th"),
                    };
                }
            };
            return format!("{prefix}{separator}{ordinal_suffix}");
        }
    }

    // For numbers ending in 'y', change to 'ieth'
    if base.ends_with('y') {
        return format!("{}ieth", &base[..base.len() - 1]);
    }

    // For numbers ending in 'e', just add 'th'
    if base.ends_with('e') {
        return format!("{}th", &base[..base.len() - 1]);
    }

    // Default: add 'th'
    format!("{base}th")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_russian_basic() {
        assert_eq!(num_to_words_ru(0), "ноль");
        assert_eq!(num_to_words_ru(1), "один");
        assert_eq!(num_to_words_ru(2), "два");
        assert_eq!(num_to_words_ru(10), "десять");
        assert_eq!(num_to_words_ru(11), "одиннадцать");
        assert_eq!(num_to_words_ru(19), "девятнадцать");
        assert_eq!(num_to_words_ru(20), "двадцать");
        assert_eq!(num_to_words_ru(21), "двадцать один");
        assert_eq!(num_to_words_ru(100), "сто");
        assert_eq!(num_to_words_ru(101), "сто один");
        assert_eq!(num_to_words_ru(111), "сто одиннадцать");
        assert_eq!(num_to_words_ru(200), "двести");
    }

    #[test]
    fn test_russian_thousands() {
        assert_eq!(num_to_words_ru(1000), "одна тысяча");
        assert_eq!(num_to_words_ru(2000), "две тысячи");
        assert_eq!(num_to_words_ru(5000), "пять тысяч");
        assert_eq!(num_to_words_ru(11000), "одиннадцать тысяч");
        assert_eq!(num_to_words_ru(21000), "двадцать одна тысяча");
        assert_eq!(num_to_words_ru(1001), "одна тысяча один");
        assert_eq!(num_to_words_ru(2345), "две тысячи триста сорок пять");
    }

    #[test]
    fn test_russian_millions() {
        assert_eq!(num_to_words_ru(1_000_000), "один миллион");
        assert_eq!(num_to_words_ru(2_000_000), "два миллиона");
        assert_eq!(num_to_words_ru(5_000_000), "пять миллионов");
    }

    #[test]
    fn test_russian_negative() {
        assert_eq!(num_to_words_ru(-1), "минус один");
        assert_eq!(num_to_words_ru(-100), "минус сто");
    }

    #[test]
    fn test_english_basic() {
        assert_eq!(num_to_words_en(0), "zero");
        assert_eq!(num_to_words_en(1), "one");
        assert_eq!(num_to_words_en(10), "ten");
        assert_eq!(num_to_words_en(11), "eleven");
        assert_eq!(num_to_words_en(20), "twenty");
        assert_eq!(num_to_words_en(21), "twenty-one");
        assert_eq!(num_to_words_en(100), "one hundred");
        assert_eq!(num_to_words_en(101), "one hundred one");
        assert_eq!(num_to_words_en(111), "one hundred eleven");
    }

    #[test]
    fn test_english_thousands() {
        assert_eq!(num_to_words_en(1000), "one thousand");
        assert_eq!(num_to_words_en(2000), "two thousand");
        assert_eq!(num_to_words_en(1001), "one thousand one");
        assert_eq!(
            num_to_words_en(2345),
            "two thousand three hundred forty-five"
        );
    }

    #[test]
    fn test_english_millions() {
        assert_eq!(num_to_words_en(1_000_000), "one million");
        assert_eq!(
            num_to_words_en(2_500_000),
            "two million five hundred thousand"
        );
    }

    #[test]
    fn test_english_ordinals() {
        assert_eq!(ordinal_to_words_en(1), "first");
        assert_eq!(ordinal_to_words_en(2), "second");
        assert_eq!(ordinal_to_words_en(3), "third");
        assert_eq!(ordinal_to_words_en(4), "fourth");
        assert_eq!(ordinal_to_words_en(5), "fifth");
        assert_eq!(ordinal_to_words_en(20), "twentieth");
        assert_eq!(ordinal_to_words_en(21), "twenty-first");
    }

    #[test]
    fn test_russian_ordinals() {
        assert_eq!(ordinal_to_words_ru(1), "первый");
        assert_eq!(ordinal_to_words_ru(2), "второй");
        assert_eq!(ordinal_to_words_ru(3), "третий");
    }
}
