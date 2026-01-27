use crate::pipeline::TtsPipeline;
use tracing::info;
use tts_core::TtsResult;

/// Warm up model cache with sample synthesis runs.
///
/// This prevents JIT compilation from affecting first user requests.
/// Performs 3 warmup runs to ensure model kernels are compiled.
///
/// # Arguments
/// * `pipeline` - TTS pipeline to warm up
/// * `sample_text` - Sample text for warmup synthesis
///
/// # Example
/// ```ignore
/// use runtime::warm::warm_model_cache;
/// warm_model_cache(&pipeline, "Привет! Это пример текста.")?;
/// ```
pub fn warm_model_cache(pipeline: &TtsPipeline, sample_text: &str) -> TtsResult<()> {
    info!("Warming model cache with sample synthesis...");

    // 3 warmup runs
    for i in 1..=3 {
        info!("Warmup run {}/3", i);
        if let Err(e) = pipeline.synthesize_with_speaker(sample_text, None, None) {
            tracing::warn!("Warmup run {} failed: {}", i, e);
            // Continue anyway - some runs might fail during warmup
        }
    }

    info!("Cold start pre-warming complete");
    Ok(())
}

/// Warm up cache with Russian and English sample texts.
pub fn warm_model_cache_multilingual(pipeline: &TtsPipeline) -> TtsResult<()> {
    info!("Warming multilingual model cache...");

    let samples = vec![
        ("ru", "Привет! Это пример текста для прогрева системы."),
        ("en", "Hello! This is a sample text for system warmup."),
        (
            "ru",
            "Ниже представлен план цикла статей о том, как создать собственную модель.",
        ),
        (
            "en",
            "Below is the plan for a series of articles on how to create your own model.",
        ),
    ];

    for (lang, text) in samples {
        info!(lang, "Warming with {} text", lang);
        let lang_iso = translate_lang(lang);
        if let Some(lang_value) = lang_iso {
            if let Err(e) = pipeline.synthesize_with_speaker(text, Some(lang_value), None) {
                tracing::warn!("Warmup for {} failed: {}", lang, e);
            }
        } else if let Err(e) = pipeline.synthesize_with_speaker(text, None, None) {
            tracing::warn!("Warmup for {} failed: {}", lang, e);
        }
    }

    info!("Multilingual cold start pre-warming complete");
    Ok(())
}

fn translate_lang(lang: &str) -> Option<tts_core::Lang> {
    match lang {
        "ru" => Some(tts_core::Lang::Ru),
        "en" => Some(tts_core::Lang::En),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lang_translation() {
        assert_eq!(translate_lang("ru"), Some(tts_core::Lang::Ru));
        assert_eq!(translate_lang("en"), Some(tts_core::Lang::En));
        assert_eq!(translate_lang("de"), None);
    }
}
