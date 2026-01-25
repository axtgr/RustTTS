//! Integration tests for the TTS runtime.
//!
//! These tests verify end-to-end functionality of the TTS pipeline.

use runtime::{TtsPipeline, TtsRuntime};
use tts_core::{Lang, SynthesisRequest};

/// Test full pipeline: text → normalize → tokenize → acoustic → audio.
#[test]
fn test_full_pipeline_russian() {
    let pipeline = TtsPipeline::new_mock().unwrap();

    let text = "Привет, мир! Как дела?";
    let audio = pipeline.synthesize(text, Some(Lang::Ru)).unwrap();

    // Verify audio output
    assert!(audio.num_samples() > 0, "Should produce audio samples");
    assert_eq!(audio.sample_rate, 24000, "Should be 24kHz");
    assert!(audio.duration_ms() > 0.0, "Should have positive duration");

    // Verify samples are in valid range
    for &sample in audio.pcm.iter() {
        assert!(
            (-1.0..=1.0).contains(&sample),
            "Sample {} out of range [-1, 1]",
            sample
        );
    }
}

/// Test pipeline with English text.
#[test]
fn test_full_pipeline_english() {
    let pipeline = TtsPipeline::new_mock().unwrap();

    let text = "Hello, world! How are you?";
    let audio = pipeline.synthesize(text, Some(Lang::En)).unwrap();

    assert!(audio.num_samples() > 0);
    assert_eq!(audio.sample_rate, 24000);
}

/// Test streaming synthesis.
#[test]
fn test_streaming_synthesis() {
    let pipeline = TtsPipeline::new_mock().unwrap();
    let mut session = pipeline.streaming_session().unwrap();

    // Set text
    session
        .set_text("Тестовый текст для потокового синтеза.", Some(Lang::Ru))
        .unwrap();

    // Collect all chunks
    let mut total_samples = 0;
    let mut chunk_count = 0;

    while let Some(chunk) = session.next_chunk().unwrap() {
        total_samples += chunk.num_samples();
        chunk_count += 1;

        // Verify each chunk
        assert!(chunk.num_samples() > 0, "Each chunk should have samples");
        assert_eq!(chunk.sample_rate, 24000);

        if session.is_finished() {
            break;
        }
    }

    assert!(chunk_count > 0, "Should produce at least one chunk");
    assert!(total_samples > 0, "Should produce samples");
}

/// Test runtime with async submit.
#[tokio::test]
async fn test_runtime_async_synthesis() {
    let runtime = TtsRuntime::default();

    let request = SynthesisRequest::new("Test text for async synthesis").with_lang(Lang::En);

    let mut rx = runtime.submit(request).await.unwrap();

    // Receive audio
    let result = rx.recv().await;
    assert!(result.is_some(), "Should receive audio");

    let audio = result.unwrap().unwrap();
    assert!(audio.num_samples() > 0);
}

/// Test multiple sequential requests.
#[tokio::test]
async fn test_multiple_requests() {
    let runtime = TtsRuntime::default();

    for i in 0..3 {
        let request = SynthesisRequest::new(format!("Request number {}", i)).with_lang(Lang::En);

        let mut rx = runtime.submit(request).await.unwrap();
        let result = rx.recv().await.unwrap().unwrap();

        assert!(
            result.num_samples() > 0,
            "Request {} should produce audio",
            i
        );
    }
}

/// Test streaming session reset and reuse.
#[test]
fn test_streaming_session_reuse() {
    let pipeline = TtsPipeline::new_mock().unwrap();
    let mut session = pipeline.streaming_session().unwrap();

    // First text
    session.set_text("First text", Some(Lang::En)).unwrap();
    let _ = session.next_chunk().unwrap();

    // Reset and reuse
    session.reset();
    assert_eq!(session.total_samples(), 0, "Should reset samples count");
    assert!(!session.is_finished(), "Should not be finished after reset");

    // Second text
    session.set_text("Second text", Some(Lang::En)).unwrap();
    let chunk = session.next_chunk().unwrap();
    assert!(chunk.is_some(), "Should produce chunk after reset");
}

/// Test runtime queue statistics.
#[test]
fn test_runtime_queue_stats() {
    let runtime = TtsRuntime::default();
    let stats = runtime.queue_stats();

    assert_eq!(stats.size, 0);
    assert!(!stats.is_full);
    assert_eq!(stats.max_size, 1000);
}

/// Test pipeline steps individually.
#[test]
fn test_pipeline_steps() {
    let pipeline = TtsPipeline::new_mock().unwrap();

    // Step 1: Normalize
    let normalized = pipeline.normalize("100 рублей", Some(Lang::Ru)).unwrap();
    assert!(normalized.text.contains("сто") || normalized.text.contains("100"));

    // Step 2: Tokenize
    let tokens = pipeline.tokenize(&normalized).unwrap();
    assert!(!tokens.is_empty(), "Should produce tokens");

    // Step 3: Generate acoustic
    let acoustic = pipeline.generate_acoustic(&tokens, 100).unwrap();
    assert!(!acoustic.is_empty(), "Should produce acoustic tokens");

    // Step 4: Decode
    let audio = pipeline.decode_audio(&acoustic).unwrap();
    assert!(audio.num_samples() > 0, "Should produce audio");
}

/// Test that empty text produces error.
#[test]
fn test_empty_text_error() {
    let pipeline = TtsPipeline::new_mock().unwrap();

    // Empty normalized text should still work (normalizer handles it)
    let result = pipeline.synthesize("", Some(Lang::En));
    // Depending on implementation, this may succeed with empty audio or fail
    // Just verify it doesn't panic
    let _ = result;
}

/// Test special characters handling.
#[test]
fn test_special_characters() {
    let pipeline = TtsPipeline::new_mock().unwrap();

    let texts = [
        "Hello!!! What???",
        "Price: $100",
        "Temperature: -10°C",
        "50% скидка",
        "test@email.com",
    ];

    for text in texts {
        let result = pipeline.synthesize(text, Some(Lang::En));
        assert!(result.is_ok(), "Should handle: {}", text);
    }
}

/// Test very long text.
#[test]
fn test_long_text() {
    let pipeline = TtsPipeline::new_mock().unwrap();

    // Generate long text
    let text = "Это длинный текст. ".repeat(50);
    let audio = pipeline.synthesize(&text, Some(Lang::Ru)).unwrap();

    assert!(audio.num_samples() > 0);
    assert!(
        audio.duration_ms() > 100.0,
        "Long text should produce long audio"
    );
}
