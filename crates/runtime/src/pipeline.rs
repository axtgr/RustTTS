//! TTS Pipeline - full text-to-speech processing pipeline.
//!
//! Combines all components: normalizer → tokenizer → acoustic model → codec.

use std::path::Path;
use std::sync::Arc;

use candle_core::Device;
use tracing::{debug, info, instrument};

use acoustic_model::{CodePredictor, CodePredictorConfig, Model as AcousticModel, SamplingConfig};
use audio_codec_12hz::{Codec12Hz, DEFAULT_CROSSFADE_MS, StreamingDecoder};
use text_normalizer::Normalizer;
use text_tokenizer::{MockTokenizer, Qwen3TTSTokens, Tokenizer};
use tts_core::{
    AudioChunk, AudioCodec, Lang, NormText, SynthesisRequest, TextNormalizer, TextTokenizer,
    TtsError, TtsResult,
};

/// Backend selection for the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineBackend {
    /// Use mock components (for testing without model weights).
    Mock,
    /// Use real neural network components.
    Neural,
}

/// Configuration for the TTS pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Backend selection.
    pub backend: PipelineBackend,
    /// Default language for text normalization.
    pub default_lang: Lang,
    /// Crossfade duration in milliseconds for streaming.
    pub crossfade_ms: f32,
    /// Default chunk size in tokens for streaming.
    pub chunk_tokens: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Default speaker for CustomVoice models (e.g., "vivian", "ryan").
    pub default_speaker: Option<String>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            backend: PipelineBackend::Mock,
            default_lang: Lang::Ru,
            crossfade_ms: DEFAULT_CROSSFADE_MS,
            chunk_tokens: 10,
            max_seq_len: 4096,
            default_speaker: None,
        }
    }
}

impl PipelineConfig {
    /// Create a mock pipeline configuration.
    pub fn mock() -> Self {
        Self {
            backend: PipelineBackend::Mock,
            ..Default::default()
        }
    }

    /// Create a neural pipeline configuration.
    pub fn neural() -> Self {
        Self {
            backend: PipelineBackend::Neural,
            ..Default::default()
        }
    }
}

/// Tokenizer backend - either mock or real.
enum TokenizerBackend {
    Mock(MockTokenizer),
    Real(Box<Tokenizer>),
}

impl std::fmt::Debug for TokenizerBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mock(_) => write!(f, "MockTokenizer"),
            Self::Real(_) => write!(f, "Tokenizer"),
        }
    }
}

/// Acoustic model backend - either mock or real neural model.
enum AcousticBackend {
    /// Mock acoustic generation (no model weights needed).
    Mock,
    /// Real neural acoustic model with optional CodePredictor.
    Neural {
        model: Arc<AcousticModel>,
        code_predictor: Option<Arc<CodePredictor>>,
    },
}

impl std::fmt::Debug for AcousticBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mock => write!(f, "MockAcoustic"),
            Self::Neural { code_predictor, .. } => {
                if code_predictor.is_some() {
                    write!(f, "NeuralAcoustic(with CodePredictor)")
                } else {
                    write!(f, "NeuralAcoustic(zeroth only)")
                }
            }
        }
    }
}

/// The main TTS pipeline combining all processing stages.
///
/// Supports both mock (for testing) and real (with model weights) backends.
pub struct TtsPipeline {
    normalizer: Normalizer,
    tokenizer: TokenizerBackend,
    acoustic: AcousticBackend,
    codec: Codec12Hz,
    config: PipelineConfig,
    /// Special token IDs for Qwen3-TTS.
    special_tokens: Qwen3TTSTokens,
    /// Device for tensor operations.
    device: Device,
}

impl std::fmt::Debug for TtsPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TtsPipeline")
            .field("config", &self.config)
            .finish()
    }
}

impl TtsPipeline {
    /// Create a new mock pipeline for testing.
    pub fn new_mock() -> TtsResult<Self> {
        let config = PipelineConfig::mock();

        info!("Creating mock TTS pipeline");

        Ok(Self {
            normalizer: Normalizer::new(),
            tokenizer: TokenizerBackend::Mock(MockTokenizer::new(65536)),
            acoustic: AcousticBackend::Mock,
            codec: Codec12Hz::new_mock(),
            config,
            special_tokens: Qwen3TTSTokens::default(),
            device: Device::Cpu,
        })
    }

    /// Load pipeline from pretrained model directories.
    ///
    /// # Arguments
    /// * `talker_dir` - Path to Qwen3-TTS talker model directory
    /// * `tokenizer_dir` - Path to tokenizer directory (or same as talker_dir)
    /// * `codec_dir` - Path to Qwen3-TTS-Tokenizer-12Hz directory
    /// * `device` - Device to load models on (CPU or CUDA)
    #[instrument(skip_all)]
    pub fn from_pretrained(
        talker_dir: impl AsRef<Path>,
        tokenizer_dir: impl AsRef<Path>,
        codec_dir: impl AsRef<Path>,
        device: &Device,
    ) -> TtsResult<Self> {
        let talker_dir = talker_dir.as_ref();
        let tokenizer_dir = tokenizer_dir.as_ref();
        let codec_dir = codec_dir.as_ref();

        info!(
            "Loading pipeline from: talker={}, tokenizer={}, codec={}",
            talker_dir.display(),
            tokenizer_dir.display(),
            codec_dir.display()
        );

        // Load tokenizer
        let tokenizer = Tokenizer::from_pretrained(tokenizer_dir)?;
        info!(vocab_size = tokenizer.vocab_size(), "Tokenizer loaded");

        // Load audio codec
        let codec = Codec12Hz::from_pretrained(codec_dir, device)?;
        info!("Audio codec loaded");

        // Try to load acoustic model if weights exist
        let weights_path = talker_dir.join("model.safetensors");
        let acoustic = if weights_path.exists() {
            info!("Loading acoustic model from {}", talker_dir.display());
            match AcousticModel::from_pretrained(talker_dir, device) {
                Ok(model) => {
                    info!(
                        layers = model.config().num_layers,
                        hidden = model.config().hidden_size,
                        "Acoustic model loaded"
                    );

                    // Try to load CodePredictor from the same weights file
                    let code_predictor =
                        Self::try_load_code_predictor(&weights_path, model.config(), device);

                    AcousticBackend::Neural {
                        model: Arc::new(model),
                        code_predictor,
                    }
                }
                Err(e) => {
                    info!("Failed to load acoustic model: {}, using mock", e);
                    AcousticBackend::Mock
                }
            }
        } else {
            info!("No model.safetensors found, using mock acoustic backend");
            AcousticBackend::Mock
        };

        let config = PipelineConfig::neural();

        info!("Pipeline created with {:?} acoustic backend", acoustic);

        Ok(Self {
            normalizer: Normalizer::new(),
            tokenizer: TokenizerBackend::Real(Box::new(tokenizer)),
            acoustic,
            codec,
            config,
            special_tokens: Qwen3TTSTokens::default(),
            device: device.clone(),
        })
    }

    /// Try to load CodePredictor from the same weights file as the main model.
    fn try_load_code_predictor(
        weights_path: &Path,
        model_config: &acoustic_model::AcousticModelConfig,
        device: &Device,
    ) -> Option<Arc<CodePredictor>> {
        // Create CodePredictor config from main model config
        let cp_config = CodePredictorConfig {
            hidden_size: model_config.hidden_size,
            num_layers: 5, // Qwen3-TTS CodePredictor has 5 layers
            num_attention_heads: model_config.num_attention_heads,
            num_kv_heads: model_config.num_kv_heads,
            intermediate_size: model_config.intermediate_size,
            head_dim: model_config.head_dim,
            num_code_groups: model_config.num_code_groups,
            codebook_size: model_config.codebook_size,
            max_position_embeddings: model_config.max_position_embeddings,
            rope_theta: model_config.rope_theta,
            rms_norm_eps: model_config.rms_norm_eps,
        };

        match CodePredictor::load(weights_path, cp_config, device) {
            Ok(cp) => {
                info!(
                    "CodePredictor loaded: {} layers, {} residual codebooks",
                    5,
                    model_config.num_code_groups - 1
                );
                Some(Arc::new(cp))
            }
            Err(e) => {
                info!(
                    "CodePredictor not found or failed to load: {}, will use zeroth codebook only",
                    e
                );
                None
            }
        }
    }

    /// Create a neural pipeline with loaded models (legacy API).
    ///
    /// Note: Prefer using `from_pretrained()` instead.
    #[instrument(skip(_acoustic_model_path, _tokenizer_path, _codec_path))]
    #[deprecated(since = "0.2.0", note = "Use from_pretrained() instead")]
    pub fn new_neural(
        _acoustic_model_path: impl AsRef<std::path::Path>,
        _tokenizer_path: impl AsRef<std::path::Path>,
        _codec_path: impl AsRef<std::path::Path>,
    ) -> TtsResult<Self> {
        // For now, return mock pipeline
        info!("Legacy neural pipeline API called, using mock");
        Self::new_mock()
    }

    /// Get the pipeline configuration.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Normalize text for the given language.
    pub fn normalize(&self, text: &str, lang: Option<Lang>) -> TtsResult<NormText> {
        let lang = lang.unwrap_or(self.config.default_lang);
        self.normalizer.normalize(text, Some(lang))
    }

    /// Tokenize normalized text.
    pub fn tokenize(&self, text: &NormText) -> TtsResult<Vec<u32>> {
        let seq = match &self.tokenizer {
            TokenizerBackend::Mock(t) => t.encode(text)?,
            TokenizerBackend::Real(t) => t.encode(text)?,
        };
        Ok(seq.ids)
    }

    /// Check if using real (non-mock) tokenizer.
    pub fn has_real_tokenizer(&self) -> bool {
        matches!(self.tokenizer, TokenizerBackend::Real(_))
    }

    /// Check if using real (non-mock) acoustic model.
    pub fn has_real_acoustic(&self) -> bool {
        matches!(self.acoustic, AcousticBackend::Neural { .. })
    }

    /// Check if CodePredictor is loaded for multi-codebook generation.
    pub fn has_code_predictor(&self) -> bool {
        matches!(
            &self.acoustic,
            AcousticBackend::Neural {
                code_predictor: Some(_),
                ..
            }
        )
    }

    /// Check if using real (non-mock) codec.
    pub fn has_real_codec(&self) -> bool {
        self.codec.is_neural()
    }

    /// Check if all components are using real (non-mock) backends.
    pub fn is_fully_neural(&self) -> bool {
        self.has_real_tokenizer() && self.has_real_acoustic() && self.has_real_codec()
    }

    /// Get the device this pipeline is running on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Generate acoustic tokens from text tokens.
    ///
    /// Uses real acoustic model if available, otherwise falls back to mock.
    /// If CodePredictor is available, generates all 16 codebooks;
    /// otherwise, generates only the zeroth codebook.
    pub fn generate_acoustic(&self, text_tokens: &[u32], max_tokens: usize) -> TtsResult<Vec<u32>> {
        if text_tokens.is_empty() {
            return Err(TtsError::invalid_input("empty text tokens"));
        }

        match &self.acoustic {
            AcousticBackend::Neural {
                model,
                code_predictor,
            } => self.generate_acoustic_neural(
                model,
                code_predictor.as_deref(),
                text_tokens,
                max_tokens,
            ),
            AcousticBackend::Mock => self.generate_acoustic_mock(text_tokens, max_tokens),
        }
    }

    /// Generate acoustic tokens using the neural model.
    ///
    /// If CodePredictor is provided, generates all 16 codebooks and returns
    /// them in interleaved format. Otherwise, returns only zeroth codebook.
    fn generate_acoustic_neural(
        &self,
        model: &AcousticModel,
        code_predictor: Option<&CodePredictor>,
        text_tokens: &[u32],
        max_tokens: usize,
    ) -> TtsResult<Vec<u32>> {
        // Build text input sequence with special tokens:
        // [tts_bos] [text_tokens...]
        // Note: codec_bos is passed separately to generate_with_hidden
        let mut text_input_ids = Vec::with_capacity(text_tokens.len() + 1);
        text_input_ids.push(self.special_tokens.tts_bos_token_id);
        text_input_ids.extend_from_slice(text_tokens);

        debug!(
            text_len = text_input_ids.len(),
            codec_bos = self.special_tokens.codec_bos_id,
            codec_eos = self.special_tokens.codec_eos_id,
            max_tokens = max_tokens,
            has_code_predictor = code_predictor.is_some(),
            "Starting neural acoustic generation"
        );

        // Configure sampling
        let sampling_config = SamplingConfig {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            seed: None,
        };

        // Generate zeroth codebook tokens with hidden states
        // text_input_ids go through text_embedding, codec_bos goes through codec_embedding
        let (zeroth_tokens, hidden_states) = model
            .generate_with_hidden(
                &text_input_ids,
                self.special_tokens.codec_bos_id,
                max_tokens,
                sampling_config.clone(),
                Some(self.special_tokens.codec_eos_id),
            )
            .map_err(|e| TtsError::inference(format!("acoustic generation failed: {e}")))?;

        debug!(
            zeroth_tokens = zeroth_tokens.len(),
            hidden_shape = ?hidden_states.dims(),
            "Zeroth codebook generation complete"
        );

        // If no CodePredictor, return only zeroth codebook
        let Some(cp) = code_predictor else {
            debug!("No CodePredictor, returning zeroth codebook only");
            return Ok(zeroth_tokens);
        };

        // Use CodePredictor to generate residual codebooks (1-15)
        if zeroth_tokens.is_empty() {
            debug!("No tokens generated, returning empty");
            return Ok(zeroth_tokens);
        }

        // Create tensor from zeroth codebook tokens
        let zeroth_tensor = candle_core::Tensor::new(zeroth_tokens.as_slice(), &self.device)
            .map_err(|e| TtsError::inference(format!("tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| TtsError::inference(format!("unsqueeze failed: {e}")))?;

        // Predict all codebooks using CodePredictor
        let all_codes = cp
            .predict_from_hidden(&hidden_states, &zeroth_tensor, sampling_config)
            .map_err(|e| TtsError::inference(format!("code prediction failed: {e}")))?;

        // Convert to interleaved format: [frame0_code0, frame0_code1, ..., frame1_code0, ...]
        let (_, num_frames, num_groups) = all_codes
            .dims3()
            .map_err(|e| TtsError::inference(format!("dims3 failed: {e}")))?;

        let all_codes_flat: Vec<u32> = all_codes
            .squeeze(0)
            .map_err(|e| TtsError::inference(format!("squeeze failed: {e}")))?
            .to_vec2()
            .map_err(|e| TtsError::inference(format!("to_vec2 failed: {e}")))?
            .into_iter()
            .flatten()
            .collect();

        debug!(
            num_frames = num_frames,
            num_groups = num_groups,
            total_codes = all_codes_flat.len(),
            "Multi-codebook generation complete"
        );

        Ok(all_codes_flat)
    }

    /// Generate acoustic tokens with speaker for CustomVoice models.
    ///
    /// This builds the proper prompt format following Python implementation:
    /// 1. Role prefix: [im_start, assistant, newline] - goes through text_embedding only
    /// 2. Codec prefix: [think, think_bos, lang_id?, think_eos, speaker_id?, pad, bos]
    ///    with text: [tts_pad, ..., tts_bos] - combined embeddings
    /// 3. Text tokens: text_tokens with codec_pad - combined embeddings
    /// 4. Final: [tts_eos + codec_pad], then [tts_pad + codec_bos] to start generation
    fn generate_acoustic_with_speaker(
        &self,
        model: &AcousticModel,
        code_predictor: Option<&CodePredictor>,
        text_tokens: &[u32],
        speaker: Option<&str>,
        lang: Lang,
        max_tokens: usize,
    ) -> TtsResult<Vec<u32>> {
        use candle_core::Tensor;

        let st = &self.special_tokens;

        // Get speaker token ID (optional - for CustomVoice models)
        let speaker_id = speaker.and_then(|s| st.speaker_ids.by_name(s));

        // Get language token ID
        let lang_name = match lang {
            Lang::Ru => "russian",
            Lang::En => "english",
            Lang::Mixed => "english", // Default to English for mixed
        };
        let lang_id = st.language_ids.by_name(lang_name);

        info!(
            "Building CustomVoice prompt: speaker={:?} (id={:?}), lang={} (id={:?}), text_tokens={}",
            speaker,
            speaker_id,
            lang_name,
            lang_id,
            text_tokens.len()
        );

        // ========== PART 1: Role prefix (text embedding only) ==========
        // [im_start, assistant, newline] - these go through text_embedding only, not combined with codec
        let role_prefix: Vec<u32> = vec![
            st.im_start_token_id,  // 151644
            st.assistant_token_id, // 77091
            st.newline_token_id,   // 198
        ];

        let role_prefix_tensor = Tensor::new(role_prefix.as_slice(), &self.device)
            .map_err(|e| TtsError::inference(format!("tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| TtsError::inference(format!("unsqueeze failed: {e}")))?;

        let role_prefix_embed = model
            .get_text_embedding(&role_prefix_tensor)
            .map_err(|e| TtsError::inference(format!("text embed failed: {e}")))?;

        // ========== PART 2: Codec prefix with combined embeddings ==========
        // Build codec prefix: [think, think_bos, (lang_id), think_eos, (speaker_id), pad, bos]
        let mut codec_prefix: Vec<u32> = vec![st.codec_think_id, st.codec_think_bos_id];
        if let Some(lid) = lang_id {
            codec_prefix.push(lid);
        }
        codec_prefix.push(st.codec_think_eos_id);
        if let Some(sid) = speaker_id {
            codec_prefix.push(sid);
        }
        codec_prefix.push(st.codec_pad_id);
        codec_prefix.push(st.codec_bos_id);

        // Text for codec prefix: all tts_pad except last which is tts_bos
        // But we take [:-1] because the last codec_bos pairs with first text token
        let text_for_codec_prefix: Vec<u32> = codec_prefix
            .iter()
            .take(codec_prefix.len() - 1) // exclude last (codec_bos)
            .map(|_| st.tts_pad_token_id)
            .collect();

        let codec_prefix_for_embed: Vec<u32> = codec_prefix[..codec_prefix.len() - 1].to_vec();

        let codec_prefix_tensor = Tensor::new(codec_prefix_for_embed.as_slice(), &self.device)
            .map_err(|e| TtsError::inference(format!("tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| TtsError::inference(format!("unsqueeze failed: {e}")))?;

        let text_for_codec_prefix_tensor =
            Tensor::new(text_for_codec_prefix.as_slice(), &self.device)
                .map_err(|e| TtsError::inference(format!("tensor creation failed: {e}")))?
                .unsqueeze(0)
                .map_err(|e| TtsError::inference(format!("unsqueeze failed: {e}")))?;

        let codec_prefix_embed = model
            .get_codec_embedding(&codec_prefix_tensor)
            .map_err(|e| TtsError::inference(format!("codec embed failed: {e}")))?;

        let text_for_codec_prefix_embed =
            model
                .get_text_embedding(&text_for_codec_prefix_tensor)
                .map_err(|e| TtsError::inference(format!("text embed failed: {e}")))?;

        // Combine: tts_pad + codec_prefix (element-wise)
        let codec_prefix_combined = (&text_for_codec_prefix_embed + &codec_prefix_embed)
            .map_err(|e| TtsError::inference(format!("embed add failed: {e}")))?;

        // ========== PART 3: Text tokens with codec_pad + first text with codec_bos ==========
        // First text token pairs with codec_bos
        // Rest of text tokens pair with codec_pad
        // Then tts_eos pairs with codec_pad
        // Finally tts_pad pairs with codec_bos (start of generation)

        // Build text sequence: [tts_bos, text_tokens..., tts_eos, tts_pad]
        let mut text_seq: Vec<u32> = Vec::with_capacity(text_tokens.len() + 3);
        text_seq.push(st.tts_bos_token_id);
        text_seq.extend_from_slice(text_tokens);
        text_seq.push(st.tts_eos_token_id);
        text_seq.push(st.tts_pad_token_id);

        // Build codec sequence: [codec_bos, codec_pad..., codec_pad, codec_bos]
        let mut codec_seq: Vec<u32> = Vec::with_capacity(text_tokens.len() + 3);
        codec_seq.push(st.codec_bos_id); // pairs with tts_bos
        for _ in 0..text_tokens.len() {
            codec_seq.push(st.codec_pad_id); // pairs with each text token
        }
        codec_seq.push(st.codec_pad_id); // pairs with tts_eos
        codec_seq.push(st.codec_bos_id); // pairs with tts_pad (start generation)

        let text_seq_tensor = Tensor::new(text_seq.as_slice(), &self.device)
            .map_err(|e| TtsError::inference(format!("tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| TtsError::inference(format!("unsqueeze failed: {e}")))?;

        let codec_seq_tensor = Tensor::new(codec_seq.as_slice(), &self.device)
            .map_err(|e| TtsError::inference(format!("tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| TtsError::inference(format!("unsqueeze failed: {e}")))?;

        let text_seq_embed = model
            .get_text_embedding(&text_seq_tensor)
            .map_err(|e| TtsError::inference(format!("text embed failed: {e}")))?;

        let codec_seq_embed = model
            .get_codec_embedding(&codec_seq_tensor)
            .map_err(|e| TtsError::inference(format!("codec embed failed: {e}")))?;

        // Combine text + codec embeddings
        let text_combined = (&text_seq_embed + &codec_seq_embed)
            .map_err(|e| TtsError::inference(format!("embed add failed: {e}")))?;

        // ========== PART 4: Concatenate all parts ==========
        // [role_prefix_embed] + [codec_prefix_combined] + [text_combined]
        let combined_embeds = Tensor::cat(
            &[role_prefix_embed, codec_prefix_combined, text_combined],
            1,
        )
        .map_err(|e| TtsError::inference(format!("cat failed: {e}")))?;

        debug!(
            role_len = role_prefix.len(),
            codec_prefix_len = codec_prefix_for_embed.len(),
            text_seq_len = text_seq.len(),
            combined_shape = ?combined_embeds.dims(),
            "Combined embeddings built"
        );

        // Configure sampling
        let sampling_config = SamplingConfig {
            temperature: 0.9,
            top_p: 1.0,
            top_k: 50,
            seed: None,
        };

        // Generate from combined embeddings
        let (zeroth_tokens, hidden_states) = model
            .generate_from_embeds(
                &combined_embeds,
                max_tokens,
                sampling_config.clone(),
                Some(st.codec_eos_id),
            )
            .map_err(|e| TtsError::inference(format!("generation failed: {e}")))?;

        debug!(
            zeroth_tokens = zeroth_tokens.len(),
            hidden_shape = ?hidden_states.dims(),
            "Zeroth codebook generation complete"
        );

        // If no CodePredictor, return only zeroth codebook
        let Some(_cp) = code_predictor else {
            return Ok(zeroth_tokens);
        };

        if zeroth_tokens.is_empty() {
            return Ok(zeroth_tokens);
        }

        // Return zeroth codebook only for backward compatibility
        // Use generate_acoustic_multi_codebook for full 16-codebook generation
        Ok(zeroth_tokens)
    }

    /// Generate all 16 codebooks using CodePredictor.
    ///
    /// Returns `[16][seq_len]` array of tokens for all codebooks.
    /// If CodePredictor is not available, only zeroth codebook is filled,
    /// others are set to zero.
    fn generate_acoustic_multi_codebook(
        &self,
        model: &AcousticModel,
        code_predictor: Option<&CodePredictor>,
        text_tokens: &[u32],
        speaker: Option<&str>,
        lang: Lang,
        max_tokens: usize,
    ) -> TtsResult<Vec<Vec<u32>>> {
        use candle_core::Tensor;

        let st = &self.special_tokens;

        // Get speaker and language IDs
        let speaker_id = speaker.and_then(|s| st.speaker_ids.by_name(s));
        let lang_name = match lang {
            Lang::Ru => "russian",
            Lang::En => "english",
            Lang::Mixed => "english",
        };
        let lang_id = st.language_ids.by_name(lang_name);

        // Build prompt (same as generate_acoustic_with_speaker)
        let role_prefix: Vec<u32> = vec![
            st.im_start_token_id,
            st.assistant_token_id,
            st.newline_token_id,
        ];

        let role_prefix_tensor = Tensor::new(role_prefix.as_slice(), &self.device)
            .map_err(|e| TtsError::inference(format!("tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| TtsError::inference(format!("unsqueeze failed: {e}")))?;

        let role_prefix_embed = model
            .get_text_embedding(&role_prefix_tensor)
            .map_err(|e| TtsError::inference(format!("text embed failed: {e}")))?;

        // Build codec prefix
        let mut codec_prefix: Vec<u32> = vec![st.codec_think_id, st.codec_think_bos_id];
        if let Some(lid) = lang_id {
            codec_prefix.push(lid);
        }
        codec_prefix.push(st.codec_think_eos_id);
        if let Some(sid) = speaker_id {
            codec_prefix.push(sid);
        }
        codec_prefix.push(st.codec_pad_id);
        codec_prefix.push(st.codec_bos_id);

        let text_for_codec_prefix: Vec<u32> = codec_prefix
            .iter()
            .take(codec_prefix.len() - 1)
            .map(|_| st.tts_pad_token_id)
            .collect();

        let codec_prefix_for_embed: Vec<u32> = codec_prefix[..codec_prefix.len() - 1].to_vec();

        let codec_prefix_tensor = Tensor::new(codec_prefix_for_embed.as_slice(), &self.device)
            .map_err(|e| TtsError::inference(format!("tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| TtsError::inference(format!("unsqueeze failed: {e}")))?;

        let text_for_codec_prefix_tensor =
            Tensor::new(text_for_codec_prefix.as_slice(), &self.device)
                .map_err(|e| TtsError::inference(format!("tensor creation failed: {e}")))?
                .unsqueeze(0)
                .map_err(|e| TtsError::inference(format!("unsqueeze failed: {e}")))?;

        let codec_prefix_embed = model
            .get_codec_embedding(&codec_prefix_tensor)
            .map_err(|e| TtsError::inference(format!("codec embed failed: {e}")))?;

        let text_for_codec_prefix_embed =
            model
                .get_text_embedding(&text_for_codec_prefix_tensor)
                .map_err(|e| TtsError::inference(format!("text embed failed: {e}")))?;

        let codec_prefix_combined = (&text_for_codec_prefix_embed + &codec_prefix_embed)
            .map_err(|e| TtsError::inference(format!("embed add failed: {e}")))?;

        // Build text sequence
        let mut text_seq: Vec<u32> = Vec::with_capacity(text_tokens.len() + 3);
        text_seq.push(st.tts_bos_token_id);
        text_seq.extend_from_slice(text_tokens);
        text_seq.push(st.tts_eos_token_id);
        text_seq.push(st.tts_pad_token_id);

        let mut codec_seq: Vec<u32> = Vec::with_capacity(text_tokens.len() + 3);
        codec_seq.push(st.codec_bos_id);
        for _ in 0..text_tokens.len() {
            codec_seq.push(st.codec_pad_id);
        }
        codec_seq.push(st.codec_pad_id);
        codec_seq.push(st.codec_bos_id);

        let text_seq_tensor = Tensor::new(text_seq.as_slice(), &self.device)
            .map_err(|e| TtsError::inference(format!("tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| TtsError::inference(format!("unsqueeze failed: {e}")))?;

        let codec_seq_tensor = Tensor::new(codec_seq.as_slice(), &self.device)
            .map_err(|e| TtsError::inference(format!("tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| TtsError::inference(format!("unsqueeze failed: {e}")))?;

        let text_seq_embed = model
            .get_text_embedding(&text_seq_tensor)
            .map_err(|e| TtsError::inference(format!("text embed failed: {e}")))?;

        let codec_seq_embed = model
            .get_codec_embedding(&codec_seq_tensor)
            .map_err(|e| TtsError::inference(format!("codec embed failed: {e}")))?;

        let text_combined = (&text_seq_embed + &codec_seq_embed)
            .map_err(|e| TtsError::inference(format!("embed add failed: {e}")))?;

        let combined_embeds = Tensor::cat(
            &[role_prefix_embed, codec_prefix_combined, text_combined],
            1,
        )
        .map_err(|e| TtsError::inference(format!("cat failed: {e}")))?;

        let sampling_config = SamplingConfig {
            temperature: 0.9,
            top_p: 1.0,
            top_k: 50,
            seed: None,
        };

        // Generate zeroth codebook
        let (zeroth_tokens, hidden_states) = model
            .generate_from_embeds(
                &combined_embeds,
                max_tokens,
                sampling_config.clone(),
                Some(st.codec_eos_id),
            )
            .map_err(|e| TtsError::inference(format!("generation failed: {e}")))?;

        // Filter special tokens from zeroth codebook
        let zeroth_filtered: Vec<u32> = zeroth_tokens
            .iter()
            .filter(|&&t| t < 2048)
            .copied()
            .collect();

        if zeroth_filtered.is_empty() {
            return Err(TtsError::inference(
                "no valid audio tokens generated".to_string(),
            ));
        }

        let seq_len = zeroth_filtered.len();

        // If no CodePredictor, return zeroth + zeros
        let Some(cp) = code_predictor else {
            let mut result = Vec::with_capacity(16);
            result.push(zeroth_filtered);
            for _ in 1..16 {
                result.push(vec![0u32; seq_len]);
            }
            return Ok(result);
        };

        // Use CodePredictor to generate all codebooks
        let zeroth_tensor = Tensor::new(zeroth_tokens.as_slice(), &self.device)
            .map_err(|e| TtsError::inference(format!("tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| TtsError::inference(format!("unsqueeze failed: {e}")))?;

        info!(
            "Running CodePredictor: hidden={:?}, zeroth={}",
            hidden_states.dims(),
            zeroth_tokens.len()
        );

        let all_codes = cp
            .predict_from_hidden(&hidden_states, &zeroth_tensor, sampling_config)
            .map_err(|e| TtsError::inference(format!("code prediction failed: {e}")))?;

        // Convert from [1, seq_len, 16] to [16][seq_len]
        // all_codes is [batch, seq_len, num_codebooks]
        let all_codes_2d: Vec<Vec<u32>> = all_codes
            .squeeze(0)
            .map_err(|e| TtsError::inference(format!("squeeze failed: {e}")))?
            .to_vec2()
            .map_err(|e| TtsError::inference(format!("to_vec2 failed: {e}")))?;

        // Transpose: from [seq_len][16] to [16][seq_len]
        let num_codebooks = 16;
        let mut result: Vec<Vec<u32>> = (0..num_codebooks)
            .map(|_| Vec::with_capacity(seq_len))
            .collect();

        for frame in all_codes_2d.iter() {
            for (cb_idx, &token) in frame.iter().enumerate() {
                if cb_idx < num_codebooks {
                    // Filter special tokens (>= 2048) for each codebook
                    let filtered_token = if token < 2048 { token } else { 0 };
                    result[cb_idx].push(filtered_token);
                }
            }
        }

        info!(
            "Multi-codebook generation complete: {} codebooks x {} tokens",
            result.len(),
            result[0].len()
        );

        Ok(result)
    }

    /// Generate acoustic tokens from text tokens with optional speaker.
    ///
    /// Uses real acoustic model if available, otherwise falls back to mock.
    /// If speaker is provided, uses CustomVoice prompt format.
    pub fn generate_acoustic_with_options(
        &self,
        text_tokens: &[u32],
        speaker: Option<&str>,
        lang: Lang,
        max_tokens: usize,
    ) -> TtsResult<Vec<u32>> {
        if text_tokens.is_empty() {
            return Err(TtsError::invalid_input("empty text tokens"));
        }

        match &self.acoustic {
            AcousticBackend::Neural {
                model,
                code_predictor,
            } => {
                // Use CustomVoice format if speaker is provided or if we have speaker configured
                let use_speaker = speaker.is_some() || self.config.default_speaker.is_some();
                let actual_speaker = speaker.or(self.config.default_speaker.as_deref());

                if use_speaker {
                    self.generate_acoustic_with_speaker(
                        model,
                        code_predictor.as_deref(),
                        text_tokens,
                        actual_speaker,
                        lang,
                        max_tokens,
                    )
                } else {
                    self.generate_acoustic_neural(
                        model,
                        code_predictor.as_deref(),
                        text_tokens,
                        max_tokens,
                    )
                }
            }
            AcousticBackend::Mock => self.generate_acoustic_mock(text_tokens, max_tokens),
        }
    }

    /// Generate mock acoustic tokens (for testing without model weights).
    fn generate_acoustic_mock(
        &self,
        text_tokens: &[u32],
        max_tokens: usize,
    ) -> TtsResult<Vec<u32>> {
        // Mock: generate acoustic tokens based on input
        // Tokens must be < 2048 (codebook size for Qwen3-TTS)
        const CODEBOOK_SIZE: u32 = 2048;

        let num_tokens = (text_tokens.len() * 2).min(max_tokens);
        let acoustic_tokens: Vec<u32> = text_tokens
            .iter()
            .cycle()
            .take(num_tokens)
            .map(|&t| (t * 7 + 13) % CODEBOOK_SIZE)
            .collect();

        debug!(
            input_tokens = text_tokens.len(),
            output_tokens = acoustic_tokens.len(),
            "Generated acoustic tokens (mock)"
        );

        Ok(acoustic_tokens)
    }

    /// Decode acoustic tokens to audio.
    pub fn decode_audio(&self, tokens: &[u32]) -> TtsResult<AudioChunk> {
        self.codec.decode(tokens)
    }

    /// Full synthesis: text → audio.
    #[instrument(skip(self), fields(text_len = text.len()))]
    pub fn synthesize(&self, text: &str, lang: Option<Lang>) -> TtsResult<AudioChunk> {
        self.synthesize_with_speaker(text, lang, None)
    }

    /// Full synthesis with speaker: text → audio.
    ///
    /// # Arguments
    /// * `text` - Text to synthesize
    /// * `lang` - Language (optional, defaults to pipeline default)
    /// * `speaker` - Speaker name for CustomVoice models (e.g., "vivian", "ryan")
    #[instrument(skip(self), fields(text_len = text.len(), speaker))]
    pub fn synthesize_with_speaker(
        &self,
        text: &str,
        lang: Option<Lang>,
        speaker: Option<&str>,
    ) -> TtsResult<AudioChunk> {
        let lang = lang.unwrap_or(self.config.default_lang);

        // 1. Normalize
        let normalized = self.normalize(text, Some(lang))?;
        debug!(normalized = %normalized.text, "Text normalized");

        // 2. Tokenize
        let text_tokens = self.tokenize(&normalized)?;
        debug!(num_tokens = text_tokens.len(), "Text tokenized");

        // 3. Generate acoustic tokens with speaker
        let acoustic_tokens = self.generate_acoustic_with_options(
            &text_tokens,
            speaker,
            lang,
            self.config.max_seq_len,
        )?;
        debug!(
            num_acoustic = acoustic_tokens.len(),
            "Acoustic tokens generated"
        );

        // 4. Filter out special tokens before decoding
        // Special tokens (>=2048) cannot be decoded by the audio codec
        let filtered_tokens: Vec<u32> = acoustic_tokens
            .iter()
            .filter(|&&t| t < 2048)
            .copied()
            .collect();

        if filtered_tokens.is_empty() {
            return Err(TtsError::inference(
                "no valid audio tokens generated (all were special tokens)".to_string(),
            ));
        }

        debug!(
            original = acoustic_tokens.len(),
            filtered = filtered_tokens.len(),
            removed = acoustic_tokens.len() - filtered_tokens.len(),
            "Filtered special tokens"
        );

        // 5. Decode to audio
        let audio = self.decode_audio(&filtered_tokens)?;
        debug!(
            samples = audio.num_samples(),
            duration_ms = audio.duration_ms(),
            "Audio decoded"
        );

        Ok(audio)
    }

    /// Full synthesis with multi-codebook decoding (higher quality).
    ///
    /// Uses CodePredictor to generate all 16 codebooks for better audio quality.
    /// Falls back to zeroth codebook only if CodePredictor is not available.
    #[instrument(skip(self), fields(text_len = text.len(), speaker))]
    pub fn synthesize_with_multi_codebook(
        &self,
        text: &str,
        lang: Option<Lang>,
        speaker: Option<&str>,
    ) -> TtsResult<AudioChunk> {
        let lang = lang.unwrap_or(self.config.default_lang);

        // 1. Normalize
        let normalized = self.normalize(text, Some(lang))?;
        debug!(normalized = %normalized.text, "Text normalized");

        // 2. Tokenize
        let text_tokens = self.tokenize(&normalized)?;
        debug!(num_tokens = text_tokens.len(), "Text tokenized");

        // 3. Generate multi-codebook acoustic tokens
        let multi_tokens = match &self.acoustic {
            AcousticBackend::Neural {
                model,
                code_predictor,
            } => {
                let actual_speaker = speaker.or(self.config.default_speaker.as_deref());
                self.generate_acoustic_multi_codebook(
                    model,
                    code_predictor.as_deref(),
                    &text_tokens,
                    actual_speaker,
                    lang,
                    self.config.max_seq_len,
                )?
            }
            AcousticBackend::Mock => {
                // Mock: generate single codebook and pad
                let tokens = self.generate_acoustic_mock(&text_tokens, self.config.max_seq_len)?;
                let seq_len = tokens.len();
                let mut result = Vec::with_capacity(16);
                result.push(tokens);
                for _ in 1..16 {
                    result.push(vec![0u32; seq_len]);
                }
                result
            }
        };

        debug!(
            codebooks = multi_tokens.len(),
            seq_len = multi_tokens.first().map(|v| v.len()).unwrap_or(0),
            "Multi-codebook tokens generated"
        );

        // 4. Decode using multi-codebook decoder
        let audio = self.codec.decode_multi(&multi_tokens)?;
        debug!(
            samples = audio.num_samples(),
            duration_ms = audio.duration_ms(),
            "Audio decoded (multi-codebook)"
        );

        Ok(audio)
    }

    /// Synthesize from a SynthesisRequest.
    pub fn synthesize_request(&self, request: &SynthesisRequest) -> TtsResult<AudioChunk> {
        self.synthesize(&request.text, Some(request.lang))
    }

    /// Synthesize from a SynthesisRequest with speaker.
    pub fn synthesize_request_with_speaker(
        &self,
        request: &SynthesisRequest,
        speaker: Option<&str>,
    ) -> TtsResult<AudioChunk> {
        self.synthesize_with_speaker(&request.text, Some(request.lang), speaker)
    }

    /// Get list of available speakers for CustomVoice models.
    pub fn available_speakers(&self) -> &'static [&'static str] {
        text_tokenizer::SpeakerIds::available_speakers()
    }

    /// Create a streaming session for incremental synthesis.
    pub fn streaming_session(&self) -> TtsResult<StreamingSession<'_>> {
        StreamingSession::new(self)
    }
}

/// A streaming synthesis session for incremental audio generation.
pub struct StreamingSession<'a> {
    pipeline: &'a TtsPipeline,
    decoder: StreamingDecoder,
    text_tokens: Vec<u32>,
    generated_acoustic: Vec<u32>,
    position: usize,
    finished: bool,
}

impl<'a> StreamingSession<'a> {
    /// Create a new streaming session.
    fn new(pipeline: &'a TtsPipeline) -> TtsResult<Self> {
        let decoder = StreamingDecoder::new(Codec12Hz::new_mock(), pipeline.config.crossfade_ms);

        Ok(Self {
            pipeline,
            decoder,
            text_tokens: Vec::new(),
            generated_acoustic: Vec::new(),
            position: 0,
            finished: false,
        })
    }

    /// Set the text to synthesize.
    #[instrument(skip(self), fields(text_len = text.len()))]
    pub fn set_text(&mut self, text: &str, lang: Option<Lang>) -> TtsResult<()> {
        // Normalize and tokenize
        let normalized = self.pipeline.normalize(text, lang)?;
        self.text_tokens = self.pipeline.tokenize(&normalized)?;

        // Pre-generate all acoustic tokens for streaming
        self.generated_acoustic = self
            .pipeline
            .generate_acoustic(&self.text_tokens, self.pipeline.config.max_seq_len)?;

        debug!(
            text_tokens = self.text_tokens.len(),
            acoustic_tokens = self.generated_acoustic.len(),
            "Streaming session initialized"
        );

        self.position = 0;
        self.finished = false;

        Ok(())
    }

    /// Generate the next audio chunk.
    pub fn next_chunk(&mut self) -> TtsResult<Option<AudioChunk>> {
        if self.finished {
            return Ok(None);
        }

        if self.generated_acoustic.is_empty() {
            return Err(TtsError::invalid_input("call set_text() first"));
        }

        let chunk_tokens = self.pipeline.config.chunk_tokens;
        let end_pos = (self.position + chunk_tokens).min(self.generated_acoustic.len());

        if self.position >= end_pos {
            // No more tokens, flush the decoder
            self.finished = true;
            return match self.decoder.flush() {
                Ok(chunk) => Ok(Some(chunk)),
                Err(_) => Ok(None),
            };
        }

        let acoustic_chunk = &self.generated_acoustic[self.position..end_pos];
        self.position = end_pos;

        // Decode to audio with crossfade
        let audio = self.decoder.process(acoustic_chunk)?;

        debug!(
            position = self.position,
            total = self.generated_acoustic.len(),
            samples = audio.num_samples(),
            "Generated streaming chunk"
        );

        // Check if we've processed all tokens
        if self.position >= self.generated_acoustic.len() {
            self.finished = true;
        }

        Ok(Some(audio))
    }

    /// Check if all tokens have been processed.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Get total acoustic tokens generated.
    pub fn total_acoustic_tokens(&self) -> usize {
        self.generated_acoustic.len()
    }

    /// Get current position in token stream.
    pub fn current_position(&self) -> usize {
        self.position
    }

    /// Get total audio samples produced so far.
    pub fn total_samples(&self) -> usize {
        self.decoder.total_samples()
    }

    /// Reset the session for reuse.
    pub fn reset(&mut self) {
        self.decoder.reset();
        self.text_tokens.clear();
        self.generated_acoustic.clear();
        self.position = 0;
        self.finished = false;
    }
}

impl std::fmt::Debug for StreamingSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingSession")
            .field("text_tokens", &self.text_tokens.len())
            .field("generated_acoustic", &self.generated_acoustic.len())
            .field("position", &self.position)
            .field("finished", &self.finished)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.backend, PipelineBackend::Mock);
        assert_eq!(config.default_lang, Lang::Ru);
    }

    #[test]
    fn test_pipeline_mock_creation() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        assert_eq!(pipeline.config().backend, PipelineBackend::Mock);
    }

    #[test]
    fn test_pipeline_normalize() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let normalized = pipeline.normalize("Привет мир", Some(Lang::Ru)).unwrap();
        assert!(!normalized.text.is_empty());
    }

    #[test]
    fn test_pipeline_tokenize() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let normalized = pipeline.normalize("Hello world", Some(Lang::En)).unwrap();
        let tokens = pipeline.tokenize(&normalized).unwrap();
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_pipeline_generate_acoustic() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let text_tokens = vec![1, 2, 3, 4, 5];
        let acoustic = pipeline.generate_acoustic(&text_tokens, 100).unwrap();

        assert!(!acoustic.is_empty());
        assert!(acoustic.len() <= 100);
    }

    #[test]
    fn test_pipeline_full_synthesis() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let audio = pipeline.synthesize("Тест синтеза", Some(Lang::Ru)).unwrap();

        assert!(audio.num_samples() > 0);
        assert_eq!(audio.sample_rate, 24000);
    }

    #[test]
    fn test_pipeline_synthesis_request() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let request = SynthesisRequest::new("Test synthesis").with_lang(Lang::En);

        let audio = pipeline.synthesize_request(&request).unwrap();
        assert!(audio.num_samples() > 0);
    }

    #[test]
    fn test_streaming_session_creation() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let session = pipeline.streaming_session();
        assert!(session.is_ok());
    }

    #[test]
    fn test_streaming_session_set_text() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let mut session = pipeline.streaming_session().unwrap();

        let result = session.set_text("Привет", Some(Lang::Ru));
        assert!(result.is_ok());
        assert!(session.total_acoustic_tokens() > 0);
    }

    #[test]
    fn test_streaming_session_next_chunk() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let mut session = pipeline.streaming_session().unwrap();

        session.set_text("Привет мир", Some(Lang::Ru)).unwrap();

        // Get first chunk
        let chunk = session.next_chunk().unwrap();
        assert!(chunk.is_some());

        let audio = chunk.unwrap();
        assert!(audio.num_samples() > 0);
    }

    #[test]
    fn test_streaming_session_full_iteration() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let mut session = pipeline.streaming_session().unwrap();

        session.set_text("Test", Some(Lang::En)).unwrap();

        let mut chunks = Vec::new();
        while let Some(chunk) = session.next_chunk().unwrap() {
            chunks.push(chunk);
            if session.is_finished() {
                break;
            }
        }

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_streaming_session_reset() {
        let pipeline = TtsPipeline::new_mock().unwrap();
        let mut session = pipeline.streaming_session().unwrap();

        session.set_text("Test", Some(Lang::En)).unwrap();
        session.next_chunk().unwrap();

        session.reset();
        assert_eq!(session.total_acoustic_tokens(), 0);
        assert_eq!(session.total_samples(), 0);
        assert!(!session.is_finished());
    }
}
