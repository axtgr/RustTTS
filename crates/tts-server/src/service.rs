//! TTS gRPC service implementation.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use futures::Stream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::{debug, info, instrument, warn};

use runtime::TtsPipeline;
use tts_core::Lang;

use crate::proto::{
    AudioChunk, AudioFormat, HealthRequest, HealthResponse, InfoRequest, InfoResponse, Language,
    SynthesizeRequest, SynthesizeResponse, tts_service_server::TtsService,
};

/// TTS service implementation.
pub struct TtsServiceImpl {
    pipeline: Arc<TtsPipeline>,
    start_time: Instant,
    version: String,
}

impl TtsServiceImpl {
    /// Create a new TTS service.
    pub fn new(pipeline: Arc<TtsPipeline>) -> Self {
        Self {
            pipeline,
            start_time: Instant::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Convert proto Language to tts_core Lang.
    fn proto_lang_to_lang(lang: Language) -> Lang {
        match lang {
            Language::Ru => Lang::Ru,
            Language::En => Lang::En,
            Language::Mixed => Lang::Mixed,
            Language::Unspecified => Lang::Ru, // Default to Russian
        }
    }

    /// Convert audio samples to bytes (PCM S16LE).
    fn samples_to_bytes(samples: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for &sample in samples {
            let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            bytes.extend_from_slice(&sample_i16.to_le_bytes());
        }
        bytes
    }
}

#[tonic::async_trait]
impl TtsService for TtsServiceImpl {
    type SynthesizeStreamStream =
        Pin<Box<dyn Stream<Item = Result<AudioChunk, Status>> + Send + 'static>>;

    #[instrument(skip(self, request), fields(text_len))]
    async fn synthesize(
        &self,
        request: Request<SynthesizeRequest>,
    ) -> Result<Response<SynthesizeResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        tracing::Span::current().record("text_len", req.text.len());

        if req.text.is_empty() {
            return Err(Status::invalid_argument("text cannot be empty"));
        }

        if req.text.len() > 10000 {
            return Err(Status::invalid_argument("text too long (max 10000 chars)"));
        }

        let lang =
            Self::proto_lang_to_lang(Language::try_from(req.language).unwrap_or(Language::Ru));

        info!(
            text_len = req.text.len(),
            lang = %lang,
            request_id = %req.request_id,
            "Processing synthesis request"
        );

        // Synthesize
        let audio = self
            .pipeline
            .synthesize(&req.text, Some(lang))
            .map_err(|e| Status::internal(format!("synthesis failed: {e}")))?;

        let processing_time_ms = start.elapsed().as_secs_f32() * 1000.0;
        let duration_ms = audio.duration_ms();
        let rtf = if duration_ms > 0.0 {
            processing_time_ms / duration_ms
        } else {
            0.0
        };

        // Convert to bytes
        let audio_data = Self::samples_to_bytes(&audio.pcm);

        debug!(
            samples = audio.num_samples(),
            duration_ms = duration_ms,
            processing_ms = processing_time_ms,
            rtf = rtf,
            "Synthesis completed"
        );

        Ok(Response::new(SynthesizeResponse {
            audio_data,
            audio_format: AudioFormat::PcmS16le as i32,
            sample_rate: audio.sample_rate,
            num_samples: audio.num_samples() as u64,
            duration_ms,
            processing_time_ms,
            rtf,
            request_id: req.request_id,
        }))
    }

    #[instrument(skip(self, request), fields(text_len))]
    async fn synthesize_stream(
        &self,
        request: Request<SynthesizeRequest>,
    ) -> Result<Response<Self::SynthesizeStreamStream>, Status> {
        let req = request.into_inner();

        tracing::Span::current().record("text_len", req.text.len());

        if req.text.is_empty() {
            return Err(Status::invalid_argument("text cannot be empty"));
        }

        if req.text.len() > 10000 {
            return Err(Status::invalid_argument("text too long (max 10000 chars)"));
        }

        let lang =
            Self::proto_lang_to_lang(Language::try_from(req.language).unwrap_or(Language::Ru));
        let request_id = req.request_id.clone();

        info!(
            text_len = req.text.len(),
            lang = %lang,
            request_id = %request_id,
            "Starting streaming synthesis"
        );

        let pipeline = Arc::clone(&self.pipeline);
        let (tx, rx) = mpsc::channel(32);

        // Spawn streaming task
        tokio::spawn(async move {
            let mut session = match pipeline.streaming_session() {
                Ok(s) => s,
                Err(e) => {
                    warn!("Failed to create streaming session: {e}");
                    let _ = tx
                        .send(Err(Status::internal(format!(
                            "failed to create session: {e}"
                        ))))
                        .await;
                    return;
                }
            };

            if let Err(e) = session.set_text(&req.text, Some(lang)) {
                warn!("Failed to set text: {e}");
                let _ = tx
                    .send(Err(Status::internal(format!("failed to set text: {e}"))))
                    .await;
                return;
            }

            let mut chunk_index = 0u32;
            let mut start_ms = 0.0f32;

            loop {
                match session.next_chunk() {
                    Ok(Some(audio_chunk)) => {
                        let audio_data = Self::samples_to_bytes(&audio_chunk.pcm);
                        let duration_ms = audio_chunk.duration_ms();
                        let is_final = session.is_finished();

                        let chunk = AudioChunk {
                            chunk_index,
                            audio_data,
                            sample_rate: audio_chunk.sample_rate,
                            num_samples: audio_chunk.num_samples() as u32,
                            start_ms,
                            duration_ms,
                            is_final,
                            request_id: request_id.clone(),
                        };

                        if tx.send(Ok(chunk)).await.is_err() {
                            debug!("Client disconnected");
                            break;
                        }

                        chunk_index += 1;
                        start_ms += duration_ms;

                        if is_final {
                            break;
                        }
                    }
                    Ok(None) => {
                        debug!("Stream finished");
                        break;
                    }
                    Err(e) => {
                        warn!("Chunk generation failed: {e}");
                        let _ = tx
                            .send(Err(Status::internal(format!("chunk failed: {e}"))))
                            .await;
                        break;
                    }
                }
            }

            debug!(
                chunks = chunk_index,
                request_id = %request_id,
                "Streaming synthesis completed"
            );
        });

        let stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let uptime = self.start_time.elapsed().as_secs();

        Ok(Response::new(HealthResponse {
            status: crate::proto::health_response::Status::Serving as i32,
            version: self.version.clone(),
            uptime_seconds: uptime,
        }))
    }

    async fn get_info(
        &self,
        _request: Request<InfoRequest>,
    ) -> Result<Response<InfoResponse>, Status> {
        Ok(Response::new(InfoResponse {
            name: "Qwen3-TTS Rust Server".to_string(),
            version: self.version.clone(),
            model_name: "qwen3-tts-mock".to_string(),
            supported_languages: vec![
                Language::Ru as i32,
                Language::En as i32,
                Language::Mixed as i32,
            ],
            default_sample_rate: 24000,
            max_text_length: 10000,
            num_speakers: 1,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_samples_to_bytes() {
        let samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let bytes = TtsServiceImpl::samples_to_bytes(&samples);

        assert_eq!(bytes.len(), samples.len() * 2);
    }

    #[test]
    fn test_proto_lang_conversion() {
        assert!(matches!(
            TtsServiceImpl::proto_lang_to_lang(Language::Ru),
            Lang::Ru
        ));
        assert!(matches!(
            TtsServiceImpl::proto_lang_to_lang(Language::En),
            Lang::En
        ));
        assert!(matches!(
            TtsServiceImpl::proto_lang_to_lang(Language::Unspecified),
            Lang::Ru
        ));
    }
}
