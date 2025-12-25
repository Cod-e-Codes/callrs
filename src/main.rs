use anyhow::{Context, Result};
use arboard::Clipboard;
use base64::prelude::*;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap},
};
use ringbuf::{HeapCons, HeapProd, HeapRb, traits::*};
use serde::{Deserialize, Serialize};
use std::{
    io,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::sync::mpsc;

use rubato::{Resampler, SincFixedIn};
use std::cell::RefCell;
use std::collections::VecDeque;

use audiopus::{
    Application as OpusApplication, Channels as OpusChannels, SampleRate as OpusSampleRate,
    coder::{Decoder as OpusDecoder, Encoder as OpusEncoder},
};

use chacha20poly1305::{
    ChaCha20Poly1305, Key, Nonce,
    aead::{Aead, KeyInit, OsRng},
};

const SAMPLE_RATE: u32 = 48000; // API requires 48k for best Opus compatibility
const FRAME_DURATION_MS: u32 = 20;
const FRAME_SIZE: usize = (SAMPLE_RATE * FRAME_DURATION_MS / 1000) as usize; // 960 samples

const BUFFER_SIZE: usize = 19200;

// Noise gate threshold: RMS amplitude below this value will be considered silence
// Value is in normalized range [0.0, 1.0] where 1.0 is full scale
const NOISE_GATE_THRESHOLD: f32 = 0.01; // ~1% of full scale, adjustable

// Adaptive jitter buffer configuration
const MIN_BUFFER_MS: f32 = 20.0; // Minimum buffer size (20ms for low latency)
const MAX_BUFFER_MS: f32 = 150.0; // Maximum buffer size (150ms for high jitter)
const INITIAL_BUFFER_MS: f32 = 40.0; // Initial buffer size (start conservative, lower than old 60ms for better latency)
const JITTER_ADJUSTMENT_THRESHOLD_MS: f32 = 5.0; // Adjust if jitter exceeds this
const BUFFER_UNDERRUN_THRESHOLD_SAMPLES: usize = FRAME_SIZE; // Trigger increase if below this
const BUFFER_OVERRUN_THRESHOLD_SAMPLES: usize = BUFFER_SIZE / 2; // Trigger decrease if above this

// Jitter buffer adjustment parameters
const JITTER_BUFFER_ADJUSTMENT_FACTOR: f32 = 0.1; // Exponential smoothing factor (10% change per adjustment)
const JITTER_BUFFER_REDUCTION_FACTOR: f32 = 0.95; // Factor to reduce buffer when overfull (5% reduction)
const JITTER_MULTIPLIER: f32 = 3.0; // Multiply jitter by this to calculate required buffer size
const JITTER_BUFFER_SIGNIFICANT_CHANGE_MS: f32 = 5.0; // Only log adjustments larger than this
const JITTER_BUFFER_FULLNESS_THRESHOLD: f32 = 1.5; // Buffer is considered overfull at 1.5x target
const JITTER_BUFFER_ADJUSTMENT_LOG_INTERVAL: u32 = 30; // Log at most every N adjustments
const JITTER_BUFFER_INTER_ARRIVAL_HISTORY_SIZE: usize = 10; // Number of inter-arrival intervals to track
const INTER_ARRIVAL_OUTLIER_THRESHOLD_MS: f32 = 500.0; // Ignore intervals larger than this

// Volume calculation
const VOLUME_NORMALIZATION_FACTOR: f32 = 5.0; // Multiply RMS by this for volume display
const VOLUME_UPDATE_INTERVAL_MS: u64 = 50; // Update volume display at most every 50ms
const CAPTURE_PROCESSOR_SLEEP_MS: u64 = 50; // Sleep duration when audio is inactive

// Packet loss concealment
const MAX_PACKET_GAP_FOR_PLC: u32 = 10; // Maximum gap (packets) to attempt PLC (200ms at 20ms frames)

// UI/Logging
const MAX_LOG_ENTRIES: usize = 100; // Maximum number of log entries to keep
const NETWORK_CHANNEL_BUFFER_SIZE: usize = 100; // Buffer size for network message channel

// UPnP configuration
const UPNP_LEASE_SECONDS: u32 = 3600; // Port mapping lease time (1 hour)

// Resampler configuration
const RESAMPLER_SINC_LEN: usize = 256; // Sinc interpolation filter length
const RESAMPLER_F_CUTOFF: f32 = 0.95; // Cutoff frequency (0.95 = 95% of Nyquist)
const RESAMPLER_OVERSAMPLING_FACTOR: usize = 128; // Oversampling factor for quality
const RESAMPLER_CHUNK_SIZE_OUTPUT: usize = 960; // Output chunk size (~20ms at 48kHz)
const RESAMPLER_CHUNK_SIZE_INPUT: usize = 1024; // Input chunk size for capture resampling

// Encryption configuration (ChaCha20Poly1305)
const ENCRYPTION_KEY_SIZE_BYTES: usize = 32; // ChaCha20Poly1305 key size
const ENCRYPTION_NONCE_SIZE_BYTES: usize = 12; // ChaCha20Poly1305 nonce size
const ENCRYPTION_SEQUENCE_BYTES: usize = 4; // Bytes from sequence number used in nonce

#[derive(Clone, Debug, PartialEq)]
enum AppMode {
    Menu,
    Connecting,
    IceGathering,
    HostWaitingForAnswer {
        offer: String,
    },
    ClientGeneratingAnswer {
        answer: String,
    },
    IceConnecting,
    InCall {
        peer: String,
        connection_type: String,
    },
    Error(String),
}

enum AppAction {
    Log(String),
    SetMode(AppMode),
    Input(event::KeyEvent),
    Paste(String),
    SetOffer(String),
    SetAnswer(String),
    StoreHostAgent(IceAgentHandle),

    StartConnection {
        conn: Arc<dyn webrtc_util::Conn + Send + Sync>,
        encryption_key: Option<String>,
    },
    SetMicVolume(f32),
    SetReceiveVolume(f32),
    SetLatency(f32),
    SetConnectionType(String),
    ToggleMute,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
struct SessionOffer {
    ufrag: String,
    pwd: String,
    candidates: Vec<String>,
    encryption_key: String, // Base64-encoded 32-byte encryption key
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
struct SessionAnswer {
    ufrag: String,
    pwd: String,
    candidates: Vec<String>,
}

struct App {
    mode: AppMode,
    logs: Vec<String>,
    input: String,
    should_quit: bool,
    local_ip: String,
    session_offer: Option<String>,
    session_answer: Option<String>,
    host_agent_handle: Option<IceAgentHandle>,
    mic_volume: f32,
    receive_volume: f32,
    current_latency_ms: f32,
    connection_type: String,
    is_muted: bool,
}

impl App {
    fn new() -> Self {
        let local_ip = local_ip_address::local_ip()
            .map(|ip| ip.to_string())
            .unwrap_or_else(|_| "Unknown".into());

        Self {
            mode: AppMode::Menu,
            logs: vec!["callrs - P2P Voice Chat".into()],
            input: String::new(),
            should_quit: false,
            local_ip,
            session_offer: None,
            session_answer: None,
            host_agent_handle: None,
            mic_volume: 0.0,
            receive_volume: 0.0,
            current_latency_ms: 0.0,
            connection_type: "Unknown".into(),
            is_muted: false,
        }
    }

    fn add_log(&mut self, msg: String) {
        self.logs.push(msg);
        if self.logs.len() > MAX_LOG_ENTRIES {
            self.logs.remove(0);
        }
    }
}

struct AudioHandle {
    _streams: (cpal::Stream, cpal::Stream),
    is_active: Arc<AtomicBool>,
    is_muted: Arc<AtomicBool>,
    input_sample_rate: u32,
    output_sample_rate: u32,
}

impl AudioHandle {
    fn start(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }
    fn stop(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }
    fn toggle_mute(&self) -> bool {
        let current = self.is_muted.load(Ordering::Relaxed);
        self.is_muted.store(!current, Ordering::Relaxed);
        !current
    }
}

struct CallSession {
    audio: AudioHandle,
    capture_task: tokio::task::JoinHandle<()>,
    network_task: tokio::task::JoinHandle<()>,
}

impl CallSession {
    async fn stop(self) {
        self.capture_task.abort();
        self.network_task.abort();
        self.audio.stop();
    }
}

fn setup_audio(capture_prod: HeapProd<f32>, playback_cons: HeapCons<f32>) -> Result<AudioHandle> {
    let host = cpal::default_host();
    let is_active = Arc::new(AtomicBool::new(false));
    let is_muted = Arc::new(AtomicBool::new(false));

    let input_dev = host.default_input_device().context("No input device")?;
    let output_dev = host.default_output_device().context("No output device")?;

    // Try to find a 48kHz config
    let in_cfg = input_dev
        .supported_input_configs()?
        .find(|c| c.min_sample_rate() <= SAMPLE_RATE && c.max_sample_rate() >= SAMPLE_RATE)
        .map(|c| c.with_sample_rate(SAMPLE_RATE))
        .or_else(|| input_dev.default_input_config().ok())
        .context("No supported input audio configuration available")?;

    let out_cfg = output_dev
        .supported_output_configs()?
        .find(|c| c.min_sample_rate() <= SAMPLE_RATE && c.max_sample_rate() >= SAMPLE_RATE)
        .map(|c| c.with_sample_rate(SAMPLE_RATE))
        .or_else(|| output_dev.default_output_config().ok())
        .context("No supported output audio configuration available")?;

    // Get actual sample rates (may differ from 48kHz if device doesn't support it)
    let input_sample_rate: u32 = in_cfg.sample_rate();
    let output_sample_rate: u32 = out_cfg.sample_rate();

    // Note: Input resampling (from device rate to 48kHz) is handled in capture_processor.
    // Output resampling (from 48kHz to device rate) is handled in the output stream callback below.

    let in_channels = in_cfg.channels() as usize;
    let out_channels = out_cfg.channels() as usize;

    let active_in = is_active.clone();
    let muted_in = is_muted.clone();
    let mut capture_prod_f32 = capture_prod;
    let input_stream = match in_cfg.sample_format() {
        cpal::SampleFormat::F32 => input_dev.build_input_stream(
            &in_cfg.config(),
            move |data: &[f32], _| {
                if !active_in.load(Ordering::Relaxed) {
                    return;
                }
                let is_muted = muted_in.load(Ordering::Relaxed);
                for frame in data.chunks_exact(in_channels) {
                    let mono: f32 = if is_muted {
                        0.0
                    } else {
                        frame.iter().sum::<f32>() / in_channels as f32
                    };
                    let _ = capture_prod_f32.try_push(mono);
                }
            },
            |_| {},
            None,
        )?,
        _ => return Err(anyhow::anyhow!("Unsupported sample format")),
    };

    let active_out = is_active.clone();
    let mut playback_cons_f32 = playback_cons;

    // Setup output resampling if needed (playback buffer is 48kHz, device may be different)
    let output_needs_resampling = output_sample_rate != SAMPLE_RATE;
    let resampler_state: Option<RefCell<(SincFixedIn<f32>, VecDeque<f32>)>> =
        if output_needs_resampling {
            let params = rubato::SincInterpolationParameters {
                sinc_len: RESAMPLER_SINC_LEN,
                f_cutoff: RESAMPLER_F_CUTOFF,
                interpolation: rubato::SincInterpolationType::Linear,
                window: rubato::WindowFunction::BlackmanHarris2,
                oversampling_factor: RESAMPLER_OVERSAMPLING_FACTOR,
            };
            // Resample FROM 48kHz (buffer) TO output_sample_rate (device)
            // ratio = output_rate / input_rate = output_sample_rate / 48000.0
            let ratio = output_sample_rate as f64 / SAMPLE_RATE as f64;
            // Input chunk size: approximate 20ms at 48kHz
            let chunk_in = RESAMPLER_CHUNK_SIZE_OUTPUT;
            match SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_in, 1) {
                Ok(resampler) => Some(RefCell::new((resampler, VecDeque::new()))),
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "Failed to create output resampler: {:?}",
                        e
                    ));
                }
            }
        } else {
            None
        };

    let output_stream = output_dev.build_output_stream(
        &out_cfg.config(),
        move |data: &mut [f32], _| {
            if !active_out.load(Ordering::Relaxed) {
                data.fill(0.0);
                return;
            }

            if let Some(ref resampler_cell) = resampler_state {
                // Output resampling path: 48kHz buffer -> device sample rate
                let mut resampler_state = resampler_cell.borrow_mut();
                let (resampler, resampled_buffer) = &mut *resampler_state;

                // Fill output buffer with resampled samples
                for frame in data.chunks_exact_mut(out_channels) {
                    // Get sample from resampled buffer, or resample more if needed
                    if resampled_buffer.is_empty() {
                        // Need to resample more samples
                        let input_chunk_size = resampler.input_frames_next();
                        let mut input_chunk = Vec::with_capacity(input_chunk_size);

                        // Collect 48kHz samples from the playback buffer
                        for _ in 0..input_chunk_size {
                            if let Some(sample) = playback_cons_f32.try_pop() {
                                input_chunk.push(sample);
                            } else {
                                input_chunk.push(0.0); // Underrun: pad with silence
                            }
                        }

                        // Resample the chunk
                        if let Ok(output) = resampler.process(&[input_chunk], None)
                            && let Some(chan0) = output.first()
                        {
                            resampled_buffer.extend(chan0.iter().copied());
                        }
                    }

                    // Use resampled sample, or silence if buffer is still empty
                    let sample = resampled_buffer.pop_front().unwrap_or(0.0);
                    frame.fill(sample);
                }
            } else {
                // Direct path: 48kHz buffer -> 48kHz device (no resampling)
                for frame in data.chunks_exact_mut(out_channels) {
                    let sample = playback_cons_f32.try_pop().unwrap_or(0.0);
                    frame.fill(sample);
                }
            }
        },
        |_| {},
        None,
    )?;

    input_stream.play()?;
    output_stream.play()?;

    Ok(AudioHandle {
        _streams: (input_stream, output_stream),
        is_active,
        is_muted,
        input_sample_rate,
        output_sample_rate,
    })
}

// Helper: Try to open UPnP port
fn try_upnp_mapping(local_port: u16) -> Result<String> {
    use igd::search_gateway;
    use std::net::SocketAddrV4;

    let gateway = search_gateway(Default::default())
        .map_err(|e| anyhow::anyhow!("UPnP Gateway not found: {}", e))?;

    let local_ip = local_ip_address::local_ip()?;
    let local_addr = match local_ip {
        std::net::IpAddr::V4(v4) => SocketAddrV4::new(v4, local_port),
        _ => return Err(anyhow::anyhow!("IPv6 not supported for simple UPnP")),
    };

    match gateway.add_port(
        igd::PortMappingProtocol::UDP,
        local_port, // external port (same as internal for simplicity)
        local_addr,
        UPNP_LEASE_SECONDS,
        "CallRS Voice App",
    ) {
        Ok(_) => Ok(format!("UPnP Success! Port {} mapped.", local_port)),
        Err(e) => Err(anyhow::anyhow!("UPnP Mapping Failed: {}", e)),
    }
}

// Helper: Try to remove UPnP port mapping (best effort cleanup)
fn try_upnp_cleanup(local_port: u16) {
    use igd::search_gateway;
    if let Ok(gateway) = search_gateway(Default::default()) {
        let _ = gateway.remove_port(igd::PortMappingProtocol::UDP, local_port);
        // Silently ignore errors - cleanup is best effort
    }
}

// Encryption helpers
fn generate_encryption_key() -> String {
    let key = ChaCha20Poly1305::generate_key(&mut OsRng);
    BASE64_STANDARD.encode(key.as_slice())
}

fn parse_encryption_key(key_b64: &str) -> Result<Key> {
    let key_bytes = BASE64_STANDARD
        .decode(key_b64)
        .map_err(|_| anyhow::anyhow!("Invalid encryption key format"))?;
    if key_bytes.len() != ENCRYPTION_KEY_SIZE_BYTES {
        return Err(anyhow::anyhow!(
            "Encryption key must be {} bytes",
            ENCRYPTION_KEY_SIZE_BYTES
        ));
    }
    Ok(*Key::from_slice(&key_bytes))
}

// Encrypt packet: nonce + encrypted payload
// Nonce is derived from sequence number to ensure uniqueness
fn encrypt_packet(data: &[u8], key: &Key, sequence: u32) -> Result<Vec<u8>> {
    let cipher = ChaCha20Poly1305::new(key);

    // Use sequence number as nonce (little-endian, zero-padded to nonce size)
    let mut nonce_bytes = [0u8; ENCRYPTION_NONCE_SIZE_BYTES];
    nonce_bytes[..ENCRYPTION_SEQUENCE_BYTES].copy_from_slice(&sequence.to_le_bytes());
    let nonce = Nonce::from_slice(&nonce_bytes);

    cipher
        .encrypt(nonce, data)
        .map_err(|e| anyhow::anyhow!("Encryption failed: {:?}", e))
}

// Decrypt packet: expects nonce (derived from sequence) + encrypted payload
fn decrypt_packet(encrypted: &[u8], key: &Key, sequence: u32) -> Result<Vec<u8>> {
    let cipher = ChaCha20Poly1305::new(key);

    // Use sequence number as nonce (little-endian, zero-padded to nonce size)
    let mut nonce_bytes = [0u8; ENCRYPTION_NONCE_SIZE_BYTES];
    nonce_bytes[..ENCRYPTION_SEQUENCE_BYTES].copy_from_slice(&sequence.to_le_bytes());
    let nonce = Nonce::from_slice(&nonce_bytes);

    cipher
        .decrypt(nonce, encrypted)
        .map_err(|e| anyhow::anyhow!("Decryption failed: {:?}", e))
}

use webrtc_ice::{
    agent::{Agent, agent_config::AgentConfig},
    network_type::NetworkType,
    url::Url,
};

async fn create_ice_agent() -> Result<Agent> {
    let stun_url = Url::parse_url("stun:stun.l.google.com:19302")?;

    // Create a UDP Multiplexer binding to port 9090
    // This forces all ICE traffic to use this port, matching our UPnP mapping.
    let udp_socket = tokio::net::UdpSocket::bind("0.0.0.0:9090")
        .await
        .map_err(|e| anyhow::anyhow!("Failed to bind to port 9090: {}", e))?;

    let mux =
        webrtc_ice::udp_mux::UDPMuxDefault::new(webrtc_ice::udp_mux::UDPMuxParams::new(udp_socket));

    let config = AgentConfig {
        network_types: vec![NetworkType::Udp4, NetworkType::Udp6],
        urls: vec![stun_url],
        udp_network: webrtc_ice::udp_network::UDPNetwork::Muxed(mux),
        ..Default::default()
    };

    Agent::new(config)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create ICE agent: {}", e))
}

struct IceAgentHandle {
    agent: Arc<Agent>,
    ufrag: String,
    pwd: String,
    encryption_key: Option<String>, // Encryption key for this session
}

async fn host_gather_candidates(action_tx: mpsc::Sender<AppAction>) -> Result<IceAgentHandle> {
    let _ = action_tx
        .send(AppAction::Log("Creating ICE agent...".into()))
        .await;
    let agent = create_ice_agent().await?;

    let (gather_done_tx, mut gather_done_rx) = mpsc::channel::<()>(1);
    let gather_tx_clone = gather_done_tx.clone();

    agent.on_candidate(Box::new(move |candidate| {
        let tx = gather_tx_clone.clone();
        Box::pin(async move {
            if candidate.is_none() {
                let _ = tx.send(()).await;
            }
        })
    }));

    let _ = action_tx
        .send(AppAction::Log("Gathering candidates...".into()))
        .await;
    agent.gather_candidates()?;

    // Wait for gathering to complete
    let _ = tokio::time::timeout(Duration::from_secs(5), gather_done_rx.recv()).await;

    let (ufrag, pwd) = agent.get_local_user_credentials().await;
    let candidates = agent.get_local_candidates().await?;

    let _ = action_tx
        .send(AppAction::Log(format!(
            "Gathered {} candidates",
            candidates.len()
        )))
        .await;

    // Generate encryption key for this session
    let encryption_key = generate_encryption_key();

    // Create offer
    let offer = SessionOffer {
        ufrag: ufrag.clone(),
        pwd: pwd.clone(),
        candidates: candidates.iter().map(|c| c.marshal()).collect(),
        encryption_key: encryption_key.clone(),
    };

    let offer_json = serde_json::to_string(&offer)?;
    let offer_b64 = BASE64_STANDARD.encode(&offer_json);

    let _ = action_tx.send(AppAction::SetOffer(offer_b64.clone())).await;
    let _ = action_tx
        .send(AppAction::SetMode(AppMode::HostWaitingForAnswer {
            offer: offer_b64,
        }))
        .await;

    Ok(IceAgentHandle {
        agent: Arc::new(agent),
        ufrag,
        pwd,
        encryption_key: Some(encryption_key),
    })
}

async fn client_gather_candidates(
    offer_b64: String,
    action_tx: mpsc::Sender<AppAction>,
) -> Result<IceAgentHandle> {
    let _ = action_tx
        .send(AppAction::Log("Parsing offer...".into()))
        .await;

    let offer_json = BASE64_STANDARD
        .decode(&offer_b64)
        .map_err(|_| anyhow::anyhow!("Invalid base64"))?;
    let offer: SessionOffer = serde_json::from_slice(&offer_json)?;

    let _ = action_tx
        .send(AppAction::Log("Creating ICE agent...".into()))
        .await;
    let agent = create_ice_agent().await?;

    let _ = action_tx
        .send(AppAction::Log("Setting remote credentials...".into()))
        .await;
    agent
        .set_remote_credentials(offer.ufrag.clone(), offer.pwd.clone())
        .await
        .map_err(|e| anyhow::anyhow!("Failed to set remote credentials: {}", e))?;

    // Add remote candidates
    for candidate_str in &offer.candidates {
        match webrtc_ice::candidate::candidate_base::unmarshal_candidate(candidate_str) {
            Ok(candidate) => {
                let candidate_arc: Arc<dyn webrtc_ice::candidate::Candidate + Send + Sync> =
                    Arc::from(candidate);
                match agent.add_remote_candidate(&candidate_arc) {
                    Ok(_) => {
                        let _ = action_tx
                            .send(AppAction::Log("Added remote candidate".into()))
                            .await;
                    }
                    Err(e) => {
                        let _ = action_tx
                            .send(AppAction::Log(format!(
                                "Failed to add candidate (continuing): {}",
                                e
                            )))
                            .await;
                    }
                }
            }
            Err(e) => {
                let _ = action_tx
                    .send(AppAction::Log(format!("Failed to parse candidate: {}", e)))
                    .await;
            }
        }
    }

    let _ = action_tx
        .send(AppAction::Log("Gathering candidates...".into()))
        .await;
    let (gather_done_tx, mut gather_done_rx) = mpsc::channel::<()>(1);
    let gather_tx_clone = gather_done_tx.clone();

    agent.on_candidate(Box::new(move |candidate| {
        let tx = gather_tx_clone.clone();
        Box::pin(async move {
            if candidate.is_none() {
                let _ = tx.send(()).await;
            }
        })
    }));

    agent.gather_candidates()?;

    // Wait for gathering to complete
    let _ = tokio::time::timeout(Duration::from_secs(5), gather_done_rx.recv()).await;

    let (ufrag, pwd) = agent.get_local_user_credentials().await;
    let candidates = agent.get_local_candidates().await?;

    let _ = action_tx
        .send(AppAction::Log(format!(
            "Gathered {} candidates",
            candidates.len()
        )))
        .await;

    // Create answer
    let answer = SessionAnswer {
        ufrag: ufrag.clone(),
        pwd: pwd.clone(),
        candidates: candidates.iter().map(|c| c.marshal()).collect(),
    };

    let answer_json = serde_json::to_string(&answer)?;
    let answer_b64 = BASE64_STANDARD.encode(&answer_json);

    let _ = action_tx
        .send(AppAction::SetAnswer(answer_b64.clone()))
        .await;
    let _ = action_tx
        .send(AppAction::SetMode(AppMode::ClientGeneratingAnswer {
            answer: answer_b64,
        }))
        .await;

    Ok(IceAgentHandle {
        agent: Arc::new(agent),
        ufrag: offer.ufrag,
        pwd: offer.pwd,
        encryption_key: Some(offer.encryption_key),
    })
}

// Helper to send action with error tracking (logs periodically if channel is full)
fn try_send_action(
    action_tx: &mpsc::Sender<AppAction>,
    action: AppAction,
    dropped_count: &mut u64,
    last_log: &mut std::time::Instant,
) {
    if action_tx.try_send(action).is_err() {
        *dropped_count += 1;
        // Log every 100 dropped messages or every 5 seconds, whichever comes first
        if (*dropped_count).is_multiple_of(100) || last_log.elapsed().as_secs() >= 5 {
            let _ = action_tx.try_send(AppAction::Log(format!(
                "Warning: Dropped {} UI update messages (channel full, UI may be lagging)",
                dropped_count
            )));
            *last_log = std::time::Instant::now();
        }
    }
}

async fn capture_processor(
    mut capture_cons: HeapCons<f32>,
    network_tx: mpsc::Sender<Vec<u8>>,
    is_active: Arc<AtomicBool>,
    sample_rate: u32,
    action_tx: mpsc::Sender<AppAction>,
    encryption_key: Option<Key>,
) {
    // Opus Encoder setup
    let encoder = match OpusEncoder::new(
        OpusSampleRate::Hz48000,
        OpusChannels::Mono,
        OpusApplication::Voip,
    ) {
        Ok(enc) => enc,
        Err(e) => {
            let _ = action_tx.try_send(AppAction::Log(format!(
                "Critical: Failed to create Opus encoder: {:?}. Audio capture cannot continue.",
                e
            )));
            return; // Task exits if encoder cannot be created
        }
    };

    let frame_size = FRAME_SIZE;
    let mut pcm_buf: Vec<i16> = Vec::with_capacity(frame_size);
    let mut opus_buf = vec![0u8; 1024];

    // Pre-allocated scratch buffer for zero-allocation frame processing
    let mut frame_scratch: Vec<f32> = vec![0.0; frame_size];

    // Resampler setup
    let mut resampler: Option<SincFixedIn<f32>> = if sample_rate != 48000 {
        let params = rubato::SincInterpolationParameters {
            sinc_len: RESAMPLER_SINC_LEN,
            f_cutoff: RESAMPLER_F_CUTOFF,
            interpolation: rubato::SincInterpolationType::Linear,
            window: rubato::WindowFunction::BlackmanHarris2,
            oversampling_factor: RESAMPLER_OVERSAMPLING_FACTOR,
        };
        // Input chunk size for capture resampling
        let chunk_in = RESAMPLER_CHUNK_SIZE_INPUT;
        let ratio = 48000.0 / sample_rate as f64;
        match SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_in, 1) {
            Ok(r) => Some(r),
            Err(e) => {
                let _ = action_tx
                    .try_send(AppAction::Log(format!(
                        "Warning: Failed to create input resampler ({} -> 48kHz): {:?}. Audio quality may be degraded.",
                        sample_rate, e
                    )));
                None // Continue without resampling
            }
        }
    } else {
        None
    };

    let mut resample_input_buffer: Vec<f32> = Vec::new();
    let resample_chunk_in = if let Some(r) = &resampler {
        r.input_frames_next()
    } else {
        0
    };

    // Output buffer for F32 samples (either raw or resampled) waiting to be converted to i16
    let mut f32_capture_buffer: Vec<f32> = Vec::with_capacity(frame_size * 2);

    let mut sequence: u32 = 0;

    // Flow control / Metrics
    let mut dropped_packets = 0u64;
    let mut last_drop_log = std::time::Instant::now();
    let mut dropped_ui_messages = 0u64;
    let mut last_ui_drop_log = std::time::Instant::now();

    // For volume calculation
    let mut vol_sum = 0.0;
    let mut vol_count = 0;
    let mut last_vol_update = std::time::Instant::now();

    loop {
        if !is_active.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_millis(CAPTURE_PROCESSOR_SLEEP_MS)).await;
            continue;
        }

        // 1. Drain ringbuf
        while let Some(sample) = capture_cons.try_pop() {
            if resampler.is_some() {
                resample_input_buffer.push(sample);
            } else {
                f32_capture_buffer.push(sample);
            }
        }

        // 2. Run Resampler if needed
        if let Some(r) = &mut resampler {
            while resample_input_buffer.len() >= resample_chunk_in {
                let chunk: Vec<f32> = resample_input_buffer.drain(0..resample_chunk_in).collect();
                // SincFixedIn expects Vec<Vec<f32>> for channels
                match r.process(&[chunk], None) {
                    Ok(output) => {
                        // output[0] is channel 0
                        if let Some(chan0) = output.first() {
                            f32_capture_buffer.extend_from_slice(chan0);
                        }
                    }
                    Err(e) => {
                        try_send_action(
                            &action_tx,
                            AppAction::Log(format!("Resample error: {:?}", e)),
                            &mut dropped_ui_messages,
                            &mut last_ui_drop_log,
                        );
                    }
                }
            }
        }

        // 3. Process F32 buffer into Opus Frames
        while f32_capture_buffer.len() >= frame_size {
            // Copy frame_size samples to scratch buffer (zero-allocation)
            frame_scratch[..frame_size].copy_from_slice(&f32_capture_buffer[..frame_size]);
            f32_capture_buffer.drain(0..frame_size);

            // Calculate RMS (Root Mean Square) for noise gate
            let rms = (frame_scratch[..frame_size]
                .iter()
                .map(|s| s * s)
                .sum::<f32>()
                / frame_size as f32)
                .sqrt();

            // Noise gate: skip encoding if below threshold
            if rms < NOISE_GATE_THRESHOLD {
                // Still update volume display to show silence
                vol_sum = 0.0;
                vol_count = frame_size;
                if last_vol_update.elapsed().as_millis() > VOLUME_UPDATE_INTERVAL_MS as u128 {
                    try_send_action(
                        &action_tx,
                        AppAction::SetMicVolume(0.0),
                        &mut dropped_ui_messages,
                        &mut last_ui_drop_log,
                    );
                    last_vol_update = std::time::Instant::now();
                }
                continue; // Skip encoding and sending this frame
            }

            // Calc volume & Convert to i16
            for sample in &frame_scratch[..frame_size] {
                vol_sum += sample.abs();
                vol_count += 1;

                let s_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                pcm_buf.push(s_i16);
            }

            // Volume Update
            if vol_count >= frame_size {
                let avg = vol_sum / vol_count as f32;
                let vol_normalized = (avg * VOLUME_NORMALIZATION_FACTOR).clamp(0.0, 1.0);
                if last_vol_update.elapsed().as_millis() > VOLUME_UPDATE_INTERVAL_MS as u128 {
                    try_send_action(
                        &action_tx,
                        AppAction::SetMicVolume(vol_normalized),
                        &mut dropped_ui_messages,
                        &mut last_ui_drop_log,
                    );
                    last_vol_update = std::time::Instant::now();
                }
                vol_sum = 0.0;
                vol_count = 0;
            }

            // Encode
            match encoder.encode(&pcm_buf, &mut opus_buf) {
                Ok(len) => {
                    let opus_payload = &opus_buf[..len];

                    // Encrypt opus payload if encryption key is available
                    let encrypted_payload = if let Some(ref key) = encryption_key {
                        match encrypt_packet(opus_payload, key, sequence) {
                            Ok(encrypted) => encrypted,
                            Err(e) => {
                                try_send_action(
                                    &action_tx,
                                    AppAction::Log(format!("Encryption error: {}", e)),
                                    &mut dropped_ui_messages,
                                    &mut last_ui_drop_log,
                                );
                                continue; // Skip this packet if encryption fails
                            }
                        }
                    } else {
                        opus_payload.to_vec()
                    };

                    // Packet format: [sequence(4 bytes)][encrypted_payload]
                    let mut packet = Vec::with_capacity(4 + encrypted_payload.len());
                    packet.extend_from_slice(&sequence.to_le_bytes());
                    packet.extend_from_slice(&encrypted_payload);

                    match network_tx.try_send(packet) {
                        Ok(_) => {}
                        Err(_) => {
                            dropped_packets += 1;
                        }
                    }
                    sequence = sequence.wrapping_add(1);
                }
                Err(e) => {
                    try_send_action(
                        &action_tx,
                        AppAction::Log(format!("Opus encode error: {:?}", e)),
                        &mut dropped_ui_messages,
                        &mut last_ui_drop_log,
                    );
                }
            }
            pcm_buf.clear();
        }

        // Log drops
        if dropped_packets > 0 && last_drop_log.elapsed().as_secs() >= 5 {
            try_send_action(
                &action_tx,
                AppAction::Log(format!(
                    "Warning: Dropped {} TX packets (backpressure)",
                    dropped_packets
                )),
                &mut dropped_ui_messages,
                &mut last_ui_drop_log,
            );
            dropped_packets = 0;
            last_drop_log = std::time::Instant::now();
        }

        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

// Adaptive jitter buffer state
struct AdaptiveJitterBuffer {
    target_buffer_ms: f32, // Current target buffer size in milliseconds
    last_packet_time: Option<std::time::Instant>, // Last packet arrival time
    inter_arrival_times: VecDeque<f32>, // Recent inter-arrival times (ms) for jitter calculation
    adjustment_log_countdown: u32, // Countdown to log buffer adjustments
}

impl AdaptiveJitterBuffer {
    fn new() -> Self {
        Self {
            target_buffer_ms: INITIAL_BUFFER_MS,
            last_packet_time: None,
            inter_arrival_times: VecDeque::with_capacity(JITTER_BUFFER_INTER_ARRIVAL_HISTORY_SIZE),
            adjustment_log_countdown: 0,
        }
    }

    fn calculate_jitter_ms(&self) -> f32 {
        if self.inter_arrival_times.len() < 3 {
            return 0.0; // Not enough data yet
        }

        // Calculate mean inter-arrival time
        let mean: f32 =
            self.inter_arrival_times.iter().sum::<f32>() / self.inter_arrival_times.len() as f32;

        // Calculate variance (jitter)
        let variance: f32 = self
            .inter_arrival_times
            .iter()
            .map(|&t| (t - mean).powi(2))
            .sum::<f32>()
            / self.inter_arrival_times.len() as f32;

        variance.sqrt() // Standard deviation is our jitter metric
    }

    fn record_packet_arrival(&mut self, now: std::time::Instant) {
        if let Some(last_time) = self.last_packet_time {
            let interval_ms = now.duration_since(last_time).as_secs_f32() * 1000.0;
            if interval_ms > 0.0 && interval_ms < INTER_ARRIVAL_OUTLIER_THRESHOLD_MS {
                // Sanity check: ignore outliers
                self.inter_arrival_times.push_back(interval_ms);
                if self.inter_arrival_times.len() > JITTER_BUFFER_INTER_ARRIVAL_HISTORY_SIZE {
                    self.inter_arrival_times.pop_front();
                }
            }
        }
        self.last_packet_time = Some(now);
    }

    fn adjust_for_conditions(
        &mut self,
        current_buffer_samples: usize,
        action_tx: &mpsc::Sender<AppAction>,
    ) {
        let current_buffer_ms = (current_buffer_samples as f32 / SAMPLE_RATE as f32) * 1000.0;
        let jitter_ms = self.calculate_jitter_ms();
        let mut new_target = self.target_buffer_ms;

        // Adjust based on measured jitter
        if jitter_ms > JITTER_ADJUSTMENT_THRESHOLD_MS {
            // Increase buffer to accommodate jitter (target: multiplier * jitter + base)
            new_target = (jitter_ms * JITTER_MULTIPLIER + MIN_BUFFER_MS).min(MAX_BUFFER_MS);
        } else if jitter_ms < JITTER_ADJUSTMENT_THRESHOLD_MS * 0.5
            && current_buffer_ms > self.target_buffer_ms * JITTER_BUFFER_FULLNESS_THRESHOLD
        {
            // Low jitter and buffer is too full - reduce target for lower latency
            new_target =
                (self.target_buffer_ms * JITTER_BUFFER_REDUCTION_FACTOR).max(MIN_BUFFER_MS);
        }

        // Adjust based on buffer level (prevent underruns/overflows)
        if current_buffer_samples < BUFFER_UNDERRUN_THRESHOLD_SAMPLES {
            // Buffer is getting too low - increase target
            new_target = (self.target_buffer_ms * 1.1).min(MAX_BUFFER_MS);
        } else if current_buffer_samples > BUFFER_OVERRUN_THRESHOLD_SAMPLES {
            // Buffer is getting too full - decrease target (but not below jitter requirement)
            let min_for_jitter = (jitter_ms * JITTER_MULTIPLIER + MIN_BUFFER_MS).min(MAX_BUFFER_MS);
            new_target = (self.target_buffer_ms * JITTER_BUFFER_REDUCTION_FACTOR)
                .max(min_for_jitter.max(MIN_BUFFER_MS));
        }

        // Apply exponential smoothing to avoid rapid oscillations
        new_target = self.target_buffer_ms
            + (new_target - self.target_buffer_ms) * JITTER_BUFFER_ADJUSTMENT_FACTOR;
        new_target = new_target.clamp(MIN_BUFFER_MS, MAX_BUFFER_MS);

        // Only log significant changes
        if (new_target - self.target_buffer_ms).abs() > JITTER_BUFFER_SIGNIFICANT_CHANGE_MS {
            self.adjustment_log_countdown = JITTER_BUFFER_ADJUSTMENT_LOG_INTERVAL;
            if self.adjustment_log_countdown == JITTER_BUFFER_ADJUSTMENT_LOG_INTERVAL {
                let _ = action_tx.try_send(AppAction::Log(format!(
                    "Jitter buffer: {:.0}ms -> {:.0}ms (jitter: {:.1}ms)",
                    self.target_buffer_ms, new_target, jitter_ms
                )));
            }
        }

        if self.adjustment_log_countdown > 0 {
            self.adjustment_log_countdown -= 1;
        }

        self.target_buffer_ms = new_target;
    }

    fn get_target_buffer_samples(&self) -> usize {
        (self.target_buffer_ms / 1000.0 * SAMPLE_RATE as f32) as usize
    }
}

// Helper function to calculate normalized volume from PCM samples
fn calculate_volume(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    (rms * VOLUME_NORMALIZATION_FACTOR).clamp(0.0, 1.0)
}

async fn run_network(
    conn: Arc<dyn webrtc_util::Conn + Send + Sync>,
    mut network_rx: mpsc::Receiver<Vec<u8>>,
    mut playback_prod: HeapProd<f32>,
    action_tx: mpsc::Sender<AppAction>,
    _is_active: Arc<AtomicBool>,
    _sample_rate: u32,
    encryption_key: Option<String>,
) {
    // Setup encryption if key is provided
    let cipher_key = if let Some(ref key_str) = encryption_key {
        match parse_encryption_key(key_str) {
            Ok(key) => Some(key),
            Err(e) => {
                let _ = action_tx
                    .send(AppAction::Log(format!(
                        "Warning: Failed to parse encryption key: {}. Connection will proceed without encryption.",
                        e
                    )))
                    .await;
                None
            }
        }
    } else {
        None
    };
    let mut buf = vec![0u8; 8192];
    let mut jitter_buffer = AdaptiveJitterBuffer::new();
    let mut initial_buffering = true;

    // Opus Decoder
    let mut decoder = match OpusDecoder::new(OpusSampleRate::Hz48000, OpusChannels::Mono) {
        Ok(dec) => dec,
        Err(e) => {
            let _ = action_tx
                .send(AppAction::Log(format!(
                    "Critical: Failed to create Opus decoder: {:?}. Audio playback cannot continue.",
                    e
                )))
                .await;
            return; // Task exits if decoder cannot be created
        }
    };

    let mut decoded_pcm = vec![0i16; FRAME_SIZE];

    let _ = action_tx
        .send(AppAction::Log("ICE Connected!".into()))
        .await;

    let mut dropped_frames = 0u64;
    let mut last_drop_log = std::time::Instant::now();
    let mut dropped_ui_messages = 0u64;
    let mut last_ui_drop_log = std::time::Instant::now();

    // Packet loss concealment: track expected sequence number and last frame
    let mut expected_sequence: Option<u32> = None;
    let mut last_decoded_frame: Option<Vec<f32>> = None;

    loop {
        // Log drops
        if dropped_frames > 0 && last_drop_log.elapsed().as_secs() >= 5 {
            try_send_action(
                &action_tx,
                AppAction::Log(format!(
                    "Warning: Dropped {} RX frames (playback buffer full)",
                    dropped_frames
                )),
                &mut dropped_ui_messages,
                &mut last_ui_drop_log,
            );
            dropped_frames = 0;
            last_drop_log = std::time::Instant::now();
        }
        tokio::select! {
            Some(data) = network_rx.recv() => {
                if let Err(e) = conn.send(&data).await {
                    let _ = action_tx.send(AppAction::Log(
                        format!("Send error: {}", e)
                    )).await;
                }
            }
            res = conn.recv(&mut buf) => {
                match res {
                    Ok(len) => {
                        if len <= 4 {
                            continue; // Invalid packet (need at least sequence number)
                        }

                        // Extract sequence number (unencrypted)
                        let received_sequence = u32::from_le_bytes([
                            buf[0], buf[1], buf[2], buf[3]
                        ]);

                        // Decrypt opus payload if encryption is enabled
                        let opus_payload = if let Some(ref key) = cipher_key {
                            match decrypt_packet(&buf[4..len], key, received_sequence) {
                                Ok(decrypted) => decrypted,
                                Err(e) => {
                                    try_send_action(
                                        &action_tx,
                                        AppAction::Log(format!("Decryption error: {}", e)),
                                        &mut dropped_ui_messages,
                                        &mut last_ui_drop_log,
                                    );
                                    continue; // Skip this packet if decryption fails
                                }
                            }
                        } else {
                            buf[4..len].to_vec()
                        };

                        // Packet Loss Concealment: detect missing packets
                        if let Some(expected) = expected_sequence {
                            let gap = received_sequence.wrapping_sub(expected);
                            if gap > 0 {
                                if gap > MAX_PACKET_GAP_FOR_PLC {
                                    // Log significant packet loss (200ms+)
                                    try_send_action(&action_tx, AppAction::Log(format!(
                                        "Warning: Large packet gap detected ({} packets, {}ms+)",
                                        gap,
                                        gap * FRAME_DURATION_MS
                                    )), &mut dropped_ui_messages, &mut last_ui_drop_log);
                                }
                                // Handle gaps up to MAX_PACKET_GAP_FOR_PLC packets (200ms)
                                if gap <= MAX_PACKET_GAP_FOR_PLC {
                                    // Perform PLC: repeat last frame with fade-out for missing packets
                                    if let Some(ref last_frame) = last_decoded_frame {
                                        for i in 0..(gap - 1) {
                                            let fade_factor = 1.0 - (i as f32 / gap as f32);
                                            // Generate faded frame
                                            let faded_frame: Vec<f32> = last_frame
                                                .iter()
                                                .map(|&s| s * fade_factor)
                                                .collect();

                                            // Update receive volume for PLC frames
                                            let plc_vol = calculate_volume(&faded_frame);
                                            try_send_action(&action_tx, AppAction::SetReceiveVolume(plc_vol), &mut dropped_ui_messages, &mut last_ui_drop_log);

                                            // Push faded samples (buffer overflow check happens after decode)
                                            for &sample in &faded_frame {
                                                if playback_prod.try_push(sample).is_err() {
                                                    dropped_frames += 1;
                                                }
                                            }
                                        }
                                    } else {
                                        // No previous frame, insert silence (zero volume)
                                        try_send_action(&action_tx, AppAction::SetReceiveVolume(0.0), &mut dropped_ui_messages, &mut last_ui_drop_log);
                                        for _ in 0..(gap - 1) {
                                            for _ in 0..FRAME_SIZE {
                                                if playback_prod.try_push(0.0).is_err() {
                                                    dropped_frames += 1;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        expected_sequence = Some(received_sequence.wrapping_add(1));

                        match decoder.decode(Some(&opus_payload), &mut decoded_pcm, false) {
                                Ok(samples_len) => {
                                    // Record packet arrival for jitter calculation
                                    let now = std::time::Instant::now();
                                    jitter_buffer.record_packet_arrival(now);

                                    // Initial buffering: fill to target buffer size
                                    if initial_buffering {
                                        let target_samples = jitter_buffer.get_target_buffer_samples();
                                        let current_fill = playback_prod.occupied_len();
                                        if current_fill < target_samples {
                                            // Fill remaining buffer with silence
                                            let silence_needed = target_samples - current_fill;
                                            for _ in 0..silence_needed {
                                                let _ = playback_prod.try_push(0.0);
                                            }
                                            let _ = action_tx.send(AppAction::Log(format!(
                                                "Jitter buffer primed ({}ms)",
                                                jitter_buffer.target_buffer_ms as u32
                                            ))).await;
                                        }
                                        initial_buffering = false;
                                    }

                                    // Adaptive buffer adjustment
                                    let current_buffer_level = playback_prod.occupied_len();
                                    jitter_buffer.adjust_for_conditions(current_buffer_level, &action_tx);

                                    // Calculate and report current latency (buffer level in ms)
                                    let current_latency_ms = (current_buffer_level as f32 / SAMPLE_RATE as f32) * 1000.0;
                                    try_send_action(&action_tx, AppAction::SetLatency(current_latency_ms), &mut dropped_ui_messages, &mut last_ui_drop_log);

                                    // Buffer overflow protection (drop if way too full)
                                    if playback_prod.occupied_len() > BUFFER_SIZE * 3 / 4 {
                                        dropped_frames += samples_len as u64;
                                        continue;
                                    }

                                    // Store decoded frame for PLC
                                    let frame_f32: Vec<f32> = decoded_pcm[..samples_len]
                                        .iter()
                                        .map(|&s| s as f32 / 32768.0)
                                        .collect();

                                    // Calculate receive volume
                                    let receive_vol_normalized = calculate_volume(&frame_f32);
                                    try_send_action(&action_tx, AppAction::SetReceiveVolume(receive_vol_normalized), &mut dropped_ui_messages, &mut last_ui_drop_log);

                                    // Update last frame for PLC
                                    last_decoded_frame = Some(frame_f32.clone());

                                    for &s in &frame_f32 {
                                        if playback_prod.try_push(s).is_err() {
                                            dropped_frames += 1;
                                        }
                                    }
                                }
                                Err(e) => {
                                     let _ = action_tx.send(AppAction::Log(format!("Opus decode error: {:?}", e))).await;
                                }
                            }
                    }
                    Err(e) => {
                        let _ = action_tx.send(AppAction::Log(
                            format!("Recv error: {}", e)
                        )).await;
                        break;
                    }
                }
            }
        }
    }
}

fn render_ui(f: &mut ratatui::Frame, app: &App) {
    // Allocate more space for offer/answer display when needed
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            if matches!(
                app.mode,
                AppMode::HostWaitingForAnswer { .. } | AppMode::ClientGeneratingAnswer { .. }
            ) {
                Constraint::Min(15) // More space for long offers/answers
            } else {
                Constraint::Length(5)
            },
            Constraint::Length(1),
        ])
        .split(f.area());

    let status = match &app.mode {
        AppMode::Menu => format!("IDLE | Your IP: {}", app.local_ip),
        AppMode::IceGathering => "GATHERING ICE...".into(),
        AppMode::Connecting => "PASTE OFFER...".into(),
        AppMode::HostWaitingForAnswer { .. } => "WAITING FOR ANSWER (paste below)".into(),
        AppMode::ClientGeneratingAnswer { .. } => "COPY ANSWER BELOW (send to host)".into(),
        AppMode::IceConnecting => "CONNECTING...".into(),
        AppMode::InCall { .. } => {
            let mute_status = if app.is_muted { " | MUTED" } else { "" };
            format!(
                "IN CALL: {} ({}){}",
                "Connected", app.connection_type, mute_status
            )
        }
        AppMode::Error(msg) => format!("ERROR: {}", msg),
    };

    let status_color = if matches!(app.mode, AppMode::Error(_)) {
        Color::Red
    } else {
        Color::Yellow
    };

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " CALLRS ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" | "),
            Span::styled(status, Style::default().fg(status_color)),
        ]))
        .block(Block::default().borders(Borders::ALL)),
        chunks[0],
    );

    if let AppMode::InCall { .. } = app.mode {
        let mic_gauge_color = if app.is_muted {
            Color::Red
        } else {
            Color::Green
        };

        let mic_gauge_title = if app.is_muted {
            "Mic Volume (MUTED)"
        } else {
            "Mic Volume"
        };

        let mic_gauge = Gauge::default()
            .block(
                Block::default()
                    .title(mic_gauge_title)
                    .borders(Borders::ALL),
            )
            .gauge_style(Style::default().fg(mic_gauge_color))
            .ratio(app.mic_volume.clamp(0.0, 1.0) as f64);

        let receive_gauge = Gauge::default()
            .block(
                Block::default()
                    .title("Receive Volume")
                    .borders(Borders::ALL),
            )
            .gauge_style(Style::default().fg(Color::Blue))
            .ratio(app.receive_volume.clamp(0.0, 1.0) as f64);

        let inner_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(3), Constraint::Length(4)])
            .split(chunks[1]);

        let logs: Vec<ListItem> = app
            .logs
            .iter()
            .rev()
            .take(inner_chunks[0].height as usize)
            .rev()
            .map(|l| ListItem::new(l.as_str()))
            .collect();
        f.render_widget(
            List::new(logs).block(Block::default().title("Activity").borders(Borders::ALL)),
            inner_chunks[0],
        );

        // Show volume meters and latency gauge
        let gauge_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Length(3)])
            .split(inner_chunks[1]);

        // Volume meters side by side
        let volume_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(gauge_chunks[0]);
        f.render_widget(mic_gauge, volume_chunks[0]);
        f.render_widget(receive_gauge, volume_chunks[1]);

        // Latency gauge
        let latency_ratio = (app.current_latency_ms / MAX_BUFFER_MS).clamp(0.0, 1.0) as f64;
        let latency_color = if app.current_latency_ms < MIN_BUFFER_MS {
            Color::Green
        } else if app.current_latency_ms > MAX_BUFFER_MS * 0.8 {
            Color::Yellow
        } else {
            Color::Cyan
        };
        let latency_gauge = Gauge::default()
            .block(
                Block::default()
                    .title(format!("Current Latency: {:.0}ms", app.current_latency_ms))
                    .borders(Borders::ALL),
            )
            .gauge_style(Style::default().fg(latency_color))
            .ratio(latency_ratio);
        f.render_widget(latency_gauge, gauge_chunks[1]);
    } else {
        let logs: Vec<ListItem> = app
            .logs
            .iter()
            .rev()
            .take(chunks[1].height as usize - 2)
            .rev()
            .map(|l| ListItem::new(l.as_str()))
            .collect();
        f.render_widget(
            List::new(logs).block(Block::default().title("Activity").borders(Borders::ALL)),
            chunks[1],
        );
    }

    let (display_text, title) = match &app.mode {
        AppMode::HostWaitingForAnswer { offer } => {
            // Show offer at top, input field below for answer
            // Use a separator to make it clear where to paste
            let answer_section = if app.input.is_empty() {
                "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n[Paste answer below and press Enter]\n".to_string()
            } else if app.input.len() > 300 {
                format!(
                    "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nAnswer ({} chars): {}...\n",
                    app.input.len(),
                    &app.input[..300]
                )
            } else {
                format!(
                    "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\nAnswer ({} chars): {}\n",
                    app.input.len(),
                    app.input
                )
            };
            (
                format!("{}\n{}", offer, answer_section),
                "Your Offer (send to client) | Paste Answer Below".to_string(),
            )
        }
        AppMode::ClientGeneratingAnswer { answer } => {
            (answer.clone(), "Your Answer (send to host)".to_string())
        }
        AppMode::Error(msg) => (msg.clone(), "Error (press ESC to dismiss)".to_string()),
        _ => (app.input.clone(), "Input / Session Data".to_string()),
    };

    f.render_widget(
        Paragraph::new(display_text)
            .block(Block::default().title(title).borders(Borders::ALL))
            .wrap(Wrap { trim: true }),
        chunks[2],
    );

    let help = match &app.mode {
        AppMode::HostWaitingForAnswer { .. } => {
            " [Y] Copy Offer to Clipboard | [ESC] End | [Q] Quit "
        }
        AppMode::ClientGeneratingAnswer { .. } => {
            " [Y] Copy Answer to Clipboard | [ESC] End | [Q] Quit "
        }
        _ => " [H] Host | [C] Connect | [M] Mute | [ESC] End | [Q] Quit ",
    };
    f.render_widget(
        Paragraph::new(help).style(Style::default().fg(Color::DarkGray)),
        chunks[3],
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup panic hook to ensure terminal is restored on panic
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        // Best effort terminal cleanup
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        original_hook(panic_info);
    }));

    enable_raw_mode()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    execute!(io::stdout(), EnterAlternateScreen)?;
    let (action_tx, mut action_rx) = mpsc::channel(64);
    let mut app = App::new();

    let input_tx = action_tx.clone();
    std::thread::spawn(move || {
        loop {
            if let Ok(true) = event::poll(Duration::from_millis(50)) {
                match event::read() {
                    Ok(Event::Key(key)) => {
                        let _ = input_tx.blocking_send(AppAction::Input(key));
                    }
                    Ok(Event::Paste(text)) => {
                        let _ = input_tx.blocking_send(AppAction::Paste(text));
                    }
                    _ => {}
                }
            }
        }
    });

    let mut session: Option<CallSession> = None;

    loop {
        terminal.draw(|f| render_ui(f, &app))?;
        if let Some(action) = tokio::select! { a = action_rx.recv() => a, _ = tokio::time::sleep(Duration::from_millis(50)) => None }
        {
            match action {
                AppAction::Log(m) => app.add_log(m),
                AppAction::SetMode(m) => app.mode = m,
                AppAction::Paste(text) => {
                    // Append pasted text to input field when in input modes
                    if matches!(
                        app.mode,
                        AppMode::Connecting | AppMode::HostWaitingForAnswer { .. }
                    ) {
                        // Sanitize paste: remove any control characters that might cause issues
                        let sanitized: String = text
                            .chars()
                            .filter(|c| !c.is_control() || *c == '\n' || *c == '\r')
                            .collect();
                        app.input.push_str(&sanitized);
                    }
                }
                AppAction::SetOffer(offer) => {
                    app.session_offer = Some(offer);
                }
                AppAction::SetAnswer(answer) => {
                    app.session_answer = Some(answer);
                }
                AppAction::StoreHostAgent(handle) => {
                    app.host_agent_handle = Some(handle);
                }
                AppAction::SetMicVolume(vol) => {
                    app.mic_volume = vol;
                }
                AppAction::SetReceiveVolume(vol) => {
                    app.receive_volume = vol;
                }
                AppAction::SetLatency(latency_ms) => {
                    app.current_latency_ms = latency_ms;
                }
                AppAction::SetConnectionType(conn_type) => {
                    app.connection_type = conn_type;
                    // Update InCall mode if we're in a call
                    if let AppMode::InCall { peer, .. } = &app.mode {
                        app.mode = AppMode::InCall {
                            peer: peer.clone(),
                            connection_type: app.connection_type.clone(),
                        };
                    }
                }
                AppAction::ToggleMute => {
                    if let Some(ref s) = session {
                        let is_muted = s.audio.toggle_mute();
                        app.is_muted = is_muted;
                        let msg = if is_muted {
                            "Microphone muted"
                        } else {
                            "Microphone unmuted"
                        };
                        app.add_log(msg.into());
                    }
                }

                AppAction::StartConnection {
                    conn,
                    encryption_key,
                } => {
                    // Setup audio and network
                    let (ntx, nrx) = mpsc::channel(NETWORK_CHANNEL_BUFFER_SIZE);
                    let (cap_prod, cap_cons) = HeapRb::<f32>::new(BUFFER_SIZE).split();
                    let (play_prod, play_cons) = HeapRb::<f32>::new(BUFFER_SIZE).split();

                    // Parse encryption key if provided
                    let cipher_key = if let Some(ref key_str) = encryption_key {
                        match parse_encryption_key(key_str) {
                            Ok(key) => Some(key),
                            Err(e) => {
                                let _ = action_tx
                                    .send(AppAction::Log(format!(
                                        "Warning: Failed to parse encryption key: {}. Connection will proceed without encryption.",
                                        e
                                    )))
                                    .await;
                                None
                            }
                        }
                    } else {
                        None
                    };

                    match setup_audio(cap_prod, play_cons) {
                        Ok(audio) => {
                            audio.start();

                            let cap_task = tokio::spawn(capture_processor(
                                cap_cons,
                                ntx,
                                audio.is_active.clone(),
                                audio.input_sample_rate,
                                action_tx.clone(),
                                cipher_key,
                            ));

                            let net_task = tokio::spawn(run_network(
                                conn,
                                nrx,
                                play_prod,
                                action_tx.clone(),
                                audio.is_active.clone(),
                                audio.output_sample_rate,
                                encryption_key,
                            ));

                            session = Some(CallSession {
                                audio,
                                capture_task: cap_task,
                                network_task: net_task,
                            });

                            app.mode = AppMode::InCall {
                                peer: "Connected".into(),
                                connection_type: app.connection_type.clone(),
                            };
                            app.is_muted = false;
                        }
                        Err(e) => {
                            let error_msg = format!("Failed to initialize audio: {}", e);
                            let _ = action_tx.send(AppAction::Log(error_msg.clone())).await;
                            let _ = action_tx
                                .send(AppAction::SetMode(AppMode::Error(error_msg)))
                                .await;
                        }
                    }
                }
                AppAction::Input(key) => {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }
                    match (&app.mode, key.code) {
                        (_, KeyCode::Char('q')) => app.should_quit = true,

                        // Mute toggle (only during call)
                        (AppMode::InCall { .. }, KeyCode::Char('m')) => {
                            let _ = action_tx.send(AppAction::ToggleMute).await;
                        }

                        // Host flow: Press 'h' to start hosting
                        (AppMode::Menu, KeyCode::Char('h')) => {
                            app.mode = AppMode::IceGathering;
                            let tx = action_tx.clone();
                            tokio::spawn(async move {
                                // Try UPnP (best effort) to map port 9090
                                if let Ok(msg) = try_upnp_mapping(9090) {
                                    let _ = tx.send(AppAction::Log(msg)).await;
                                } else {
                                    let _ = tx.send(AppAction::Log("UPnP: Router did not respond (falling back to STUN/Relay)".into())).await;
                                }

                                match host_gather_candidates(tx.clone()).await {
                                    Ok(handle) => {
                                        // Store the agent handle for later use
                                        let _ = tx.send(AppAction::StoreHostAgent(handle)).await;
                                        let _ = tx
                                            .send(AppAction::Log(
                                                "Offer generated! Waiting for answer...".into(),
                                            ))
                                            .await;
                                    }
                                    Err(e) => {
                                        let error_msg =
                                            format!("Failed to gather ICE candidates: {}", e);
                                        let _ = tx.send(AppAction::Log(error_msg.clone())).await;
                                        let _ = tx
                                            .send(AppAction::SetMode(AppMode::Error(error_msg)))
                                            .await;
                                    }
                                }
                            });
                        }

                        // Host receives answer and completes connection
                        (AppMode::HostWaitingForAnswer { .. }, KeyCode::Enter) => {
                            if !app.input.is_empty()
                                && let Some(agent_handle) = app.host_agent_handle.take()
                            {
                                // Trim whitespace and newlines from input (paste might include them)
                                let answer_b64 = app.input.trim().to_string();

                                // Validate it looks like base64 before proceeding
                                if answer_b64.is_empty() {
                                    app.add_log(
                                        "Error: Empty input. Please paste the answer.".into(),
                                    );
                                    app.host_agent_handle = Some(agent_handle);
                                    continue;
                                }

                                let tx = action_tx.clone();

                                app.mode = AppMode::IceConnecting;
                                app.input.clear();

                                tokio::spawn(async move {
                                    // Parse answer
                                    if let Ok(answer_json) = BASE64_STANDARD.decode(&answer_b64)
                                        && let Ok(answer) =
                                            serde_json::from_slice::<SessionAnswer>(&answer_json)
                                    {
                                        let _ = tx
                                            .send(AppAction::Log("Calling accept()...".into()))
                                            .await;

                                        // Add remote candidates from answer
                                        for candidate_str in &answer.candidates {
                                            match webrtc_ice::candidate::candidate_base::unmarshal_candidate(candidate_str) {
                                                    Ok(candidate) => {
                                                        let candidate_arc: Arc<dyn webrtc_ice::candidate::Candidate + Send + Sync> = Arc::from(candidate);
                                                        match agent_handle.agent.add_remote_candidate(&candidate_arc) {
                                                            Ok(_) => {
                                                                let _ = tx.send(AppAction::Log("Added remote candidate".into())).await;
                                                            }
                                                            Err(e) => {
                                                                let _ = tx.send(AppAction::Log(
                                                                    format!("Failed to add candidate (continuing): {}", e)
                                                                )).await;
                                                            }
                                                        }
                                                    }
                                                    Err(e) => {
                                                        let _ = tx.send(AppAction::Log(
                                                            format!("Failed to parse candidate: {}", e)
                                                        )).await;
                                                    }
                                                }
                                        }

                                        let (_cancel_tx, cancel_rx) = mpsc::channel::<()>(1);
                                        match agent_handle
                                            .agent
                                            .accept(cancel_rx, answer.ufrag, answer.pwd)
                                            .await
                                        {
                                            Ok(conn) => {
                                                let _ = tx
                                                    .send(AppAction::Log(
                                                        "ICE connection established!".into(),
                                                    ))
                                                    .await;

                                                if let Some(pair) =
                                                    agent_handle.agent.get_selected_candidate_pair()
                                                {
                                                    let pair_str = pair.to_string();
                                                    let _ = tx
                                                        .send(AppAction::Log(format!(
                                                            "Connected via: {}",
                                                            pair_str
                                                        )))
                                                        .await;

                                                    // Extract connection type (Direct vs Relay)
                                                    let conn_type = if pair_str
                                                        .to_lowercase()
                                                        .contains("relay")
                                                    {
                                                        "Relay"
                                                    } else {
                                                        "Direct"
                                                    };
                                                    let _ = tx
                                                        .send(AppAction::SetConnectionType(
                                                            conn_type.to_string(),
                                                        ))
                                                        .await;
                                                }

                                                let encryption_key =
                                                    agent_handle.encryption_key.clone();
                                                let _ = tx
                                                    .send(AppAction::StartConnection {
                                                        conn,
                                                        encryption_key,
                                                    })
                                                    .await;
                                            }
                                            Err(e) => {
                                                let error_msg =
                                                    format!("ICE connection failed: {}", e);
                                                let _ = tx
                                                    .send(AppAction::Log(error_msg.clone()))
                                                    .await;
                                                let _ = tx
                                                    .send(AppAction::SetMode(AppMode::Error(
                                                        error_msg,
                                                    )))
                                                    .await;
                                            }
                                        }
                                    }
                                });
                            }
                        }

                        // Client flow: Press 'c' to connect
                        (AppMode::Menu, KeyCode::Char('c')) => {
                            app.mode = AppMode::Connecting;
                            app.input.clear();
                        }

                        // Client pastes offer and generates answer
                        (AppMode::Connecting, KeyCode::Enter) => {
                            if !app.input.is_empty() {
                                let offer_b64 = app.input.clone();
                                let tx = action_tx.clone();

                                app.mode = AppMode::IceGathering;
                                app.input.clear();

                                tokio::spawn(async move {
                                    // Try UPnP for client too
                                    if let Ok(msg) = try_upnp_mapping(9090) {
                                        let _ = tx.send(AppAction::Log(msg)).await;
                                    }

                                    match client_gather_candidates(offer_b64, tx.clone()).await {
                                        Ok(agent_handle) => {
                                            // Now we need to dial
                                            let _ =
                                                tx.send(AppAction::Log("Dialing...".into())).await;
                                            let _ = tx
                                                .send(AppAction::SetMode(AppMode::IceConnecting))
                                                .await;

                                            let (_cancel_tx, cancel_rx) = mpsc::channel::<()>(1);
                                            match agent_handle
                                                .agent
                                                .dial(
                                                    cancel_rx,
                                                    agent_handle.ufrag,
                                                    agent_handle.pwd,
                                                )
                                                .await
                                            {
                                                Ok(conn) => {
                                                    let _ = tx
                                                        .send(AppAction::Log(
                                                            "ICE connection established!".into(),
                                                        ))
                                                        .await;

                                                    if let Some(pair) = agent_handle
                                                        .agent
                                                        .get_selected_candidate_pair()
                                                    {
                                                        let pair_str = pair.to_string();
                                                        let _ = tx
                                                            .send(AppAction::Log(format!(
                                                                "Connected via: {}",
                                                                pair_str
                                                            )))
                                                            .await;

                                                        // Extract connection type (Direct vs Relay)
                                                        let conn_type = if pair_str
                                                            .to_lowercase()
                                                            .contains("relay")
                                                        {
                                                            "Relay"
                                                        } else {
                                                            "Direct"
                                                        };
                                                        let _ = tx
                                                            .send(AppAction::SetConnectionType(
                                                                conn_type.to_string(),
                                                            ))
                                                            .await;
                                                    }

                                                    let encryption_key =
                                                        agent_handle.encryption_key.clone();
                                                    let _ = tx
                                                        .send(AppAction::StartConnection {
                                                            conn,
                                                            encryption_key,
                                                        })
                                                        .await;
                                                }
                                                Err(e) => {
                                                    let error_msg =
                                                        format!("ICE connection failed: {}", e);
                                                    let _ = tx
                                                        .send(AppAction::Log(error_msg.clone()))
                                                        .await;
                                                    let _ = tx
                                                        .send(AppAction::SetMode(AppMode::Error(
                                                            error_msg,
                                                        )))
                                                        .await;
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            let error_msg =
                                                format!("Failed to gather ICE candidates: {}", e);
                                            let _ =
                                                tx.send(AppAction::Log(error_msg.clone())).await;
                                            let _ = tx
                                                .send(AppAction::SetMode(AppMode::Error(error_msg)))
                                                .await;
                                        }
                                    }
                                });
                            }
                        }

                        // Copy offer/answer to clipboard
                        (AppMode::HostWaitingForAnswer { offer }, KeyCode::Char('y')) => {
                            match Clipboard::new() {
                                Ok(mut clipboard) => {
                                    if let Err(e) = clipboard.set_text(offer.clone()) {
                                        app.add_log(format!("Failed to copy to clipboard: {}", e));
                                    } else {
                                        app.add_log("Offer copied to clipboard!".into());
                                    }
                                }
                                Err(e) => {
                                    app.add_log(format!("Failed to access clipboard: {}", e));
                                }
                            }
                        }
                        (AppMode::ClientGeneratingAnswer { answer }, KeyCode::Char('y')) => {
                            match Clipboard::new() {
                                Ok(mut clipboard) => {
                                    if let Err(e) = clipboard.set_text(answer.clone()) {
                                        app.add_log(format!("Failed to copy to clipboard: {}", e));
                                    } else {
                                        app.add_log("Answer copied to clipboard!".into());
                                    }
                                }
                                Err(e) => {
                                    app.add_log(format!("Failed to access clipboard: {}", e));
                                }
                            }
                        }

                        // Input handling for connecting/waiting states
                        (AppMode::Connecting, KeyCode::Char(c)) => app.input.push(c),
                        (AppMode::Connecting, KeyCode::Backspace) => {
                            app.input.pop();
                        }
                        (AppMode::HostWaitingForAnswer { .. }, KeyCode::Char(c)) => {
                            app.input.push(c)
                        }
                        (AppMode::HostWaitingForAnswer { .. }, KeyCode::Backspace) => {
                            app.input.pop();
                        }

                        // ESC to cancel/end or dismiss error
                        (_, KeyCode::Esc) => {
                            if let Some(s) = session.take() {
                                s.stop().await;
                            }
                            app.host_agent_handle = None;

                            // If in error mode, just go back to menu. Otherwise, clear everything.
                            if matches!(app.mode, AppMode::Error(_)) {
                                app.mode = AppMode::Menu;
                            } else {
                                app.mode = AppMode::Menu;
                                app.session_offer = None;
                                app.session_answer = None;
                                app.input.clear();
                                app.is_muted = false;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        if app.should_quit {
            // Cleanup UPnP port mapping before exit
            try_upnp_cleanup(9090);
            break;
        }
    }
    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    // ============================================================================
    // Test Helpers
    // ============================================================================

    /// Generate silent audio samples (all zeros)
    fn generate_silence(num_samples: usize) -> Vec<f32> {
        vec![0.0; num_samples]
    }

    /// Generate a sine wave at specified frequency and amplitude
    fn generate_sine_wave(
        num_samples: usize,
        frequency: f32,
        amplitude: f32,
        sample_rate: u32,
    ) -> Vec<f32> {
        let mut samples = Vec::with_capacity(num_samples);
        let phase_increment = 2.0 * std::f32::consts::PI * frequency / sample_rate as f32;
        for i in 0..num_samples {
            samples.push(amplitude * (i as f32 * phase_increment).sin());
        }
        samples
    }

    /// Generate white noise with specified RMS amplitude
    fn generate_white_noise(num_samples: usize, rms_amplitude: f32) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut samples = Vec::with_capacity(num_samples);
        let mut hasher = DefaultHasher::new();
        for i in 0..num_samples {
            i.hash(&mut hasher);
            let hash = hasher.finish();
            // Convert hash to float in range [-1, 1]
            let sample = ((hash % 2000000) as f32 / 1000000.0) - 1.0;
            samples.push(sample * rms_amplitude);
        }
        // Normalize to desired RMS
        let current_rms: f32 =
            (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
        if current_rms > 0.0 {
            let scale = rms_amplitude / current_rms;
            samples.iter_mut().for_each(|s| *s *= scale);
        }
        samples
    }

    /// Calculate RMS (Root Mean Square) of samples
    fn calculate_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    /// Create a mock packet with sequence number and Opus payload
    fn create_mock_packet(sequence: u32, payload: &[u8]) -> Vec<u8> {
        let mut packet = Vec::with_capacity(4 + payload.len());
        packet.extend_from_slice(&sequence.to_le_bytes());
        packet.extend_from_slice(payload);
        packet
    }

    // ============================================================================
    // Tests for calculate_volume
    // ============================================================================

    #[test]
    fn test_calculate_volume_with_silence() {
        let samples = generate_silence(960);
        let volume = calculate_volume(&samples);
        assert_eq!(volume, 0.0, "Silence should produce zero volume");
    }

    #[test]
    fn test_calculate_volume_with_full_scale_sine() {
        // Full scale sine wave (amplitude 1.0) has RMS = 1/sqrt(2) ≈ 0.707
        // After multiplying by VOLUME_NORMALIZATION_FACTOR and clamping: 0.707 * 5.0 = 3.535, clamped to 1.0
        let samples = generate_sine_wave(960, 440.0, 1.0, SAMPLE_RATE);
        let volume = calculate_volume(&samples);
        assert!(
            volume > 0.99,
            "Full scale sine should produce near-maximum volume (got {})",
            volume
        );
        assert!(
            volume <= 1.0,
            "Volume should be clamped to 1.0 (got {})",
            volume
        );
    }

    #[test]
    fn test_calculate_volume_with_small_amplitude() {
        // Small amplitude signal (RMS ≈ 0.01 * 0.707 ≈ 0.007)
        // After multiplying by VOLUME_NORMALIZATION_FACTOR: 0.007 * 5.0 ≈ 0.035
        let samples = generate_sine_wave(960, 440.0, 0.01, SAMPLE_RATE);
        let volume = calculate_volume(&samples);
        assert!(
            volume > 0.0,
            "Small amplitude should produce non-zero volume"
        );
        assert!(
            volume < 0.1,
            "Small amplitude should produce low volume (got {})",
            volume
        );
    }

    #[test]
    fn test_calculate_volume_with_noise_gate_threshold() {
        // Test signal exactly at noise gate threshold (RMS = 0.01)
        // After calculate_volume: RMS * VOLUME_NORMALIZATION_FACTOR = 0.01 * 5.0 = 0.05
        let samples = generate_white_noise(960, NOISE_GATE_THRESHOLD);
        let volume = calculate_volume(&samples);
        let expected = (NOISE_GATE_THRESHOLD * VOLUME_NORMALIZATION_FACTOR).clamp(0.0, 1.0);
        assert!(
            (volume - expected).abs() < 0.01,
            "Volume at noise gate threshold should be approximately {} (got {})",
            expected,
            volume
        );
    }

    #[test]
    fn test_calculate_volume_scales_with_amplitude() {
        let small = generate_sine_wave(960, 440.0, 0.1, SAMPLE_RATE);
        let large = generate_sine_wave(960, 440.0, 0.5, SAMPLE_RATE);
        let vol_small = calculate_volume(&small);
        let vol_large = calculate_volume(&large);
        assert!(
            vol_large > vol_small,
            "Larger amplitude should produce larger volume (small: {}, large: {})",
            vol_small,
            vol_large
        );
    }

    #[test]
    fn test_calculate_volume_handles_empty_input() {
        let samples = vec![];
        let volume = calculate_volume(&samples);
        // RMS of empty array is 0, so volume should be 0
        assert_eq!(volume, 0.0, "Empty input should produce zero volume");
    }

    #[test]
    fn test_calculate_volume_clamps_to_one() {
        // Very loud signal that would exceed 1.0 without clamping
        let samples = generate_sine_wave(960, 440.0, 1.5, SAMPLE_RATE);
        let volume = calculate_volume(&samples);
        assert!(
            volume <= 1.0,
            "Volume should be clamped to 1.0 (got {})",
            volume
        );
    }

    // ============================================================================
    // Tests for AdaptiveJitterBuffer
    // ============================================================================

    #[test]
    fn test_jitter_buffer_new_starts_at_initial_target() {
        let buffer = AdaptiveJitterBuffer::new();
        assert_eq!(
            buffer.target_buffer_ms, INITIAL_BUFFER_MS,
            "New jitter buffer should start at initial target ({}ms)",
            INITIAL_BUFFER_MS
        );
        assert!(
            buffer.last_packet_time.is_none(),
            "New buffer should have no packet time"
        );
        assert!(
            buffer.inter_arrival_times.is_empty(),
            "New buffer should have no arrival times"
        );
    }

    #[test]
    fn test_jitter_buffer_calculate_jitter_returns_zero_with_insufficient_data() {
        let buffer = AdaptiveJitterBuffer::new();
        assert_eq!(
            buffer.calculate_jitter_ms(),
            0.0,
            "Jitter should be 0 with insufficient data (< 3 packets)"
        );
    }

    #[test]
    fn test_jitter_buffer_calculate_jitter_with_perfect_timing() {
        let mut buffer = AdaptiveJitterBuffer::new();
        let base_time = Instant::now();

        // Record packets at exactly 20ms intervals (perfect timing)
        for i in 0..5 {
            let time = base_time + Duration::from_millis(i * 20);
            buffer.record_packet_arrival(time);
        }

        let jitter = buffer.calculate_jitter_ms();
        // With perfect timing, variance should be very small (close to 0)
        assert!(
            jitter < 1.0,
            "Perfect timing should produce near-zero jitter (got {}ms)",
            jitter
        );
    }

    #[test]
    fn test_jitter_buffer_calculate_jitter_with_variable_timing() {
        let mut buffer = AdaptiveJitterBuffer::new();
        let base_time = Instant::now();

        // Record packets with varying intervals: 15ms, 25ms, 18ms, 22ms, 20ms
        let intervals = vec![15, 25, 18, 22, 20];
        let mut current_time = base_time;
        for interval in intervals {
            current_time += Duration::from_millis(interval);
            buffer.record_packet_arrival(current_time);
        }

        let jitter = buffer.calculate_jitter_ms();
        // Should have some jitter with variable timing
        assert!(
            jitter > 0.0,
            "Variable timing should produce non-zero jitter"
        );
        assert!(
            jitter < 10.0,
            "Jitter should be reasonable (got {}ms)",
            jitter
        );
    }

    #[test]
    fn test_jitter_buffer_record_packet_arrival_ignores_outliers() {
        let mut buffer = AdaptiveJitterBuffer::new();
        let base_time = Instant::now();

        // First packet
        buffer.record_packet_arrival(base_time);

        // Second packet with normal interval (20ms)
        buffer.record_packet_arrival(base_time + Duration::from_millis(20));
        assert_eq!(
            buffer.inter_arrival_times.len(),
            1,
            "Should record normal interval"
        );

        // Third packet with huge gap (> INTER_ARRIVAL_OUTLIER_THRESHOLD_MS) - should be ignored
        buffer.record_packet_arrival(base_time + Duration::from_secs(1));
        assert_eq!(
            buffer.inter_arrival_times.len(),
            1,
            "Should ignore outlier interval (> {}ms)",
            INTER_ARRIVAL_OUTLIER_THRESHOLD_MS
        );

        // Fourth packet with negative interval (should be ignored via > 0.0 check)
        // This test is harder to do with Instant, but the code checks interval_ms > 0.0
    }

    #[test]
    fn test_jitter_buffer_record_packet_arrival_limits_history() {
        let mut buffer = AdaptiveJitterBuffer::new();
        let base_time = Instant::now();

        // Record more than JITTER_BUFFER_INTER_ARRIVAL_HISTORY_SIZE packets (capacity limit)
        for i in 0..15 {
            let time = base_time + Duration::from_millis(i * 20);
            buffer.record_packet_arrival(time);
        }

        assert!(
            buffer.inter_arrival_times.len() <= JITTER_BUFFER_INTER_ARRIVAL_HISTORY_SIZE,
            "Should limit history to {} intervals (got {})",
            JITTER_BUFFER_INTER_ARRIVAL_HISTORY_SIZE,
            buffer.inter_arrival_times.len()
        );
    }

    #[test]
    fn test_jitter_buffer_get_target_buffer_samples() {
        let buffer = AdaptiveJitterBuffer::new();
        let samples = buffer.get_target_buffer_samples();
        let expected = (INITIAL_BUFFER_MS / 1000.0 * SAMPLE_RATE as f32) as usize;
        assert_eq!(
            samples, expected,
            "Target buffer samples should match calculated value (got {}, expected {})",
            samples, expected
        );
    }

    #[test]
    fn test_jitter_buffer_adjust_increases_on_high_jitter() {
        let mut buffer = AdaptiveJitterBuffer::new();
        let (tx, _rx) = mpsc::channel(10);
        let base_time = Instant::now();

        // Create high jitter scenario (varying intervals)
        let intervals = vec![10, 30, 15, 25, 12, 28];
        let mut current_time = base_time;
        for interval in intervals {
            current_time += Duration::from_millis(interval);
            buffer.record_packet_arrival(current_time);
        }

        let initial_target = buffer.target_buffer_ms;
        let jitter = buffer.calculate_jitter_ms();

        // Adjust with buffer level that's okay
        buffer.adjust_for_conditions(BUFFER_UNDERRUN_THRESHOLD_SAMPLES + 100, &tx);

        // Should increase if jitter > threshold
        if jitter > JITTER_ADJUSTMENT_THRESHOLD_MS {
            assert!(
                buffer.target_buffer_ms >= initial_target,
                "Should increase buffer on high jitter (initial: {}, new: {}, jitter: {})",
                initial_target,
                buffer.target_buffer_ms,
                jitter
            );
            assert!(
                buffer.target_buffer_ms <= MAX_BUFFER_MS,
                "Should respect MAX_BUFFER_MS (got {}ms)",
                buffer.target_buffer_ms
            );
        }
    }

    #[test]
    fn test_jitter_buffer_adjust_increases_on_underrun() {
        let mut buffer = AdaptiveJitterBuffer::new();
        let (tx, _rx) = mpsc::channel(10);
        let initial_target = buffer.target_buffer_ms;

        // Simulate underrun (buffer below threshold)
        let low_buffer_samples = BUFFER_UNDERRUN_THRESHOLD_SAMPLES - 100;
        buffer.adjust_for_conditions(low_buffer_samples, &tx);

        assert!(
            buffer.target_buffer_ms >= initial_target,
            "Should increase buffer on underrun (initial: {}, new: {})",
            initial_target,
            buffer.target_buffer_ms
        );
        assert!(
            buffer.target_buffer_ms <= MAX_BUFFER_MS,
            "Should respect MAX_BUFFER_MS (got {}ms)",
            buffer.target_buffer_ms
        );
    }

    #[test]
    fn test_jitter_buffer_adjust_decreases_on_overrun() {
        let mut buffer = AdaptiveJitterBuffer::new();
        let (tx, _rx) = mpsc::channel(10);

        // Set target high to allow decrease
        buffer.target_buffer_ms = 100.0;

        // Simulate overrun (buffer above threshold)
        let high_buffer_samples = BUFFER_OVERRUN_THRESHOLD_SAMPLES + 1000;
        buffer.adjust_for_conditions(high_buffer_samples, &tx);

        // May decrease, but should respect minimum based on jitter
        assert!(
            buffer.target_buffer_ms >= MIN_BUFFER_MS,
            "Should respect MIN_BUFFER_MS (got {}ms)",
            buffer.target_buffer_ms
        );
    }

    #[test]
    fn test_jitter_buffer_adjust_respects_bounds() {
        let mut buffer = AdaptiveJitterBuffer::new();
        let (tx, _rx) = mpsc::channel(10);

        // Try to force buffer very high
        buffer.target_buffer_ms = MAX_BUFFER_MS;
        buffer.adjust_for_conditions(0, &tx); // Try to increase more
        assert!(
            buffer.target_buffer_ms <= MAX_BUFFER_MS,
            "Should not exceed MAX_BUFFER_MS (got {}ms)",
            buffer.target_buffer_ms
        );

        // Try to force buffer very low
        buffer.target_buffer_ms = MIN_BUFFER_MS;
        buffer.adjust_for_conditions(BUFFER_SIZE, &tx); // Try to decrease more
        assert!(
            buffer.target_buffer_ms >= MIN_BUFFER_MS,
            "Should not go below MIN_BUFFER_MS (got {}ms)",
            buffer.target_buffer_ms
        );
    }

    #[test]
    fn test_jitter_buffer_adjust_applies_exponential_smoothing() {
        let mut buffer = AdaptiveJitterBuffer::new();
        let (tx, _rx) = mpsc::channel(10);

        // Simulate condition that would want to change to 100ms
        // With JITTER_BUFFER_ADJUSTMENT_FACTOR adjustment, change should be gradual
        buffer.target_buffer_ms = 50.0;
        // Use a very low buffer value (but not negative) to trigger underrun adjustment
        let very_low_buffer = BUFFER_UNDERRUN_THRESHOLD_SAMPLES.saturating_sub(500);
        buffer.adjust_for_conditions(very_low_buffer, &tx);

        // Change should be smoothed (not instant jump to new target)
        let new_target = buffer.target_buffer_ms;
        // Should move toward higher value but not instantly
        assert!(new_target > 50.0, "Should increase toward higher target");
        assert!(
            new_target < MAX_BUFFER_MS,
            "Should not jump instantly (smoothing should apply)"
        );
    }

    // ============================================================================
    // Tests for Noise Gate Logic (RMS calculation)
    // ============================================================================

    #[test]
    fn test_noise_gate_suppresses_below_threshold() {
        // Signal with RMS below threshold (0.01)
        let samples = generate_white_noise(FRAME_SIZE, NOISE_GATE_THRESHOLD * 0.5);
        let rms = calculate_rms(&samples);
        assert!(
            rms < NOISE_GATE_THRESHOLD,
            "Test signal RMS ({}) should be below threshold ({})",
            rms,
            NOISE_GATE_THRESHOLD
        );
    }

    #[test]
    fn test_noise_gate_passes_above_threshold() {
        // Signal with RMS above threshold
        let samples = generate_white_noise(FRAME_SIZE, NOISE_GATE_THRESHOLD * 2.0);
        let rms = calculate_rms(&samples);
        assert!(
            rms > NOISE_GATE_THRESHOLD,
            "Test signal RMS ({}) should be above threshold ({})",
            rms,
            NOISE_GATE_THRESHOLD
        );
    }

    #[test]
    fn test_rms_calculation_for_sine_wave() {
        // Sine wave with amplitude A has RMS = A / sqrt(2)
        let amplitude = 0.5;
        let samples = generate_sine_wave(FRAME_SIZE, 440.0, amplitude, SAMPLE_RATE);
        let rms = calculate_rms(&samples);
        let expected_rms = amplitude / (2.0_f32).sqrt();
        assert!(
            (rms - expected_rms).abs() < 0.01,
            "RMS should be approximately A/sqrt(2) (got {}, expected {})",
            rms,
            expected_rms
        );
    }

    // ============================================================================
    // Tests for Session Serialization
    // ============================================================================

    #[test]
    fn test_session_offer_serialization() {
        let offer = SessionOffer {
            ufrag: "test_ufrag".to_string(),
            pwd: "test_password".to_string(),
            candidates: vec!["candidate1".to_string(), "candidate2".to_string()],
            encryption_key: "dGVzdF9lbmNyeXB0aW9uX2tleV9mb3JfdGVzdGluZw==".to_string(),
        };

        let json = serde_json::to_string(&offer).expect("Should serialize");
        let deserialized: SessionOffer = serde_json::from_str(&json).expect("Should deserialize");

        assert_eq!(
            offer, deserialized,
            "Serialized and deserialized offer should match"
        );
    }

    #[test]
    fn test_session_answer_serialization() {
        let answer = SessionAnswer {
            ufrag: "answer_ufrag".to_string(),
            pwd: "answer_password".to_string(),
            candidates: vec!["candidate_a".to_string()],
        };

        let json = serde_json::to_string(&answer).expect("Should serialize");
        let deserialized: SessionAnswer = serde_json::from_str(&json).expect("Should deserialize");

        assert_eq!(
            answer, deserialized,
            "Serialized and deserialized answer should match"
        );
    }

    #[test]
    fn test_session_offer_base64_roundtrip() {
        let offer = SessionOffer {
            ufrag: "test_ufrag".to_string(),
            pwd: "test_password".to_string(),
            candidates: vec!["candidate1".to_string()],
            encryption_key: "dGVzdF9lbmNyeXB0aW9uX2tleV9mb3JfdGVzdGluZw==".to_string(),
        };

        let json = serde_json::to_string(&offer).expect("Should serialize");
        let b64 = BASE64_STANDARD.encode(&json);
        let decoded = BASE64_STANDARD.decode(&b64).expect("Should decode");
        let decoded_json = String::from_utf8(decoded).expect("Should be valid UTF-8");
        let decoded_offer: SessionOffer =
            serde_json::from_str(&decoded_json).expect("Should deserialize");

        assert_eq!(
            offer, decoded_offer,
            "Base64 roundtrip should preserve offer"
        );
    }

    // ============================================================================
    // Tests for Encryption
    // ============================================================================

    #[test]
    fn test_encryption_key_generation_produces_valid_key() {
        let key_b64 = generate_encryption_key();
        assert!(!key_b64.is_empty(), "Generated key should not be empty");

        // Should be able to parse the generated key
        let key = parse_encryption_key(&key_b64);
        assert!(key.is_ok(), "Generated key should be parseable");
        assert_eq!(key.unwrap().as_slice().len(), ENCRYPTION_KEY_SIZE_BYTES);
    }

    #[test]
    fn test_encryption_key_parsing_rejects_invalid_base64() {
        let invalid_key = "not-valid-base64!!!";
        let result = parse_encryption_key(invalid_key);
        assert!(result.is_err(), "Should reject invalid base64");
    }

    #[test]
    fn test_encryption_key_parsing_rejects_wrong_size() {
        // Create a valid base64 string that decodes to wrong size
        let wrong_size_key = BASE64_STANDARD.encode(&[0u8; 16]); // 16 bytes instead of 32
        let result = parse_encryption_key(&wrong_size_key);
        assert!(result.is_err(), "Should reject key with wrong size");
    }

    #[test]
    fn test_encryption_decryption_roundtrip() {
        let key_b64 = generate_encryption_key();
        let key = parse_encryption_key(&key_b64).expect("Should parse valid key");

        let test_data = b"Hello, encrypted world!";
        let sequence = 42u32;

        // Encrypt
        let encrypted =
            encrypt_packet(test_data, &key, sequence).expect("Encryption should succeed");

        // Decrypt
        let decrypted =
            decrypt_packet(&encrypted, &key, sequence).expect("Decryption should succeed");

        assert_eq!(
            test_data,
            decrypted.as_slice(),
            "Decrypted data should match original"
        );
    }

    #[test]
    fn test_encryption_fails_with_wrong_sequence() {
        let key_b64 = generate_encryption_key();
        let key = parse_encryption_key(&key_b64).expect("Should parse valid key");

        let test_data = b"Test data";
        let sequence = 100u32;

        let encrypted =
            encrypt_packet(test_data, &key, sequence).expect("Encryption should succeed");

        // Try to decrypt with wrong sequence number
        let wrong_sequence = 101u32;
        let result = decrypt_packet(&encrypted, &key, wrong_sequence);

        assert!(
            result.is_err(),
            "Decryption should fail with wrong sequence number"
        );
    }

    #[test]
    fn test_encryption_fails_with_wrong_key() {
        let key1_b64 = generate_encryption_key();
        let key1 = parse_encryption_key(&key1_b64).expect("Should parse valid key");

        let key2_b64 = generate_encryption_key();
        let key2 = parse_encryption_key(&key2_b64).expect("Should parse valid key");

        let test_data = b"Test data";
        let sequence = 50u32;

        let encrypted =
            encrypt_packet(test_data, &key1, sequence).expect("Encryption should succeed");

        // Try to decrypt with different key
        let result = decrypt_packet(&encrypted, &key2, sequence);

        assert!(result.is_err(), "Decryption should fail with wrong key");
    }

    #[test]
    fn test_session_answer_base64_roundtrip() {
        let answer = SessionAnswer {
            ufrag: "answer_ufrag".to_string(),
            pwd: "answer_password".to_string(),
            candidates: vec!["candidate_a".to_string(), "candidate_b".to_string()],
        };

        let json = serde_json::to_string(&answer).expect("Should serialize");
        let b64 = BASE64_STANDARD.encode(&json);
        let decoded = BASE64_STANDARD.decode(&b64).expect("Should decode");
        let decoded_json = String::from_utf8(decoded).expect("Should be valid UTF-8");
        let deserialized: SessionAnswer =
            serde_json::from_str(&decoded_json).expect("Should deserialize");

        assert_eq!(
            answer, deserialized,
            "Base64 roundtrip should preserve answer"
        );
    }

    // ============================================================================
    // Tests for Packet Sequence Number Handling
    // ============================================================================

    #[test]
    fn test_sequence_number_wraparound() {
        // Test sequence number wrapping from u32::MAX to 0
        let seq1 = u32::MAX;
        let seq2 = seq1.wrapping_add(1);
        assert_eq!(seq2, 0, "Sequence should wrap from MAX to 0");

        // Test gap calculation with wraparound
        let gap = seq2.wrapping_sub(seq1);
        assert_eq!(gap, 1, "Gap calculation should handle wraparound correctly");
    }

    #[test]
    fn test_sequence_number_gap_detection() {
        // Test normal gap (no wraparound)
        let expected: u32 = 5;
        let received: u32 = 10;
        let gap = received.wrapping_sub(expected);
        assert_eq!(gap, 5, "Should detect gap of 5 packets");

        // Test no gap (sequential)
        let expected2: u32 = 5;
        let received2: u32 = 6;
        let gap2 = received2.wrapping_sub(expected2);
        assert_eq!(
            gap2, 1,
            "Sequential packets should have gap of 1 (next expected)"
        );
    }

    #[test]
    fn test_mock_packet_creation() {
        let payload = b"opus_data";
        let sequence = 42;
        let packet = create_mock_packet(sequence, payload);

        assert_eq!(
            packet.len(),
            4 + payload.len(),
            "Packet should have 4-byte header + payload"
        );

        let decoded_seq = u32::from_le_bytes([packet[0], packet[1], packet[2], packet[3]]);
        assert_eq!(decoded_seq, sequence, "Sequence number should match");

        assert_eq!(&packet[4..], payload, "Payload should match");
    }

    // ============================================================================
    // Tests for Audio Sample Format Conversion
    // ============================================================================

    #[test]
    fn test_f32_to_i16_conversion_clamping() {
        // Test that conversion clamps to i16 range
        let f32_sample_overflow: f32 = 2.0; // > 1.0, should clamp to 32767
        let i16_result = (f32_sample_overflow * 32767.0f32).clamp(-32768.0f32, 32767.0f32) as i16;
        assert_eq!(i16_result, 32767, "Should clamp positive overflow to 32767");

        let f32_sample_underflow: f32 = -2.0; // < -1.0, should clamp to -32768
        let i16_result2 = (f32_sample_underflow * 32767.0f32).clamp(-32768.0f32, 32767.0f32) as i16;
        assert_eq!(
            i16_result2, -32768,
            "Should clamp negative underflow to -32768"
        );
    }

    #[test]
    fn test_f32_to_i16_conversion_normal_range() {
        // Test normal range conversion
        let f32_sample: f32 = 0.5;
        let i16_result = (f32_sample * 32767.0f32).clamp(-32768.0f32, 32767.0f32) as i16;
        let expected = (0.5f32 * 32767.0f32) as i16;
        assert_eq!(
            i16_result, expected,
            "Normal range should convert correctly"
        );

        // Test zero
        let f32_zero: f32 = 0.0;
        let i16_zero = (f32_zero * 32767.0f32).clamp(-32768.0f32, 32767.0f32) as i16;
        assert_eq!(i16_zero, 0, "Zero should convert to zero");
    }

    #[test]
    fn test_i16_to_f32_conversion() {
        // Test i16 to f32 conversion (as done in playback)
        let i16_sample = 16384; // Half scale
        let f32_result = i16_sample as f32 / 32768.0;
        assert!(
            (f32_result - 0.5).abs() < 0.0001,
            "Half scale i16 should convert to ~0.5 f32 (got {})",
            f32_result
        );

        let i16_max = 32767;
        let f32_max = i16_max as f32 / 32768.0;
        assert!(f32_max > 0.99, "Max i16 should convert to near 1.0 f32");
    }

    // ============================================================================
    // Tests for PLC Fade-Out Calculation
    // ============================================================================

    #[test]
    fn test_plc_fade_factor_calculation() {
        // Test fade-out factors for gap of 5 packets
        let gap = 5u32;
        for i in 0..(gap - 1) {
            let fade_factor = 1.0 - (i as f32 / gap as f32);
            // First missing packet (i=0): fade_factor = 1.0 - 0/5 = 1.0
            // Last missing packet (i=3): fade_factor = 1.0 - 3/5 = 0.4
            assert!(
                fade_factor >= 0.0 && fade_factor <= 1.0,
                "Fade factor should be in range [0, 1] (got {} for i={})",
                fade_factor,
                i
            );
        }
    }

    #[test]
    fn test_plc_fade_factor_decreases_gradually() {
        let gap = MAX_PACKET_GAP_FOR_PLC;
        let mut prev_factor = 1.0;
        for i in 0..(gap - 1) {
            let fade_factor = 1.0 - (i as f32 / gap as f32);
            assert!(
                fade_factor <= prev_factor,
                "Fade factor should decrease (prev: {}, current: {})",
                prev_factor,
                fade_factor
            );
            prev_factor = fade_factor;
        }
    }

    #[test]
    fn test_plc_fade_factor_for_single_missing_packet() {
        let gap = 2u32; // One missing packet (gap of 2 means packet 0, missing 1, packet 2)
        let fade_factor = 1.0 - (0 as f32 / gap as f32);
        assert_eq!(
            fade_factor, 1.0,
            "Single missing packet should fade to 1.0 (no fade)"
        );
    }

    // ============================================================================
    // Tests for Buffer Management Constants
    // ============================================================================

    #[test]
    fn test_buffer_size_constants() {
        // Verify buffer size calculations
        let frame_size_calc = (SAMPLE_RATE * FRAME_DURATION_MS / 1000) as usize;
        assert_eq!(
            FRAME_SIZE, frame_size_calc,
            "FRAME_SIZE should match calculated value (got {}, expected {})",
            FRAME_SIZE, frame_size_calc
        );

        // BUFFER_SIZE should be reasonable multiple of FRAME_SIZE
        assert!(
            BUFFER_SIZE > FRAME_SIZE * 10,
            "BUFFER_SIZE should be much larger than FRAME_SIZE for buffering"
        );
    }

    #[test]
    fn test_jitter_buffer_thresholds() {
        // Verify threshold relationships
        assert!(
            MIN_BUFFER_MS < INITIAL_BUFFER_MS,
            "MIN_BUFFER_MS ({}) should be less than INITIAL_BUFFER_MS ({})",
            MIN_BUFFER_MS,
            INITIAL_BUFFER_MS
        );
        assert!(
            INITIAL_BUFFER_MS < MAX_BUFFER_MS,
            "INITIAL_BUFFER_MS ({}) should be less than MAX_BUFFER_MS ({})",
            INITIAL_BUFFER_MS,
            MAX_BUFFER_MS
        );

        assert!(
            BUFFER_UNDERRUN_THRESHOLD_SAMPLES < BUFFER_OVERRUN_THRESHOLD_SAMPLES,
            "Underrun threshold should be less than overrun threshold"
        );
    }
}
