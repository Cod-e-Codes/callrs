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

const SAMPLE_RATE: u32 = 48000; // API requires 48k for best Opus compatibility
const FRAME_DURATION_MS: u32 = 20;
const FRAME_SIZE: usize = (SAMPLE_RATE * FRAME_DURATION_MS / 1000) as usize; // 960 samples

const BUFFER_SIZE: usize = 19200;

// Noise gate threshold: RMS amplitude below this value will be considered silence
// Value is in normalized range [0.0, 1.0] where 1.0 is full scale
const NOISE_GATE_THRESHOLD: f32 = 0.01; // ~1% of full scale, adjustable

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

    StartConnection(Arc<dyn webrtc_util::Conn + Send + Sync>),
    SetMicVolume(f32),
    SetReceiveVolume(f32),
    SetConnectionType(String),
    ToggleMute,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SessionOffer {
    ufrag: String,
    pwd: String,
    candidates: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
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
            connection_type: "Unknown".into(),
            is_muted: false,
        }
    }

    fn add_log(&mut self, msg: String) {
        self.logs.push(msg);
        if self.logs.len() > 100 {
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
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: rubato::SincInterpolationType::Linear,
                window: rubato::WindowFunction::BlackmanHarris2,
                oversampling_factor: 128,
            };
            // Resample FROM 48kHz (buffer) TO output_sample_rate (device)
            // ratio = output_rate / input_rate = output_sample_rate / 48000.0
            let ratio = output_sample_rate as f64 / SAMPLE_RATE as f64;
            // Input chunk size: approximate 20ms at 48kHz = ~960 samples
            let chunk_in = 960;
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
        3600, // 1 hour lease
        "CallRS Voice App",
    ) {
        Ok(_) => Ok(format!("UPnP Success! Port {} mapped.", local_port)),
        Err(e) => Err(anyhow::anyhow!("UPnP Mapping Failed: {}", e)),
    }
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

    // Create offer
    let offer = SessionOffer {
        ufrag: ufrag.clone(),
        pwd: pwd.clone(),
        candidates: candidates.iter().map(|c| c.marshal()).collect(),
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
    })
}

async fn capture_processor(
    mut capture_cons: HeapCons<f32>,
    network_tx: mpsc::Sender<Vec<u8>>,
    is_active: Arc<AtomicBool>,
    sample_rate: u32,
    action_tx: mpsc::Sender<AppAction>,
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

    // Resampler setup
    let mut resampler: Option<SincFixedIn<f32>> = if sample_rate != 48000 {
        let params = rubato::SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: rubato::SincInterpolationType::Linear,
            window: rubato::WindowFunction::BlackmanHarris2,
            oversampling_factor: 128,
        };
        // Input chunk size: try to approximate 20ms or just use something reasonable like 1024
        // output_needed ~ 960. ratio = 48000 / sample_rate.
        let chunk_in = 1024;
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

    // For volume calculation
    let mut vol_sum = 0.0;
    let mut vol_count = 0;
    let mut last_vol_update = std::time::Instant::now();

    loop {
        if !is_active.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_millis(50)).await;
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
                        let _ =
                            action_tx.try_send(AppAction::Log(format!("Resample error: {:?}", e)));
                    }
                }
            }
        }

        // 3. Process F32 buffer into Opus Frames
        while f32_capture_buffer.len() >= frame_size {
            // Take exactly frame_size
            let frame_samples: Vec<f32> = f32_capture_buffer.drain(0..frame_size).collect();

            // Calculate RMS (Root Mean Square) for noise gate
            let rms = (frame_samples.iter().map(|s| s * s).sum::<f32>() / frame_size as f32).sqrt();

            // Noise gate: skip encoding if below threshold
            if rms < NOISE_GATE_THRESHOLD {
                // Still update volume display to show silence
                vol_sum = 0.0;
                vol_count = frame_size;
                if last_vol_update.elapsed().as_millis() > 50 {
                    let _ = action_tx.try_send(AppAction::SetMicVolume(0.0));
                    last_vol_update = std::time::Instant::now();
                }
                continue; // Skip encoding and sending this frame
            }

            // Calc volume & Convert to i16
            for sample in frame_samples {
                vol_sum += sample.abs();
                vol_count += 1;

                let s_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                pcm_buf.push(s_i16);
            }

            // Volume Update
            if vol_count >= frame_size {
                let avg = vol_sum / vol_count as f32;
                let vol_normalized = (avg * 5.0).clamp(0.0, 1.0);
                if last_vol_update.elapsed().as_millis() > 50 {
                    let _ = action_tx.try_send(AppAction::SetMicVolume(vol_normalized));
                    last_vol_update = std::time::Instant::now();
                }
                vol_sum = 0.0;
                vol_count = 0;
            }

            // Encode
            match encoder.encode(&pcm_buf, &mut opus_buf) {
                Ok(len) => {
                    let mut packet = Vec::with_capacity(4 + len);
                    packet.extend_from_slice(&sequence.to_le_bytes());
                    packet.extend_from_slice(&opus_buf[..len]);

                    match network_tx.try_send(packet) {
                        Ok(_) => {}
                        Err(_) => {
                            dropped_packets += 1;
                        }
                    }
                    sequence = sequence.wrapping_add(1);
                }
                Err(e) => {
                    let _ =
                        action_tx.try_send(AppAction::Log(format!("Opus encode error: {:?}", e)));
                }
            }
            pcm_buf.clear();
        }

        // Log drops
        if dropped_packets > 0 && last_drop_log.elapsed().as_secs() >= 5 {
            let _ = action_tx.try_send(AppAction::Log(format!(
                "Warning: Dropped {} TX packets (backpressure)",
                dropped_packets
            )));
            dropped_packets = 0;
            last_drop_log = std::time::Instant::now();
        }

        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

async fn run_network(
    conn: Arc<dyn webrtc_util::Conn + Send + Sync>,
    mut network_rx: mpsc::Receiver<Vec<u8>>,
    mut playback_prod: HeapProd<f32>,
    action_tx: mpsc::Sender<AppAction>,
    _is_active: Arc<AtomicBool>,
    _sample_rate: u32,
) {
    let mut buf = vec![0u8; 8192];
    let mut buffering = true;

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

    // Packet loss concealment: track expected sequence number and last frame
    let mut expected_sequence: Option<u32> = None;
    let mut last_decoded_frame: Option<Vec<f32>> = None;

    loop {
        // Log drops
        if dropped_frames > 0 && last_drop_log.elapsed().as_secs() >= 5 {
            let _ = action_tx.try_send(AppAction::Log(format!(
                "Warning: Dropped {} RX frames (playback buffer full)",
                dropped_frames
            )));
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
                        if len > 4 {
                            // Extract sequence number
                            let received_sequence = u32::from_le_bytes([
                                buf[0], buf[1], buf[2], buf[3]
                            ]);

                            // Opus payload (skip 4 byte sequence)
                            let opus_payload = &buf[4..len];

                            // Packet Loss Concealment: detect missing packets
                            if let Some(expected) = expected_sequence {
                                let gap = received_sequence.wrapping_sub(expected);
                                // If gap > 0, we have missing packets (handle up to 10 missing packets)
                                if gap > 0 && gap <= 10 {
                                    // Perform PLC: repeat last frame with fade-out for missing packets
                                    if let Some(ref last_frame) = last_decoded_frame {
                                        for i in 0..(gap - 1) {
                                            let fade_factor = 1.0 - (i as f32 / gap as f32);
                                            if playback_prod.occupied_len() > 9600 {
                                                dropped_frames += last_frame.len() as u64;
                                                continue;
                                            }
                                            // Calculate volume for PLC frames (faded)
                                            let mut plc_vol_sum = 0.0;
                                            for &sample in last_frame.iter() {
                                                let faded = sample * fade_factor;
                                                plc_vol_sum += faded.abs();
                                                if playback_prod.try_push(faded).is_err() {
                                                    dropped_frames += 1;
                                                }
                                            }
                                            // Update receive volume for PLC frames
                                            let plc_vol = (plc_vol_sum / last_frame.len() as f32 * 5.0).clamp(0.0, 1.0);
                                            let _ = action_tx.try_send(AppAction::SetReceiveVolume(plc_vol));
                                        }
                                    } else {
                                        // No previous frame, insert silence (zero volume)
                                        let _ = action_tx.try_send(AppAction::SetReceiveVolume(0.0));
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
                            expected_sequence = Some(received_sequence.wrapping_add(1));

                            match decoder.decode(Some(opus_payload), &mut decoded_pcm, false) {
                                Ok(samples_len) => {
                                    // Initialize jitter buffer logic (60ms pre-fill)
                                    if buffering {
                                        let jitter_samples = FRAME_SIZE * 3;
                                        for _ in 0..jitter_samples {
                                            let _ = playback_prod.try_push(0.0);
                                        }
                                        buffering = false;
                                        let _ = action_tx.send(AppAction::Log("Jitter buffer primed (60ms)".into())).await;
                                    }

                                    // Basic buffer overflow protection
                                    if playback_prod.occupied_len() > 9600 {
                                        dropped_frames += samples_len as u64;
                                        continue;
                                    }

                                    // Store decoded frame for PLC
                                    let frame_f32: Vec<f32> = decoded_pcm[..samples_len]
                                        .iter()
                                        .map(|&s| s as f32 / 32768.0)
                                        .collect();

                                    // Calculate receive volume (RMS)
                                    let receive_rms = (frame_f32.iter().map(|s| s * s).sum::<f32>() / frame_f32.len() as f32).sqrt();
                                    let receive_vol_normalized = (receive_rms * 5.0).clamp(0.0, 1.0);
                                    let _ = action_tx.try_send(AppAction::SetReceiveVolume(receive_vol_normalized));

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
            .constraints([Constraint::Min(3), Constraint::Length(3)])
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

        // Show both volume meters side by side
        let gauge_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(inner_chunks[1]);
        f.render_widget(mic_gauge, gauge_chunks[0]);
        f.render_widget(receive_gauge, gauge_chunks[1]);
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
            // Display original string - wrapping will handle display, copying will get clean text
            (
                offer.clone(),
                "Your Offer (send to client) | Paste Answer Below:",
            )
        }
        AppMode::ClientGeneratingAnswer { answer } => {
            (answer.clone(), "Your Answer (send to host)")
        }
        AppMode::Error(msg) => (msg.clone(), "Error (press ESC to dismiss)"),
        _ => (app.input.clone(), "Input / Session Data"),
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
                        app.input.push_str(&text);
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

                AppAction::StartConnection(conn) => {
                    // Setup audio and network
                    let (ntx, nrx) = mpsc::channel(100);
                    let (cap_prod, cap_cons) = HeapRb::<f32>::new(BUFFER_SIZE).split();
                    let (play_prod, play_cons) = HeapRb::<f32>::new(BUFFER_SIZE).split();

                    if let Ok(audio) = setup_audio(cap_prod, play_cons) {
                        audio.start();

                        let cap_task = tokio::spawn(capture_processor(
                            cap_cons,
                            ntx,
                            audio.is_active.clone(),
                            audio.input_sample_rate,
                            action_tx.clone(),
                        ));

                        let net_task = tokio::spawn(run_network(
                            conn,
                            nrx,
                            play_prod,
                            action_tx.clone(),
                            audio.is_active.clone(),
                            audio.output_sample_rate,
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
                                let answer_b64 = app.input.clone();
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

                                                let _ =
                                                    tx.send(AppAction::StartConnection(conn)).await;
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

                                                    let _ = tx
                                                        .send(AppAction::StartConnection(conn))
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
            break;
        }
    }
    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen)?;
    Ok(())
}
