use anyhow::{Context, Result};
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
    widgets::{Block, Borders, List, ListItem, Paragraph},
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

const BUFFER_SIZE: usize = 19200;

#[derive(Clone, Debug, PartialEq)]
enum AppMode {
    Menu,
    Connecting,
    IceGathering,
    HostWaitingForAnswer { offer: String },
    ClientGeneratingAnswer { answer: String },
    IceConnecting,
    InCall(String),
}

enum AppAction {
    Log(String),
    #[allow(dead_code)]
    Status(String),
    SetMode(AppMode),
    Input(event::KeyEvent),
    SetOffer(String),
    SetAnswer(String),
    StoreHostAgent(IceAgentHandle),

    StartConnection(Arc<dyn webrtc_util::Conn + Send + Sync>),
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
    #[allow(dead_code)]
    status: String,
    input: String,
    should_quit: bool,
    local_ip: String,
    session_offer: Option<String>,
    session_answer: Option<String>,
    host_agent_handle: Option<IceAgentHandle>,
}

impl App {
    fn new() -> Self {
        let local_ip = local_ip_address::local_ip()
            .map(|ip| ip.to_string())
            .unwrap_or_else(|_| "Unknown".into());

        Self {
            mode: AppMode::Menu,
            logs: vec!["callrs - P2P Voice Chat with ICE".into()],
            status: "Ready".into(),
            input: String::new(),
            should_quit: false,
            local_ip,
            session_offer: None,
            session_answer: None,
            host_agent_handle: None,
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
    sample_rate: u32,
}

impl AudioHandle {
    fn start(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }
    fn stop(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }
}

fn setup_audio(capture_prod: HeapProd<f32>, playback_cons: HeapCons<f32>) -> Result<AudioHandle> {
    let host = cpal::default_host();
    let is_active = Arc::new(AtomicBool::new(false));

    let input_dev = host.default_input_device().context("No input device")?;
    let output_dev = host.default_output_device().context("No output device")?;

    let in_cfg = input_dev.default_input_config()?;
    let out_cfg = output_dev.default_output_config()?;

    let sample_rate: u32 = in_cfg.sample_rate();

    let in_channels = in_cfg.channels() as usize;
    let out_channels = out_cfg.channels() as usize;

    let active_in = is_active.clone();
    let mut capture_prod_f32 = capture_prod;
    let input_stream = match in_cfg.sample_format() {
        cpal::SampleFormat::F32 => input_dev.build_input_stream(
            &in_cfg.config(),
            move |data: &[f32], _| {
                if !active_in.load(Ordering::Relaxed) {
                    return;
                }
                for frame in data.chunks_exact(in_channels) {
                    let mono: f32 = frame.iter().sum::<f32>() / in_channels as f32;
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
    let output_stream = output_dev.build_output_stream(
        &out_cfg.config(),
        move |data: &mut [f32], _| {
            if !active_out.load(Ordering::Relaxed) {
                data.fill(0.0);
                return;
            }
            for frame in data.chunks_exact_mut(out_channels) {
                let sample = playback_cons_f32.try_pop().unwrap_or(0.0);
                frame.fill(sample);
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
        sample_rate,
    })
}

use webrtc_ice::{
    agent::{Agent, agent_config::AgentConfig},
    network_type::NetworkType,
    url::Url,
};

async fn create_ice_agent() -> Result<Agent> {
    let stun_url = Url::parse_url("stun:stun.l.google.com:19302")?;

    let config = AgentConfig {
        network_types: vec![NetworkType::Udp4, NetworkType::Udp6],
        urls: vec![stun_url],
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
) {
    let frame_size = ((sample_rate as f32) * 0.02) as usize;
    let mut frame_buf = Vec::with_capacity(frame_size);
    let mut sequence: u32 = 0;

    loop {
        if !is_active.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_millis(50)).await;
            continue;
        }
        while let Some(sample) = capture_cons.try_pop() {
            let s_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            frame_buf.push(s_i16);
            if frame_buf.len() >= frame_size {
                let mut packet = Vec::with_capacity(4 + frame_size * 2);
                packet.extend_from_slice(&sequence.to_le_bytes());
                for &s in &frame_buf {
                    packet.extend_from_slice(&s.to_le_bytes());
                }
                let _ = network_tx.try_send(packet);
                frame_buf.clear();
                sequence = sequence.wrapping_add(1);
            }
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn run_network(
    conn: Arc<dyn webrtc_util::Conn + Send + Sync>,
    mut network_rx: mpsc::Receiver<Vec<u8>>,
    mut playback_prod: HeapProd<f32>,
    action_tx: mpsc::Sender<AppAction>,
    _is_active: Arc<AtomicBool>,
) {
    let mut buf = vec![0u8; 8192];

    let _ = action_tx
        .send(AppAction::Log("ICE Connected!".into()))
        .await;

    loop {
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
                            let samples = buf[4..len]
                                .chunks_exact(2)
                                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0);
                            // Basic buffer overflow protection (clock drift compensation)
                            if playback_prod.occupied_len() > 9600 {
                                continue;
                            }

                            for s in samples {
                                let _ = playback_prod.try_push(s);
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
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(5),
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
        AppMode::InCall(peer) => format!("IN CALL: {}", peer),
    };

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " CALLRS-ICE ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" | "),
            Span::styled(status, Style::default().fg(Color::Yellow)),
        ]))
        .block(Block::default().borders(Borders::ALL)),
        chunks[0],
    );

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

    let (display_text, title) = match &app.mode {
        AppMode::HostWaitingForAnswer { offer } => (
            offer.as_str(),
            "Your Offer (send to client) | Paste Answer Below:",
        ),
        AppMode::ClientGeneratingAnswer { answer } => {
            (answer.as_str(), "Your Answer (send to host)")
        }
        _ => (app.input.as_str(), "Input / Session Data"),
    };

    f.render_widget(
        Paragraph::new(display_text).block(Block::default().title(title).borders(Borders::ALL)),
        chunks[2],
    );

    let help = " [H] Host | [C] Connect | [ESC] End | [Q] Quit ";
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
            if let Ok(true) = event::poll(Duration::from_millis(50))
                && let Ok(Event::Key(key)) = event::read()
            {
                let _ = input_tx.blocking_send(AppAction::Input(key));
            }
        }
    });

    let (mut net_task, mut cap_task, mut audio_handle): (
        Option<tokio::task::JoinHandle<()>>,
        Option<tokio::task::JoinHandle<()>>,
        Option<AudioHandle>,
    ) = (None, None, None);

    loop {
        terminal.draw(|f| render_ui(f, &app))?;
        if let Some(action) = tokio::select! { a = action_rx.recv() => a, _ = tokio::time::sleep(Duration::from_millis(50)) => None }
        {
            match action {
                AppAction::Log(m) => app.add_log(m),
                AppAction::SetMode(m) => app.mode = m,
                AppAction::SetOffer(offer) => {
                    app.session_offer = Some(offer);
                }
                AppAction::SetAnswer(answer) => {
                    app.session_answer = Some(answer);
                }
                AppAction::StoreHostAgent(handle) => {
                    app.host_agent_handle = Some(handle);
                }

                AppAction::StartConnection(conn) => {
                    // Setup audio and network
                    let (ntx, nrx) = mpsc::channel(100);
                    let (cap_prod, cap_cons) = HeapRb::<f32>::new(BUFFER_SIZE).split();
                    let (play_prod, play_cons) = HeapRb::<f32>::new(BUFFER_SIZE).split();

                    if let Ok(audio) = setup_audio(cap_prod, play_cons) {
                        audio.start();

                        // Store the handles!
                        cap_task = Some(tokio::spawn(capture_processor(
                            cap_cons,
                            ntx,
                            audio.is_active.clone(),
                            audio.sample_rate,
                        )));

                        net_task = Some(tokio::spawn(run_network(
                            conn,
                            nrx,
                            play_prod,
                            action_tx.clone(),
                            audio.is_active.clone(),
                        )));

                        audio_handle = Some(audio);

                        app.mode = AppMode::InCall("Connected".into());
                    }
                }
                AppAction::Input(key) => {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }
                    match (&app.mode, key.code) {
                        (_, KeyCode::Char('q')) => app.should_quit = true,

                        // Host flow: Press 'h' to start hosting
                        (AppMode::Menu, KeyCode::Char('h')) => {
                            app.mode = AppMode::IceGathering;
                            let tx = action_tx.clone();
                            tokio::spawn(async move {
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
                                        let _ =
                                            tx.send(AppAction::Log(format!("Error: {}", e))).await;
                                        let _ = tx.send(AppAction::SetMode(AppMode::Menu)).await;
                                    }
                                }
                            });
                        }

                        // Host receives answer and completes connection
                        (AppMode::HostWaitingForAnswer { .. }, KeyCode::Enter) => {
                            if !app.input.is_empty() && app.host_agent_handle.is_some() {
                                let answer_b64 = app.input.clone();
                                let agent_handle = app.host_agent_handle.take().unwrap();
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
                                                    let _ = tx
                                                        .send(AppAction::Log(format!(
                                                            "Connected via: {}",
                                                            pair
                                                        )))
                                                        .await;
                                                }

                                                let _ =
                                                    tx.send(AppAction::StartConnection(conn)).await;
                                            }
                                            Err(e) => {
                                                let _ = tx
                                                    .send(AppAction::Log(format!(
                                                        "Accept error: {}",
                                                        e
                                                    )))
                                                    .await;
                                                let _ = tx
                                                    .send(AppAction::SetMode(AppMode::Menu))
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
                                                        let _ = tx
                                                            .send(AppAction::Log(format!(
                                                                "Connected via: {}",
                                                                pair
                                                            )))
                                                            .await;
                                                    }

                                                    let _ = tx
                                                        .send(AppAction::StartConnection(conn))
                                                        .await;
                                                }
                                                Err(e) => {
                                                    let _ = tx
                                                        .send(AppAction::Log(format!(
                                                            "Dial error: {}",
                                                            e
                                                        )))
                                                        .await;
                                                    let _ = tx
                                                        .send(AppAction::SetMode(AppMode::Menu))
                                                        .await;
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            let _ = tx
                                                .send(AppAction::Log(format!("Error: {}", e)))
                                                .await;
                                            let _ =
                                                tx.send(AppAction::SetMode(AppMode::Menu)).await;
                                        }
                                    }
                                });
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

                        // ESC to cancel/end
                        (_, KeyCode::Esc) => {
                            if let Some(t) = net_task.take() {
                                t.abort();
                            }
                            if let Some(t) = cap_task.take() {
                                t.abort();
                            }
                            if let Some(a) = audio_handle.take() {
                                a.stop();
                            }
                            app.host_agent_handle = None;

                            app.mode = AppMode::Menu;
                            app.session_offer = None;
                            app.session_answer = None;
                            app.input.clear();
                        }
                        _ => {}
                    }
                }
                _ => {}
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
