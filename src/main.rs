use anyhow::{Context, Result};
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
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use tokio::{net::UdpSocket, sync::mpsc};

const BUFFER_SIZE: usize = 19200;
const STUN_SERVER: &str = "stun.l.google.com:19302";

#[derive(Clone, Debug, PartialEq)]
enum AppMode {
    Menu,
    Hosting(u16),
    Connecting,
    IceGathering,
    InCall(SocketAddr),
}

enum AppAction {
    Log(String),
    #[allow(dead_code)]
    Status(String),
    SetMode(AppMode),
    Input(event::KeyEvent),
    SetIceData {
        candidates: Vec<IceCandidate>,
        offer: String,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct IceCandidate {
    address: String,
    port: u16,
    candidate_type: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SessionDescription {
    candidates: Vec<IceCandidate>,
}

struct App {
    mode: AppMode,
    logs: Vec<String>,
    #[allow(dead_code)]
    status: String,
    input: String,
    should_quit: bool,
    local_ip: String,
    ice_candidates: Vec<IceCandidate>,
    session_offer: Option<String>,
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
            ice_candidates: Vec::new(),
            session_offer: None,
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

async fn gather_ice_candidates(
    local_port: u16,
    action_tx: mpsc::Sender<AppAction>,
) -> Result<Vec<IceCandidate>> {
    let mut candidates = Vec::new();
    let local_ip = local_ip_address::local_ip()?.to_string();

    candidates.push(IceCandidate {
        address: local_ip.clone(),
        port: local_port,
        candidate_type: "host".to_string(),
    });

    if let Ok(public_addr) = discover_public_address(local_port).await {
        candidates.push(IceCandidate {
            address: public_addr.ip().to_string(),
            port: public_addr.port(),
            candidate_type: "srflx".to_string(),
        });
        let _ = action_tx
            .send(AppAction::Log(format!("Found public IP: {}", public_addr)))
            .await;
    }

    Ok(candidates)
}

async fn discover_public_address(local_port: u16) -> Result<SocketAddr> {
    let socket = UdpSocket::bind(format!("0.0.0.0:{}", local_port)).await?;
    let transaction_id: [u8; 12] = rand::random();
    let mut stun_request = vec![0x00, 0x01, 0x00, 0x00, 0x21, 0x12, 0xa4, 0x42];
    stun_request.extend_from_slice(&transaction_id);

    socket.send_to(&stun_request, STUN_SERVER).await?;
    let mut buf = vec![0u8; 1024];
    let (len, _) =
        tokio::time::timeout(Duration::from_secs(3), socket.recv_from(&mut buf)).await??;

    parse_stun_response(&buf[..len], &transaction_id).context("STUN parse error")
}

fn parse_stun_response(data: &[u8], expected_tid: &[u8; 12]) -> Option<SocketAddr> {
    if data.len() < 20 || &data[8..20] != expected_tid {
        return None;
    }
    let mut pos = 20;
    while pos + 4 <= data.len() {
        let attr_type = u16::from_be_bytes([data[pos], data[pos + 1]]);
        let attr_len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if attr_type == 0x0020 && attr_len >= 8 {
            let port = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) ^ 0x2112;
            let ip = u32::from_be_bytes([
                data[pos + 4] ^ 0x21,
                data[pos + 5] ^ 0x12,
                data[pos + 6] ^ 0xa4,
                data[pos + 7] ^ 0x42,
            ]);
            return Some(SocketAddr::new(std::net::Ipv4Addr::from(ip).into(), port));
        }
        pos = (pos + attr_len + 3) & !3;
    }
    None
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

async fn try_connect_with_candidates(
    socket: Arc<UdpSocket>,
    remote_candidates: Vec<IceCandidate>,
    action_tx: mpsc::Sender<AppAction>,
) -> Option<SocketAddr> {
    for candidate in &remote_candidates {
        if let Ok(addr) = format!("{}:{}", candidate.address, candidate.port).parse::<SocketAddr>()
        {
            let _ = action_tx
                .send(AppAction::Log(format!("ICE: Pinging {}", addr)))
                .await;
            for _ in 0..3 {
                let _ = socket.send_to(b"PING", addr).await;
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
    None
}

async fn run_network(
    socket: Arc<UdpSocket>,
    mut peer: Option<SocketAddr>,
    mut network_rx: mpsc::Receiver<Vec<u8>>,
    mut playback_prod: HeapProd<f32>,
    action_tx: mpsc::Sender<AppAction>,
    is_active: Arc<AtomicBool>,
) {
    let mut buf = vec![0u8; 8192];
    loop {
        tokio::select! {
            Some(data) = network_rx.recv() => {
                if let Some(addr) = peer { let _ = socket.send_to(&data, addr).await; }
            }
            res = socket.recv_from(&mut buf) => {
                if let Ok((len, addr)) = res {
                    if len == 4 && (&buf[..4] == b"PING" || &buf[..4] == b"PONG") {
                        if &buf[..4] == b"PING" { let _ = socket.send_to(b"PONG", addr).await; }
                        if peer.is_none() {
                            peer = Some(addr);
                            let _ = action_tx.send(AppAction::Log(format!("ICE Connected: {}", addr))).await;
                            let _ = action_tx.send(AppAction::SetMode(AppMode::InCall(addr))).await;
                            is_active.store(true, Ordering::Relaxed);
                        }
                        continue;
                    }
                    if Some(addr) == peer {
                        let samples = buf[4..len].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0);
                        for s in samples { let _ = playback_prod.try_push(s); }
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
        AppMode::Hosting(p) => format!("HOSTING | Port: {}", p),
        AppMode::IceGathering => "GATHERING ICE...".into(),
        AppMode::Connecting => "PASTE OFFER...".into(),
        AppMode::InCall(addr) => format!("IN CALL: {}", addr),
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

    let input_text = app.session_offer.as_ref().unwrap_or(&app.input);
    f.render_widget(
        Paragraph::new(input_text.as_str()).block(
            Block::default()
                .title("ICE Session Data")
                .borders(Borders::ALL),
        ),
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
                && let Ok(Event::Key(key)) = event::read() {
                    let _ = input_tx.blocking_send(AppAction::Input(key));
                }
        }
    });

    let (mut net_task, mut cap_task, mut audio_handle) = (None, None, None);

    loop {
        terminal.draw(|f| render_ui(f, &app))?;
        if let Some(action) = tokio::select! { a = action_rx.recv() => a, _ = tokio::time::sleep(Duration::from_millis(50)) => None }
        {
            match action {
                AppAction::Log(m) => app.add_log(m),
                AppAction::SetMode(m) => app.mode = m,
                AppAction::SetIceData { candidates, offer } => {
                    app.ice_candidates = candidates;
                    app.session_offer = Some(offer);
                }
                AppAction::Input(key) => {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }
                    match (&app.mode, key.code) {
                        (_, KeyCode::Char('q')) => app.should_quit = true,
                        (AppMode::Menu, KeyCode::Char('h')) => {
                            app.mode = AppMode::IceGathering;
                            let tx = action_tx.clone();
                            tokio::spawn(async move {
                                if let Ok(candidates) =
                                    gather_ice_candidates(9090, tx.clone()).await
                                {
                                    let offer = base64::encode(
                                        &serde_json::to_string(&SessionDescription {
                                            candidates: candidates.clone(),
                                        })
                                        .unwrap(),
                                    );
                                    let _ =
                                        tx.send(AppAction::SetIceData { candidates, offer }).await;
                                    let _ =
                                        tx.send(AppAction::SetMode(AppMode::Hosting(9090))).await;
                                }
                            });
                            let socket = Arc::new(UdpSocket::bind("0.0.0.0:9090").await?);
                            let (ntx, nrx) = mpsc::channel(100);
                            let (cap_prod, cap_cons) = HeapRb::<f32>::new(BUFFER_SIZE).split();
                            let (play_prod, play_cons) = HeapRb::<f32>::new(BUFFER_SIZE).split();
                            if let Ok(audio) = setup_audio(cap_prod, play_cons) {
                                cap_task = Some(tokio::spawn(capture_processor(
                                    cap_cons,
                                    ntx,
                                    audio.is_active.clone(),
                                    audio.sample_rate,
                                )));
                                net_task = Some(tokio::spawn(run_network(
                                    socket,
                                    None,
                                    nrx,
                                    play_prod,
                                    action_tx.clone(),
                                    audio.is_active.clone(),
                                )));
                                audio_handle = Some(audio);
                            }
                        }
                        (AppMode::Menu, KeyCode::Char('c')) => {
                            app.mode = AppMode::Connecting;
                            app.input.clear();
                        }
                        (AppMode::Connecting, KeyCode::Enter) => {
                            if let Ok(decoded) = base64::decode(&app.input)
                                && let Ok(session) =
                                    serde_json::from_slice::<SessionDescription>(&decoded)
                                {
                                    let socket = Arc::new(UdpSocket::bind("0.0.0.0:0").await?);
                                    let (ntx, nrx) = mpsc::channel(100);
                                    let (cap_p, cap_c) = HeapRb::<f32>::new(BUFFER_SIZE).split();
                                    let (play_p, play_c) = HeapRb::<f32>::new(BUFFER_SIZE).split();
                                    if let Ok(audio) = setup_audio(cap_p, play_c) {
                                        audio.start();
                                        tokio::spawn(try_connect_with_candidates(
                                            socket.clone(),
                                            session.candidates,
                                            action_tx.clone(),
                                        ));
                                        cap_task = Some(tokio::spawn(capture_processor(
                                            cap_c,
                                            ntx,
                                            audio.is_active.clone(),
                                            audio.sample_rate,
                                        )));
                                        net_task = Some(tokio::spawn(run_network(
                                            socket,
                                            None,
                                            nrx,
                                            play_p,
                                            action_tx.clone(),
                                            audio.is_active.clone(),
                                        )));
                                        audio_handle = Some(audio);
                                        app.mode = AppMode::IceGathering;
                                    }
                                }
                        }
                        (AppMode::Connecting, KeyCode::Char(c)) => app.input.push(c),
                        (AppMode::Connecting, KeyCode::Backspace) => {
                            app.input.pop();
                        }
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
                            app.mode = AppMode::Menu;
                            app.session_offer = None;
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

mod base64 {
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    pub fn encode(data: &str) -> String {
        let mut res = String::new();
        for chunk in data.as_bytes().chunks(3) {
            let b = [
                chunk[0],
                *chunk.get(1).unwrap_or(&0),
                *chunk.get(2).unwrap_or(&0),
            ];
            res.push(CHARSET[(b[0] >> 2) as usize] as char);
            res.push(CHARSET[(((b[0] & 0x03) << 4) | (b[1] >> 4)) as usize] as char);
            res.push(if chunk.len() > 1 {
                CHARSET[(((b[1] & 0x0f) << 2) | (b[2] >> 6)) as usize] as char
            } else {
                '='
            });
            res.push(if chunk.len() > 2 {
                CHARSET[(b[2] & 0x3f) as usize] as char
            } else {
                '='
            });
        }
        res
    }

    // FIXED: Corrected type mismatch and Option handling
    pub fn decode(data: &str) -> Result<Vec<u8>, ()> {
        let mut res = Vec::new();
        let bytes: Vec<u8> = data.bytes().filter(|&b| b != b'=').collect();
        for chunk in bytes.chunks(4) {
            if chunk.len() < 2 {
                return Err(());
            }
            let c1 = d(chunk[0])?;
            let c2 = d(chunk[1])?;
            res.push((c1 << 2) | (c2 >> 4));
            if let Some(&b3) = chunk.get(2) {
                let c3 = d(b3)?;
                res.push((c2 << 4) | (c3 >> 2));
                if let Some(&b4) = chunk.get(3) {
                    let c4 = d(b4)?;
                    res.push((c3 << 6) | c4);
                }
            }
        }
        Ok(res)
    }
    fn d(c: u8) -> Result<u8, ()> {
        match c {
            b'A'..=b'Z' => Ok(c - b'A'),
            b'a'..=b'z' => Ok(c - b'a' + 26),
            b'0'..=b'9' => Ok(c - b'0' + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(()),
        }
    }
}
