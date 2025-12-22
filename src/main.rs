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

#[derive(Clone, Debug, PartialEq)]
enum AppMode {
    Menu,
    Hosting(u16),
    Connecting,
    InCall(SocketAddr),
}

enum AppAction {
    Log(String),
    #[allow(dead_code)]
    Status(String),
    SetMode(AppMode),
    Input(event::KeyEvent),
}

struct App {
    mode: AppMode,
    logs: Vec<String>,
    #[allow(dead_code)]
    status: String,
    input: String,
    should_quit: bool,
    local_ip: String,
}

impl App {
    fn new() -> Self {
        let local_ip = local_ip_address::local_ip()
            .map(|ip| ip.to_string())
            .unwrap_or_else(|_| "Unknown".into());

        Self {
            mode: AppMode::Menu,
            logs: vec!["callrs - P2P Voice Chat".into()],
            status: "Ready".into(),
            input: String::new(),
            should_quit: false,
            local_ip,
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
        cpal::SampleFormat::I16 => input_dev.build_input_stream(
            &in_cfg.config(),
            move |data: &[i16], _| {
                if !active_in.load(Ordering::Relaxed) {
                    return;
                }
                for frame in data.chunks_exact(in_channels) {
                    let mono_i16: i32 = frame.iter().map(|&s| s as i32).sum();
                    let mono_f32 = (mono_i16 as f32 / in_channels as f32) / 32768.0;
                    let _ = capture_prod_f32.try_push(mono_f32);
                }
            },
            |_| {},
            None,
        )?,
        _ => return Err(anyhow::anyhow!("Unsupported input sample format")),
    };

    let active_out = is_active.clone();
    let mut playback_cons_f32 = playback_cons;
    let output_stream = match out_cfg.sample_format() {
        cpal::SampleFormat::F32 => output_dev.build_output_stream(
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
        )?,
        cpal::SampleFormat::I16 => output_dev.build_output_stream(
            &out_cfg.config(),
            move |data: &mut [i16], _| {
                if !active_out.load(Ordering::Relaxed) {
                    data.fill(0);
                    return;
                }
                for frame in data.chunks_exact_mut(out_channels) {
                    let sample = (playback_cons_f32.try_pop().unwrap_or(0.0) * 32767.0) as i16;
                    frame.fill(sample);
                }
            },
            |_| {},
            None,
        )?,
        _ => return Err(anyhow::anyhow!("Unsupported output sample format")),
    };

    input_stream.play()?;
    output_stream.play()?;

    Ok(AudioHandle {
        _streams: (input_stream, output_stream),
        is_active,
        sample_rate,
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
                if let Some(addr) = peer {
                    let _ = socket.send_to(&data, addr).await;
                }
            }
            res = socket.recv_from(&mut buf) => {
                if let Ok((len, addr)) = res {
                    if len < 4 { continue; }
                    if peer.is_none() {
                        peer = Some(addr);
                        let _ = action_tx.send(AppAction::Log(format!("Peer connected: {addr}"))).await;
                        let _ = action_tx.send(AppAction::SetMode(AppMode::InCall(addr))).await;
                        is_active.store(true, Ordering::Relaxed);
                    }
                    if Some(addr) == peer {
                        let samples = buf[4..len].chunks_exact(2)
                            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0);
                        for s in samples {
                            let _ = playback_prod.try_push(s);
                        }
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
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(f.area());

    let status_text = match &app.mode {
        AppMode::Menu => format!("IDLE | Your IP: {}", app.local_ip),
        AppMode::Hosting(p) => format!("HOSTING | Share: {}:{}", app.local_ip, p),
        AppMode::Connecting => "ENTERING ADDRESS...".to_string(),
        AppMode::InCall(addr) => format!("IN CALL WITH {}", addr),
    };

    let header = Paragraph::new(Line::from(vec![
        Span::styled(
            " CALLRS ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" | "),
        Span::styled(status_text, Style::default().fg(Color::Yellow)),
    ]))
    .block(Block::default().borders(Borders::ALL));

    f.render_widget(header, chunks[0]);

    let logs: Vec<ListItem> = app
        .logs
        .iter()
        .rev()
        .take(chunks[1].height.saturating_sub(2) as usize)
        .rev()
        .map(|l| ListItem::new(l.as_str()))
        .collect();
    f.render_widget(
        List::new(logs).block(Block::default().title("Activity").borders(Borders::ALL)),
        chunks[1],
    );

    let input_block = Paragraph::new(app.input.as_str()).block(
        Block::default()
            .title("Target Address (IP:Port)")
            .borders(Borders::ALL),
    );
    f.render_widget(input_block, chunks[2]);

    // RESTORED: Context-sensitive help bar
    let help_text = match app.mode {
        AppMode::Menu => " [H] Host | [C] Connect | [Q] Quit ",
        AppMode::Connecting => " [ENTER] Connect | [ESC] Cancel ",
        AppMode::Hosting(_) | AppMode::InCall(_) => " [ESC] Hang Up | [Q] Quit ",
    };

    f.render_widget(
        Paragraph::new(help_text).style(Style::default().fg(Color::DarkGray)),
        chunks[3],
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    enable_raw_mode()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    execute!(io::stdout(), EnterAlternateScreen)?;

    let mut app = App::new();
    let (action_tx, mut action_rx) = mpsc::channel(64);

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

    let mut network_task: Option<tokio::task::JoinHandle<()>> = None;
    let mut capture_task: Option<tokio::task::JoinHandle<()>> = None;
    let mut audio_handle: Option<AudioHandle> = None;

    loop {
        terminal.draw(|f| render_ui(f, &app))?;

        let action = tokio::select! {
            a = action_rx.recv() => a,
            _ = tokio::time::sleep(Duration::from_millis(50)) => None,
        };

        if let Some(action) = action {
            match action {
                AppAction::Log(m) => app.add_log(m),
                AppAction::Status(s) => app.status = s,
                AppAction::SetMode(m) => app.mode = m,
                AppAction::Input(key) => {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }
                    match (&app.mode, key.code) {
                        // Global Quit
                        (_, KeyCode::Char('q')) if !matches!(app.mode, AppMode::Connecting) => {
                            app.should_quit = true
                        }

                        (AppMode::Menu, KeyCode::Char('h')) => {
                            let socket = Arc::new(UdpSocket::bind("0.0.0.0:9090").await?);
                            let (ntx, nrx) = mpsc::channel(100);
                            let (cap_prod, cap_cons) = HeapRb::<f32>::new(BUFFER_SIZE).split();
                            let (play_prod, play_cons) = HeapRb::<f32>::new(BUFFER_SIZE).split();

                            if let Ok(audio) = setup_audio(cap_prod, play_cons) {
                                let sr = audio.sample_rate;
                                let active = audio.is_active.clone();
                                capture_task = Some(tokio::spawn(capture_processor(
                                    cap_cons,
                                    ntx,
                                    active.clone(),
                                    sr,
                                )));
                                network_task = Some(tokio::spawn(run_network(
                                    socket,
                                    None,
                                    nrx,
                                    play_prod,
                                    action_tx.clone(),
                                    active,
                                )));
                                audio_handle = Some(audio);
                                app.mode = AppMode::Hosting(9090);
                            }
                        }
                        (AppMode::Menu, KeyCode::Char('c')) => {
                            app.mode = AppMode::Connecting;
                            app.input.clear();
                        }
                        (AppMode::Connecting, KeyCode::Enter) => {
                            if let Ok(addr) = app.input.parse::<SocketAddr>() {
                                let socket = Arc::new(UdpSocket::bind("0.0.0.0:0").await?);
                                let (ntx, nrx) = mpsc::channel(100);
                                let (cap_prod, cap_cons) = HeapRb::<f32>::new(BUFFER_SIZE).split();
                                let (play_prod, play_cons) =
                                    HeapRb::<f32>::new(BUFFER_SIZE).split();

                                if let Ok(audio) = setup_audio(cap_prod, play_cons) {
                                    audio.start();
                                    let sr = audio.sample_rate;
                                    capture_task = Some(tokio::spawn(capture_processor(
                                        cap_cons,
                                        ntx,
                                        audio.is_active.clone(),
                                        sr,
                                    )));
                                    network_task = Some(tokio::spawn(run_network(
                                        socket,
                                        Some(addr),
                                        nrx,
                                        play_prod,
                                        action_tx.clone(),
                                        audio.is_active.clone(),
                                    )));
                                    audio_handle = Some(audio);
                                    app.mode = AppMode::InCall(addr);
                                }
                            }
                        }
                        (AppMode::Connecting, KeyCode::Char(c)) => app.input.push(c),
                        (AppMode::Connecting, KeyCode::Backspace) => {
                            app.input.pop();
                        }
                        (_, KeyCode::Esc) => {
                            if let Some(t) = network_task.take() {
                                t.abort();
                            }
                            if let Some(t) = capture_task.take() {
                                t.abort();
                            }
                            if let Some(a) = audio_handle.take() {
                                a.stop();
                            }
                            app.mode = AppMode::Menu;
                            app.add_log("Session ended".into());
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
