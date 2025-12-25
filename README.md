# callrs

Peer to peer voice chat tool written in Rust using a terminal interface. Facilitates real time audio calls using ICE for NAT traversal.

## Requirements

*   **Linux**: libasound2-dev
*   **Windows/macOS**: Standard drivers
*   **Rust**: Stable toolchain

## Build

```bash
cargo build --release
```

## Testing

Run the test suite:

```bash
cargo test
```

The test suite includes comprehensive coverage of audio processing, jitter buffer management, packet loss concealment, and session serialization.

## Usage

Start the application:

```bash
cargo run
```

### Controls

*   **H**: Host a session. Generates a session offer to share.
*   **C**: Connect to a session. Takes a session offer and generates an answer.
*   **M**: Mute/unmute microphone (during call only).
*   **Y**: Copy offer/answer to clipboard (when viewing offer or answer).
*   **Enter**: Confirm input.
*   **Esc**: End call, dismiss error, or return to menu.
*   **Q**: Quit.

### Connection Flow

1.  **Host**: Press **H**. Press **Y** to copy the generated Session Offer to your clipboard, then paste it to send to the peer.
2.  **Client**: Press **C**. Paste the Session Offer (using your terminal's paste: right-click, Ctrl+V, or Ctrl+Shift+V) in the input field, then press **Enter**. Press **Y** to copy the generated Session Answer to your clipboard, then paste it to send back to the host.
3.  **Host**: Paste the Session Answer.
4.  Connection establishes automatically.
5.  **During Call**: Press **M** to toggle microphone mute. The status bar and volume gauge will reflect the mute state.

## Architecture

*   **Language**: Rust
*   **Audio**: `cpal` (I/O), `audiopus` (Opus compression), `rubato` (resampling), `ringbuf` (lock-free buffering)
*   **Network**: `webrtc-ice` for NAT traversal, `igd` for UPnP auto-mapping, `tokio` for async UDP
*   **Interface**: `ratatui` text user interface

## Technical Features

*   **Adaptive Jitter Buffer**: Dynamically adjusts buffering (20-150ms) based on network conditions to balance latency and reliability. Monitors packet arrival patterns and adapts to maintain smooth audio even on unstable connections.

*   **Packet Loss Concealment**: Detects missing packets and generates fade-out audio to mask gaps up to 200ms, preventing jarring dropouts during temporary packet loss.

*   **Noise Gate**: Filters out background noise below a threshold (1% RMS), reducing bandwidth usage during silence and improving overall audio quality.

*   **Sample Rate Adaptation**: Automatically resamples audio to 48kHz for optimal Opus encoding, regardless of device capabilities, ensuring consistent quality across different hardware.

## License

MIT License
