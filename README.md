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

## Usage

Start the application:

```bash
cargo run
```

### Controls

*   **H**: Host a session. Generates a session offer to share.
*   **C**: Connect to a session. Takes a session offer and generates an answer.
*   **M**: Mute/unmute microphone (during call only).
*   **Enter**: Confirm input.
*   **Esc**: End call or return to menu.
*   **Q**: Quit.

### Connection Flow

1.  **Host**: Press **H**. Copy the generated Session Offer string and send it to the peer.
2.  **Client**: Press **C**. Paste the Session Offer. Copy the generated Session Answer string and send it back to the host.
3.  **Host**: Paste the Session Answer.
4.  Connection establishes automatically.
5.  **During Call**: Press **M** to toggle microphone mute. The status bar and volume gauge will reflect the mute state.

## Architecture

*   **Language**: Rust
*   **Audio**: `cpal` (I/O), `audiopus` (Opus compression), `rubato` (resampling), `ringbuf` (lock-free buffering)
*   **Network**: `webrtc-ice` for NAT traversal, `igd` for UPnP auto-mapping, `tokio` for async UDP
*   **Interface**: `ratatui` text user interface

## License

MIT License
