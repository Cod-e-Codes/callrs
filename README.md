# callrs

A terminal based peer to peer audio communication tool written in Rust. This application facilitates real-time voice calls over UDP using a text user interface.

## Prerequisites

The application requires the following system dependencies for audio processing:

* **Linux**: ALSA development files (libasound2-dev).
* **Windows/macOS**: Standard audio drivers.
* **Rust**: A working Rust toolchain (edition 2024).

## Installation

1. Clone the repository.
2. Build the project using cargo:
```bash
cargo build --release

```



## Usage

Run the executable:

```bash
cargo run

```

### Key Bindings

* **H**: Host a session. This binds the application to UDP port 9090 and waits for an incoming connection.
* **C**: Connect mode. This allows you to type the target IP address and port.
* **Enter**: Confirm the address and initiate a connection while in Connect mode.
* **Backspace**: Remove characters while typing the address.
* **Esc**: End the current call or return to the main menu.
* **Q**: Quit the application (available in Menu, Hosting, or InCall modes).

## Technical Details

* **Audio**: Uses `cpal` for hardware abstraction. It supports both F32 and I16 sample formats, converting audio to a mono stream for transmission.
* **Networking**: Uses `tokio` and asynchronous UDP sockets. Audio is transmitted in 20ms frames.
* **UI**: Built with `ratatui` and `crossterm` to provide a real-time status display and activity log.
* **Buffers**: Employs `ringbuf` for thread-safe, lock-free synchronization between the audio callback threads and the network processing tasks.

## Protocol

The application uses a simple packet structure:

* First 4 bytes: Little-endian u32 sequence number.
* Remaining bytes: Raw i16 audio samples in little-endian format.
