# Audio Domain Plan: `ovi-audio` (ASR + TTS)

> This document records the research, design decisions, and implementation plan for adding audio inference capabilities to the `ovi` ecosystem.
> Last updated: 2026-04-07

## Goals

1. **ASR (Automatic Speech Recognition)** — audio to text transcription
2. **TTS (Text-to-Speech)** — text to audio synthesis

Both powered by OpenVINO inference, following the same dual-language (C++ + Rust) pattern as `ovi-vision`.

---

## Technology Choice

### ASR: OpenVINO GenAI `WhisperPipeline`

**Why Whisper via GenAI:**
- Whisper is encoder-decoder with autoregressive decoding — implementing the decoding loop manually is not worth it
- GenAI's `WhisperPipeline` C++ API is high-level: one `generate()` call handles everything
- Supports Whisper tiny/base/small/medium/large-v3 and Distil-Whisper variants
- Automatic 30-second chunking, language detection, timestamps
- CPU/GPU/NPU support

**C++ API example:**
```cpp
#include "openvino/genai/whisper_pipeline.hpp"

ov::genai::WhisperPipeline pipe("whisper-base-ov/", "CPU");
// raw_speech: std::vector<float>, 16kHz, normalized [-1, 1]
auto result = pipe.generate(raw_speech, ov::genai::max_new_tokens(100));
// result.texts[0] = transcribed text

// With timestamps:
auto result = pipe.generate(raw_speech,
    ov::genai::max_new_tokens(100),
    ov::genai::return_timestamps(true),
    ov::genai::language("<|en|>"));
// result.chunks contains {start, end, text} entries
```

**Model conversion:**
```bash
pip install optimum[openvino]
optimum-cli export openvino --model openai/whisper-base whisper-base-ov
```

### TTS: OpenVINO GenAI `Text2SpeechPipeline` (SpeechT5)

**Why SpeechT5 via GenAI:**
- Same library as Whisper (`libopenvino_genai`) — one dependency for both
- C++ API consistency: same pattern as WhisperPipeline

**C++ API example:**
```cpp
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

ov::genai::Text2SpeechPipeline pipeline("speecht5-ov/", "CPU");
auto result = pipeline.generate("Hello, this is a test.");
// result contains mono audio at 16 kHz
```

**Model conversion:**
```bash
optimum-cli export openvino --model microsoft/speecht5_tts speecht5-ov \
  --model-kwargs '{"vocoder": "microsoft/speecht5_hifigan"}'
```

**Future upgrade path:** MeloTTS.cpp (better voice quality, more languages) can replace SpeechT5 later if needed.

---

## Dependency: `openvino.genai` C++ Library

### Current Environment

- macOS Apple Silicon (arm64)
- OpenVINO 2025.3.0 installed via Homebrew at `/opt/homebrew/`
- `openvino.genai` is **NOT** included in the Homebrew formula — must be installed separately

### Installation: Build from Source (Recommended)

Building from source avoids ABI mismatch risk. The pre-built archive bundles its own OpenVINO runtime (potentially different version), which could conflict with the Homebrew OpenVINO that `ovi-vision` links against. Building from source ensures `openvino.genai` links against the same Homebrew OpenVINO.

**Steps:**

```bash
# 1. Upgrade Homebrew OpenVINO to latest (recommended)
brew upgrade openvino

# 2. Clone openvino.genai
git clone --recursive https://github.com/openvinotoolkit/openvino.genai.git
cd openvino.genai

# 3. Build against Homebrew OpenVINO (Python bindings disabled — we only need C++)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_PYTHON=OFF \
      -DOpenVINO_DIR=/opt/homebrew/lib/cmake/openvino \
      -S ./ -B ./build/
cmake --build ./build/ --config Release -j$(sysctl -n hw.ncpu)

# 4. Install to a local prefix
cmake --install ./build/ --config Release --prefix /opt/openvino_genai

# 5. Verify
ls /opt/openvino_genai/include/openvino/genai/whisper_pipeline.hpp
ls /opt/openvino_genai/lib/libopenvino_genai.dylib
```

**What this provides:**

| Item | Path |
|------|------|
| Headers | `/opt/openvino_genai/include/openvino/genai/*.hpp` |
| Library | `/opt/openvino_genai/lib/libopenvino_genai.dylib` |
| Tokenizers plugin | `/opt/openvino_genai/lib/libopenvino_tokenizers.dylib` |
| CMake config | `/opt/openvino_genai/lib/cmake/OpenVINOGenAI/` |

**CMake usage in our project:**
```cmake
find_package(OpenVINOGenAI REQUIRED)
target_link_libraries(ovi_audio PRIVATE openvino::genai)
```

**Build requirements:** CMake 3.23+, Python 3.10+ (build dependency only), C++17 compiler (Xcode Clang).

---

## Architecture Design

### Key Difference from Vision

Vision uses **raw `ov::Core` API** (load model, set tensor, infer, get tensor). Audio uses **GenAI high-level pipelines** (`WhisperPipeline`, `Text2SpeechPipeline`) which internally manage encoder/decoder/tokenizer.

This means:
- `ovi-audio` does **NOT** share `ov::Core` with `ovi-vision` — GenAI pipelines create their own core internally
- The shared `ovi-core` crate would provide `Device`, `Error`, and possibly other common types, but **NOT** `Core` (GenAI doesn't need it passed in)

### Crate Structure

```
crates/
├── ovi-core/              # Shared types: Device, Error, Result
├── ovi-vision-sys/        # Vision FFI (links libovi_vision + libopenvino)
├── ovi-vision/            # Vision safe API
├── ovi-audio-sys/         # Audio FFI (links libopenvino_genai)
│   ├── build.rs           # CMake build for audio C++ bridge
│   └── src/lib.rs         # CXX bridge definitions
├── ovi-audio/             # Audio safe API
│   └── src/
│       ├── lib.rs
│       ├── asr.rs         # WhisperRecognizer with builder pattern
│       └── tts.rs         # SpeechSynthesizer with builder pattern
└── ovi/                   # Umbrella crate (feature flags)
```

### C++ Bridge Design

```
cpp/
├── include/ovi/
│   ├── core/              # Shared utility headers (existing)
│   ├── vision/            # Vision headers (existing)
│   └── audio/             # Audio headers (new)
│       ├── asr.hpp        # Whisper wrapper
│       ├── tts.hpp        # TTS wrapper
│       └── ffi/
│           └── bridge.hpp # CXX bridge header
├── audio/
│   ├── src/
│   │   ├── asr.cpp        # WhisperPipeline wrapper implementation
│   │   └── tts.cpp        # Text2SpeechPipeline wrapper implementation
│   └── ffi/
│       ├── bridge.cpp     # CXX bridge implementation
│       └── CMakeLists.txt # Links against openvino::genai
└── vision/                # (existing, unchanged)
```

### Rust API Design (Sketch)

```rust
use ovi_audio::{WhisperRecognizer, SpeechSynthesizer, Device};

// ASR
let recognizer = WhisperRecognizer::builder("whisper-base-ov/")
    .device(Device::Cpu)
    .language("en")
    .build()?;

let result = recognizer.transcribe(&audio_samples)?;
println!("{}", result.text);

for chunk in result.chunks {
    println!("[{:.1}s - {:.1}s] {}", chunk.start, chunk.end, chunk.text);
}

// TTS
let synth = SpeechSynthesizer::builder("speecht5-ov/")
    .device(Device::Cpu)
    .build()?;

let audio = synth.generate("Hello, world!")?;
// audio.samples: Vec<f32>, audio.sample_rate: u32
```

---

## Implementation Order

### Phase 0: Environment Setup
1. `brew upgrade openvino`
2. Build `openvino.genai` from source against Homebrew OpenVINO
3. Verify `whisper_pipeline.hpp` and `libopenvino_genai.dylib` exist

### Phase 1: Extract `ovi-core`
1. Create `crates/ovi-core/` with `Device`, `Error`, `Result`
2. Make `ovi-vision` depend on `ovi-core` instead of defining these locally
3. Verify `cargo build && cargo test`

### Phase 2: ASR (Whisper)
1. Create C++ wrapper around `WhisperPipeline` in `cpp/audio/`
2. Create CXX bridge in `cpp/audio/ffi/`
3. Create `crates/ovi-audio-sys/` with `build.rs` linking `libopenvino_genai`
4. Create `crates/ovi-audio/` with `WhisperRecognizer` builder API
5. Download a Whisper model and test

### Phase 3: TTS (SpeechT5)
1. Add `Text2SpeechPipeline` wrapper to `cpp/audio/`
2. Add `SpeechSynthesizer` to `ovi-audio`
3. Download SpeechT5 model and test

### Phase 4: Umbrella Crate
1. Create `crates/ovi/` with feature flags (`vision`, `audio`)
2. Update examples to use umbrella crate

---

## Model Management

ASR and TTS models are converted from HuggingFace, not from Open Model Zoo:

```bash
pip install optimum[openvino]

# ASR models
optimum-cli export openvino --model openai/whisper-base assets/audio/whisper-base-ov
optimum-cli export openvino --model openai/whisper-small assets/audio/whisper-small-ov

# TTS models
optimum-cli export openvino --model microsoft/speecht5_tts assets/audio/speecht5-ov \
  --model-kwargs '{"vocoder": "microsoft/speecht5_hifigan"}'
```

Models should be stored under `assets/audio/` (gitignored, like vision models).

---

## Notes

- GenAI's `WhisperPipeline` takes a model **directory path** (not a single .xml file) because it loads multiple sub-models (encoder, decoder, tokenizer)
- Audio input for Whisper must be 16kHz mono float samples in [-1, 1] range
- TTS output is typically 16kHz mono float samples
- The `openvino_tokenizers` plugin (`.dylib`) must be discoverable by OpenVINO at runtime — it's loaded as a plugin, not linked directly
