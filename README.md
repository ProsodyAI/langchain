# prosodyai-langchain

ProsodyAI for LangChain turns a recording into a transcript, diarized turns, and measured
speech delivery. It exposes what happened in the audio: recording-local speakers, ordered
acoustic windows, window-level vocal features, and movement against each speaker's own
previous window.

The integration does not treat recording-local speaker labels as durable identity. A label
such as `speaker_0` only identifies one speaker within that recording.

## Install

The package is not yet published to PyPI. Install the current public repository directly:

```bash
python -m pip install \
  "prosodyai-langchain @ git+https://github.com/ProsodyAI/langchain.git@main"
```

Set your API key in the environment:

```bash
export PROSODY_API_KEY="your-api-key"
```

## LangChain tool

The tool only reads audio inside an application-owned directory that you configure. Give the
agent a path relative to that directory, not an unrestricted filesystem path.

```python
import os
from pathlib import Path

from prosodyai_langchain import ProsodyTool

tool = ProsodyTool(
    api_key=os.environ["PROSODY_API_KEY"],
    allowed_audio_root=Path("./recordings"),
)

result = tool.invoke({"audio_path": "support-call.wav", "language": "en"})
print(result)
```

The formatted result includes:

- The transcript and duration.
- Recording-local speakers and diarized turns.
- The first measured acoustic window for each speaker.
- The largest notable level or pitch movement against that same speaker's previous window.
- Checkpoint-gated affect values only when the response explicitly sets
  `affect_available=true`.

Supported file extensions are `.wav`, `.mp3`, `.m4a`, `.flac`, and `.ogg`. The default file
size limit is 50 MiB and can be lowered with `max_audio_bytes`.

## Direct client

Use `ProsodyClient` when your application needs the response objects instead of tool-formatted
text.

```python
import os
from pathlib import Path

from prosodyai_langchain import ProsodyClient

with ProsodyClient(api_key=os.environ["PROSODY_API_KEY"]) as client:
    analysis = client.analyze(
        Path("./recordings/support-call.wav"),
        language="en",
        session_id="call-12345",
        diarize=True,
    )

print(analysis["text"])

for turn in analysis.get("turns") or []:
    print(turn["speaker_id"], turn["start_ms"], turn["end_ms"], turn["text"])

for window in analysis.get("prosody_timeline") or []:
    state = window.get("acoustic_state")
    change = window.get("acoustic_change")
    if state:
        print(window["speaker_id"], state["values"])
    if change:
        print(change["reference"], change["values"])
```

### Acoustic state

Each item in `prosody_timeline` is an ordered model window. Its `acoustic_state.values` can
contain physical and waveform-derived measurements such as:

- `rms_dbfs` and `peak_dbfs`
- `f0_median_hz`, `f0_range_semitones`, and `f0_slope_semitones_per_second`
- `spectral_tilt_db_per_octave`
- `voiced_ratio`, `pause_ratio`, `clipping_ratio`, and `voice_onset_rate_hz`

Unavailable measurements are `null`, not zero. Use `acoustic_state.masks` to distinguish a
missing measurement from a real zero value.

The underlying acoustic state schema can also carry `frames` at the Mimi frame rate, currently
12.5 Hz. Those arrays provide within-window level, pitch, spectral tilt, voicing, and voice
activity trajectories. Batch reports intentionally omit frame arrays and return the window
summaries needed by this integration. If a compatible endpoint includes frames, the client
preserves them unchanged.

### Same-speaker acoustic change

`acoustic_change.values` contains signed deltas such as `rms_db_change` and
`f0_median_semitone_change`. The `reference` field defines what the delta is measured against.
For the conversation timeline, the reference is the previous analyzed window for the same
recording-local speaker. The first window for a speaker normally has no delta.

## Feedback contracts

Corrections use the current VAD correction fields:

```python
with ProsodyClient(api_key=os.environ["PROSODY_API_KEY"]) as client:
    client.submit_correction(
        prediction_id="pred-123",
        corrected_valence=-0.2,
        corrected_arousal=0.7,
        notes="Reviewed by a human evaluator",
    )
```

Session outcomes use the KPI identifiers configured for the authenticated tenant:

```python
with ProsodyClient(api_key=os.environ["PROSODY_API_KEY"]) as client:
    client.submit_session_outcome(
        session_id="call-12345",
        outcomes=[
            {"kpi_id": "first_call_resolved", "boolean_value": True},
            {"kpi_id": "resolution_quality", "scalar_value": 4.0},
        ],
        notes="Imported from the post-call system of record",
    )
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the local test and release checks.

## License

MIT. See [LICENSE](LICENSE).
