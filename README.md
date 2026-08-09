<p align="center">
  <a href="https://prosodyai.app">
    <img src="https://prosodyai.app/logo.png" alt="ProsodyAI" width="88" />
  </a>
</p>

<h1 align="center">ProsodyAI for LangChain</h1>

<p align="center"><strong>Speech to speech infrastructure.</strong></p>

<p align="center">
  <a href="https://prosodyai.app">Product</a> ·
  <a href="https://prosodyai.app/docs/guides">Docs</a> ·
  <a href="https://prosodyai.app/docs/reference">API reference</a> ·
  <a href="https://github.com/ProsodyAI/langchain/issues">Issues</a>
</p>

`langchain-prosodyai` is the ProsodyAI integration package for LangChain. It gives an agent a
standard `BaseTool` that turns an application-owned recording into the complete structured
ProsodyAI response: transcription, recording-local speakers and turns, ordered acoustic
measurements, and same-speaker deltas.

## Install

The package is public but has not completed its first PyPI release. Install the current `main`
commit directly:

```bash
python -m pip install \
  "langchain-prosodyai @ git+https://github.com/ProsodyAI/langchain.git@main"
```

Set an organization API key in the environment:

```bash
export PROSODY_API_KEY="your-api-key"
```

## Create the tool

The tool only reads audio inside a directory your application explicitly owns. The model receives
a relative path, never unrestricted filesystem access.

```python
import os
from pathlib import Path

from langchain_prosodyai import ProsodyAnalyzeAudioTool

tool = ProsodyAnalyzeAudioTool(
    api_key=os.environ["PROSODY_API_KEY"],
    allowed_audio_root=Path("./recordings"),
)
```

Supported file extensions are `.wav`, `.mp3`, `.m4a`, `.flac`, and `.ogg`. The default size limit
is 50 MiB and can be lowered with `max_audio_bytes`.

## Invoke it directly

```python
analysis = tool.invoke({"audio_path": "recording.wav", "language": "en"})

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

The tool returns the API object unchanged. It does not collapse the response to a sentiment label,
one representative window, or a short text summary.

## Invoke it as a `ToolCall`

```python
message = tool.invoke(
    {
        "name": tool.name,
        "args": {"audio_path": "recording.wav", "language": "en"},
        "id": "call_01",
        "type": "tool_call",
    }
)

print(message.content)
```

## Use it with an agent

Install `langchain` plus the model provider your application uses, then pass the tool to
`create_agent`:

```python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-5.4",
    tools=[tool],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Analyze recording.wav and return its transcript, turns, and acoustic timeline.",
            }
        ]
    }
)
```

The integration does not depend on a particular model provider. Your application chooses the
LangChain model and controls which recording paths the agent may analyze.

## Response surface

The recorded-audio response can include:

- `text` and `duration`
- `turns` with `speaker_id`, timestamps, and transcript text
- `diarization` with the recording-local speaker set
- `prosody_timeline` with ordered `acoustic_state` and `acoustic_change`
- summary measurements such as RMS and peak level, pitch, spectral tilt, voicing, pauses,
  clipping, and voice-onset rate

Unavailable acoustic measurements are `null`, not zero. Use `acoustic_state.masks` to distinguish
a missing measurement from a measured zero.

`speaker_id` belongs to one recording. This integration does not treat `speaker_0` as a durable
identity across calls.

## Direct client

Use `ProsodyClient` when your application needs the ProsodyAI API without LangChain tool
orchestration:

```python
import os
from pathlib import Path

from langchain_prosodyai import ProsodyClient

with ProsodyClient(api_key=os.environ["PROSODY_API_KEY"]) as client:
    analysis = client.analyze(
        Path("./recordings/recording.wav"),
        language="en",
        session_id="call-12345",
        diarize=True,
    )
```

The client also exposes `analyze_base64`, `submit_correction`, and
`submit_session_outcome` for the corresponding authenticated API resources.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for tests, package checks, and the protected release path.
Report security issues through [SECURITY.md](SECURITY.md).

## License

MIT © [Prosody AI, Inc.](https://prosodyai.app)
