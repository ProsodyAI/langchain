from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.tools import ToolException

import prosodyai_langchain.tool as tool_module
from prosodyai_langchain import ProsodyTool


def _analysis_fixture() -> dict[str, Any]:
    return {
        "text": "Hello there",
        "duration": 3.4,
        "affect_available": False,
        "prosody": {"valence": 0.9, "arousal": 0.8, "dominance": 0.7},
        "diarization": {"speakers": ["speaker_0", "speaker_1"]},
        "prosody_timeline": [
            {
                "start_ms": 0,
                "end_ms": 1000,
                "speaker_id": "speaker_0",
                "acoustic_state": {
                    "values": {
                        "rms_dbfs": -21.5,
                        "f0_median_hz": 180.0,
                        "f0_range_semitones": 3.2,
                        "voiced_ratio": 0.8,
                        "pause_ratio": 0.2,
                    },
                    "frames": {
                        "frame_rate_hz": 12.5,
                        "rms_dbfs": [-22.0, -21.0],
                        "f0_hz": [175.0, 185.0],
                    },
                },
            },
            {
                "start_ms": 1000,
                "end_ms": 2000,
                "speaker_id": "speaker_0",
                "acoustic_state": {
                    "values": {
                        "rms_dbfs": -17.3,
                        "f0_median_hz": 190.0,
                        "f0_range_semitones": 3.8,
                        "voiced_ratio": 0.9,
                        "pause_ratio": 0.1,
                    }
                },
                "acoustic_change": {
                    "reference": "previous_chunk_same_session_and_speaker_scope",
                    "values": {"rms_db_change": 4.2},
                },
            },
        ],
        "turns": [
            {
                "start_ms": 0,
                "end_ms": 1500,
                "speaker_id": "speaker_0",
                "text": "Hello there",
            }
        ],
    }


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, result: dict[str, Any]) -> None:
    class FakeProsodyClient:
        def __init__(self, api_key: str, base_url: str) -> None:
            assert api_key == "super-secret-key"
            assert base_url == "https://api.prosodyai.app"

        def analyze(self, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["diarize"] is True
            assert kwargs["language"] == "en"
            return result

        def close(self) -> None:
            return None

    monkeypatch.setattr(tool_module, "ProsodyClient", FakeProsodyClient)


def test_tool_formats_speakers_windows_frames_and_same_speaker_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_root = tmp_path / "recordings"
    audio_root.mkdir()
    (audio_root / "call.wav").write_bytes(b"audio")
    _install_fake_client(monkeypatch, _analysis_fixture())

    tool = ProsodyTool(
        api_key="super-secret-key",
        allowed_audio_root=audio_root,
    )
    output = tool.invoke({"audio_path": "call.wav", "language": "en"})

    assert 'Transcript: "Hello there"' in output
    assert "Recording-local speakers: 2 (speaker_0, speaker_1)" in output
    assert "Measured windows: 2" in output
    assert "level -21.5 dBFS" in output
    assert "pitch 180 Hz" in output
    assert "2 frames at 12.5 Hz" in output
    assert "speaker_0 went louder by 4.2 dB" in output
    assert "versus their own previous window" in output
    assert "Turns: 1" in output
    assert "Checkpoint-gated affect" not in output
    assert "valence" not in output


def test_api_key_is_redacted_from_repr_and_dump(tmp_path: Path) -> None:
    audio_root = tmp_path / "recordings"
    audio_root.mkdir()
    tool = ProsodyTool(
        api_key="super-secret-key",
        allowed_audio_root=audio_root,
    )

    assert "super-secret-key" not in repr(tool)
    assert "super-secret-key" not in str(tool)
    assert "api_key" not in tool.model_dump()


def test_tool_rejects_path_escape(tmp_path: Path) -> None:
    audio_root = tmp_path / "recordings"
    audio_root.mkdir()
    (tmp_path / "outside.wav").write_bytes(b"audio")
    tool = ProsodyTool(api_key="super-secret-key", allowed_audio_root=audio_root)

    with pytest.raises(ToolException, match="outside the allowed audio root"):
        tool.invoke({"audio_path": "../outside.wav"})


def test_tool_rejects_unsupported_extension(tmp_path: Path) -> None:
    audio_root = tmp_path / "recordings"
    audio_root.mkdir()
    (audio_root / "notes.txt").write_text("not audio")
    tool = ProsodyTool(api_key="super-secret-key", allowed_audio_root=audio_root)

    with pytest.raises(ToolException, match="Unsupported audio type"):
        tool.invoke({"audio_path": "notes.txt"})


def test_tool_rejects_file_over_size_limit(tmp_path: Path) -> None:
    audio_root = tmp_path / "recordings"
    audio_root.mkdir()
    (audio_root / "call.wav").write_bytes(b"12345")
    tool = ProsodyTool(
        api_key="super-secret-key",
        allowed_audio_root=audio_root,
        max_audio_bytes=4,
    )

    with pytest.raises(ToolException, match="exceeds the configured size limit"):
        tool.invoke({"audio_path": "call.wav"})


def test_tool_sanitizes_transport_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_root = tmp_path / "recordings"
    audio_root.mkdir()
    (audio_root / "call.wav").write_bytes(b"audio")

    class FailingProsodyClient:
        def __init__(self, api_key: str, base_url: str) -> None:
            del api_key, base_url

        def analyze(self, **_: Any) -> dict[str, Any]:
            request = httpx.Request("POST", "https://internal.example/secret")
            raise httpx.ConnectError("token=private-value", request=request)

        def close(self) -> None:
            return None

    monkeypatch.setattr(tool_module, "ProsodyClient", FailingProsodyClient)
    tool = ProsodyTool(api_key="super-secret-key", allowed_audio_root=audio_root)

    with pytest.raises(ToolException) as exc_info:
        tool.invoke({"audio_path": "call.wav"})

    message = str(exc_info.value)
    assert message == "ProsodyAI could not analyze the requested audio file."
    assert "private-value" not in message
    assert "internal.example" not in message
