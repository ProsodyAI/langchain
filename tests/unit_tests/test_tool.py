from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException

import langchain_prosodyai.tool as tool_module
from langchain_prosodyai import ProsodyAnalyzeAudioTool


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
                    "masks": {"f0_available": True},
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


def test_tool_preserves_the_complete_structured_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_root = tmp_path / "recordings"
    audio_root.mkdir()
    (audio_root / "call.wav").write_bytes(b"audio")
    _install_fake_client(monkeypatch, _analysis_fixture())

    tool = ProsodyAnalyzeAudioTool(
        api_key="super-secret-key",
        allowed_audio_root=audio_root,
    )
    output = tool.invoke({"audio_path": "call.wav", "language": "en"})

    assert output == _analysis_fixture()
    assert output["diarization"]["speakers"] == ["speaker_0", "speaker_1"]
    assert output["prosody_timeline"][0]["acoustic_state"]["values"]["f0_median_hz"] == 180.0
    assert output["prosody_timeline"][0]["acoustic_state"]["masks"]["f0_available"] is True
    assert output["prosody_timeline"][1]["acoustic_change"]["values"] == {"rms_db_change": 4.2}
    assert output["prosody"]["valence"] == 0.9


def test_tool_call_returns_a_tool_message_with_the_structured_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_root = tmp_path / "recordings"
    audio_root.mkdir()
    (audio_root / "call.wav").write_bytes(b"audio")
    _install_fake_client(monkeypatch, _analysis_fixture())
    tool = ProsodyAnalyzeAudioTool(
        api_key="super-secret-key",
        allowed_audio_root=audio_root,
    )

    message = tool.invoke(
        {
            "name": tool.name,
            "args": {"audio_path": "call.wav", "language": "en"},
            "id": "call_01",
            "type": "tool_call",
        }
    )

    assert isinstance(message, ToolMessage)
    assert message.name == "prosody_analyze_audio"
    assert message.tool_call_id == "call_01"
    assert "prosody_timeline" in message.content
    assert "rms_db_change" in message.content


def test_api_key_is_redacted_from_repr_and_dump(tmp_path: Path) -> None:
    audio_root = tmp_path / "recordings"
    audio_root.mkdir()
    tool = ProsodyAnalyzeAudioTool(
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
    tool = ProsodyAnalyzeAudioTool(api_key="super-secret-key", allowed_audio_root=audio_root)

    with pytest.raises(ToolException, match="outside the allowed audio root"):
        tool.invoke({"audio_path": "../outside.wav"})


def test_tool_rejects_unsupported_extension(tmp_path: Path) -> None:
    audio_root = tmp_path / "recordings"
    audio_root.mkdir()
    (audio_root / "notes.txt").write_text("not audio")
    tool = ProsodyAnalyzeAudioTool(api_key="super-secret-key", allowed_audio_root=audio_root)

    with pytest.raises(ToolException, match="Unsupported audio type"):
        tool.invoke({"audio_path": "notes.txt"})


def test_tool_rejects_file_over_size_limit(tmp_path: Path) -> None:
    audio_root = tmp_path / "recordings"
    audio_root.mkdir()
    (audio_root / "call.wav").write_bytes(b"12345")
    tool = ProsodyAnalyzeAudioTool(
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
    tool = ProsodyAnalyzeAudioTool(api_key="super-secret-key", allowed_audio_root=audio_root)

    with pytest.raises(ToolException) as exc_info:
        tool.invoke({"audio_path": "call.wav"})

    message = str(exc_info.value)
    assert message == (
        "ProsodyAI could not analyze the requested audio file; verify the "
        "configured base_url is reachable and retry."
    )
    assert "private-value" not in message
    assert "internal.example" not in message
