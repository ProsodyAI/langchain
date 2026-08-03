"""LangChain tool for transcript, speaker turns, and measured delivery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, SecretStr

from prosodyai_langchain.client import ProsodyClient

_NOTABLE_DB = 3.0
_NOTABLE_SEMITONES = 2.0
_SUPPORTED_AUDIO_EXTENSIONS = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}
_DEFAULT_MAX_AUDIO_BYTES = 50 * 1024 * 1024


class ProsodyToolInput(BaseModel):
    """Input for a ProsodyAI analysis within the configured audio root."""

    audio_path: str = Field(
        description="Audio path relative to the tool's configured allowed_audio_root"
    )
    language: str = Field(default="en", description="Language code, such as en or es")


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if not isinstance(value, (int, float)):
        return "not measurable"
    return f"{value:.{digits}f}{suffix}"


def _frame_summary(state: dict[str, Any]) -> str | None:
    frames = state.get("frames")
    if not isinstance(frames, dict):
        return None

    frame_count = max(
        (
            len(values)
            for key, values in frames.items()
            if key != "frame_rate_hz" and isinstance(values, list)
        ),
        default=0,
    )
    if not frame_count:
        return None

    rate = frames.get("frame_rate_hz")
    if isinstance(rate, (int, float)):
        return f"{frame_count} frames at {rate:g} Hz"
    return f"{frame_count} frames"


def _describe_window(state: dict[str, Any]) -> str:
    values = state.get("values") or {}
    parts = [
        f"level {_fmt(values.get('rms_dbfs'), 1, ' dBFS')}",
        f"pitch {_fmt(values.get('f0_median_hz'), 0, ' Hz')}",
        f"pitch range {_fmt(values.get('f0_range_semitones'), 1, ' semitones')}",
        f"voiced {_fmt(values.get('voiced_ratio'))}",
        f"pause {_fmt(values.get('pause_ratio'))}",
    ]
    frame_summary = _frame_summary(state)
    if frame_summary:
        parts.append(frame_summary)
    return ", ".join(parts)


def _largest_movement(timeline: list[dict[str, Any]]) -> str | None:
    """Return the largest notable same-speaker movement in the recording."""
    best: tuple[float, str] | None = None
    for point in timeline:
        values = (point.get("acoustic_change") or {}).get("values") or {}
        for key, threshold, unit in (
            ("rms_db_change", _NOTABLE_DB, "dB"),
            ("f0_median_semitone_change", _NOTABLE_SEMITONES, "semitones"),
        ):
            value = values.get(key)
            if not isinstance(value, (int, float)) or abs(value) < threshold:
                continue
            if best is None or abs(value) > best[0]:
                at_ms = int(point.get("start_ms") or 0)
                speaker = point.get("speaker_id") or "unknown"
                label = "louder" if key == "rms_db_change" else "higher"
                if value < 0:
                    label = "quieter" if key == "rms_db_change" else "lower"
                best = (
                    abs(value),
                    f"{speaker} went {label} by {abs(value):.1f} {unit} "
                    f"at {at_ms // 1000}s versus their own previous window",
                )
    return best[1] if best else None


class ProsodyTool(BaseTool):
    """Analyze an application-owned audio file with ProsodyAI."""

    name: str = "prosody_delivery_analyzer"
    description: str = (
        "Analyzes an audio file inside the configured audio root. Returns the "
        "transcript, recording-local speakers and turns, measured delivery windows, "
        "and same-speaker acoustic changes. Input is a relative audio path."
    )
    args_schema: type[BaseModel] = ProsodyToolInput

    api_key: SecretStr = Field(repr=False, exclude=True)
    allowed_audio_root: Path
    base_url: str = "https://api.prosodyai.app"
    session_id: str | None = None
    max_audio_bytes: int = Field(default=_DEFAULT_MAX_AUDIO_BYTES, gt=0)

    def _resolve_audio_path(self, audio_path: str) -> Path:
        try:
            root = self.allowed_audio_root.expanduser().resolve(strict=True)
        except OSError:
            raise ToolException("The configured audio root is unavailable.") from None

        candidate = Path(audio_path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate

        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            raise ToolException("The requested audio file was not found.") from None

        if not resolved.is_relative_to(root):
            raise ToolException("The requested audio file is outside the allowed audio root.")
        if not resolved.is_file():
            raise ToolException("The requested audio path is not a file.")
        if resolved.suffix.lower() not in _SUPPORTED_AUDIO_EXTENSIONS:
            supported = ", ".join(sorted(_SUPPORTED_AUDIO_EXTENSIONS))
            raise ToolException(f"Unsupported audio type. Supported extensions: {supported}.")

        try:
            size = resolved.stat().st_size
        except OSError:
            raise ToolException("The requested audio file could not be inspected.") from None
        if size > self.max_audio_bytes:
            raise ToolException("The requested audio file exceeds the configured size limit.")
        return resolved

    def _run(
        self,
        audio_path: str,
        language: str = "en",
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Analyze a validated audio file and format measured results."""
        del run_manager
        resolved = self._resolve_audio_path(audio_path)
        client: ProsodyClient | None = None

        try:
            client = ProsodyClient(
                api_key=self.api_key.get_secret_value(),
                base_url=self.base_url,
            )
            result = client.analyze(
                audio=resolved,
                language=language,
                session_id=self.session_id,
                diarize=True,
            )
            return self._format_result(result)
        except ToolException:
            raise
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            message = f"ProsodyAI rejected the analysis request with HTTP {status}."
            raise ToolException(message) from None
        except Exception:
            raise ToolException("ProsodyAI could not analyze the requested audio file.") from None
        finally:
            if client is not None:
                client.close()

    @staticmethod
    def _format_result(result: dict[str, Any]) -> str:
        lines = [
            "Speech Analysis:",
            f'- Transcript: "{result.get("text", "")}"',
            f"- Duration: {_fmt(result.get('duration'), 1, 's')}",
        ]

        diarization = result.get("diarization") or {}
        speakers = diarization.get("speakers") or []
        if speakers:
            speaker_names = ", ".join(map(str, speakers))
            lines.append(f"- Recording-local speakers: {len(speakers)} ({speaker_names})")

        timeline = [
            point for point in (result.get("prosody_timeline") or []) if isinstance(point, dict)
        ]
        measured = [point for point in timeline if isinstance(point.get("acoustic_state"), dict)]

        if measured:
            lines.append(f"- Measured windows: {len(measured)}")
            lines.extend(("", "Measured delivery (first window per speaker):"))
            seen: set[str] = set()
            for point in measured:
                speaker = str(point.get("speaker_id") or "unknown")
                if speaker in seen:
                    continue
                seen.add(speaker)
                lines.append(f"- {speaker}: {_describe_window(point['acoustic_state'])}")

            movement = _largest_movement(measured)
            if movement:
                lines.extend(("", f"Largest same-speaker delivery shift: {movement}."))
        else:
            lines.append("- No acoustic measurements were available in this response.")

        turns = result.get("turns") or []
        if turns:
            lines.extend(("", f"Turns: {len(turns)}"))
            for turn in turns[:5]:
                start = int(turn.get("start_ms") or 0) // 1000
                text = str(turn.get("text") or "")[:80]
                lines.append(f"- [{start}s] {turn.get('speaker_id', 'unknown')}: {text}")

        if result.get("affect_available"):
            prosody = result.get("prosody") or {}
            lines.extend(
                (
                    "",
                    "Checkpoint-gated affect: "
                    f"valence {_fmt(prosody.get('valence'))}, "
                    f"arousal {_fmt(prosody.get('arousal'))}, "
                    f"dominance {_fmt(prosody.get('dominance'))}",
                )
            )

        return "\n".join(lines)
