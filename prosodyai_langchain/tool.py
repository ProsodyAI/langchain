"""LangChain tool for ProsodyAI: transcript, speaker turns, measured delivery."""

from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from prosodyai_langchain.client import ProsodyClient

# Movement large enough to be worth naming to an LLM. Below this the change is
# within the range a speaker covers without doing anything different.
_NOTABLE_DB = 3.0
_NOTABLE_SEMITONES = 2.0


class ProsodyToolInput(BaseModel):
    """Input for ProsodyTool."""

    audio_path: str = Field(description="Path to audio file to analyze")
    language: str = Field(default="en", description="Language code (e.g., 'en', 'es')")


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if not isinstance(value, (int, float)):
        return "not measurable"
    return f"{value:.{digits}f}{suffix}"


def _describe_window(state: dict) -> str:
    values = (state or {}).get("values") or {}
    parts = [
        f"level {_fmt(values.get('rms_dbfs'), 1, ' dBFS')}",
        f"pitch {_fmt(values.get('f0_median_hz'), 0, ' Hz')}",
        f"pitch range {_fmt(values.get('f0_range_semitones'), 1, ' semitones')}",
        f"voiced {_fmt(values.get('voiced_ratio'))}",
        f"pause {_fmt(values.get('pause_ratio'))}",
    ]
    return ", ".join(parts)


def _largest_movement(timeline: list[dict]) -> Optional[str]:
    """The biggest speaker-relative shift on the call, if any is notable."""
    best: Optional[tuple[float, str]] = None
    for point in timeline:
        values = ((point or {}).get("acoustic_change") or {}).get("values") or {}
        for key, threshold, unit in (
            ("rms_db_change", _NOTABLE_DB, "dB"),
            ("f0_median_semitone_change", _NOTABLE_SEMITONES, "semitones"),
        ):
            value = values.get(key)
            if not isinstance(value, (int, float)):
                continue
            if abs(value) < threshold:
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
                    f"at {at_ms // 1000}s versus their previous window",
                )
    return best[1] if best else None


class ProsodyTool(BaseTool):
    """
    LangChain tool for measured speech delivery.

    Returns what the audio measured — level in dBFS, pitch in Hz, pitch range
    and movement in semitones, voiced and pause ratios — per speaker, alongside
    the transcript. These are physical quantities, not inferred emotion labels:
    deciding that a rise in level and pitch means a caller is escalating is the
    caller's policy to set on top of numbers that can be checked.
    """

    name: str = "prosody_delivery_analyzer"
    description: str = (
        "Analyzes speech audio and returns the transcript, speaker turns, and "
        "measured delivery per speaker: loudness in dBFS, pitch in Hz, pitch "
        "range in semitones, voiced and pause ratios, plus how far each "
        "speaker moved from their own previous delivery. Input should be a "
        "path to an audio file. Returns measurements, not emotion labels."
    )
    args_schema: Type[BaseModel] = ProsodyToolInput

    api_key: str
    base_url: str = "https://api.prosodyai.app"
    vertical: Optional[str] = None
    session_id: Optional[str] = None

    def _run(
        self,
        audio_path: str,
        language: str = "en",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Analyze an audio file and report what it measured."""
        client = ProsodyClient(api_key=self.api_key, base_url=self.base_url)

        try:
            result = client.analyze(
                audio=audio_path,
                language=language,
                vertical=self.vertical,
                session_id=self.session_id,
            )

            lines = [
                "Speech Analysis:",
                f"- Transcript: \"{result.get('text', '')}\"",
                f"- Duration: {_fmt(result.get('duration'), 1, 's')}",
            ]

            diarization = result.get("diarization") or {}
            speakers = diarization.get("speakers") or []
            if speakers:
                lines.append(f"- Speakers: {len(speakers)} ({', '.join(map(str, speakers))})")

            timeline = [
                point
                for point in (result.get("prosody_timeline") or [])
                if isinstance(point, dict)
            ]
            measured = [point for point in timeline if point.get("acoustic_state")]

            if measured:
                lines.append(f"- Measured windows: {len(measured)}")
                lines.append("")
                lines.append("Measured delivery (first window per speaker):")
                seen: set[str] = set()
                for point in measured:
                    speaker = str(point.get("speaker_id") or "unknown")
                    if speaker in seen:
                        continue
                    seen.add(speaker)
                    lines.append(f"- {speaker}: {_describe_window(point['acoustic_state'])}")

                movement = _largest_movement(measured)
                if movement:
                    lines.append("")
                    lines.append(f"Largest delivery shift: {movement}.")
            else:
                lines.append(
                    "- No acoustic measurements in this response "
                    "(request diarize=true to receive them)."
                )

            turns = result.get("turns") or []
            if turns:
                lines.append("")
                lines.append(f"Turns: {len(turns)}")
                for turn in turns[:5]:
                    start = int(turn.get("start_ms") or 0) // 1000
                    text = str(turn.get("text") or "")[:80]
                    lines.append(f"- [{start}s] {turn.get('speaker_id', 'unknown')}: {text}")

            if result.get("affect_available"):
                prosody = result.get("prosody") or {}
                lines.append("")
                lines.append(
                    "Affect (checkpoint-gated): "
                    f"valence {_fmt(prosody.get('valence'))}, "
                    f"arousal {_fmt(prosody.get('arousal'))}, "
                    f"dominance {_fmt(prosody.get('dominance'))}"
                )

            prediction_id = result.get("prediction_id", "")
            if prediction_id:
                lines.append("")
                lines.append(f"[prediction_id: {prediction_id}]")

            return "\n".join(lines)

        except Exception as e:
            return f"Error analyzing audio: {str(e)}"

        finally:
            client.close()
