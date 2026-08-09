"""LangChain tool for the complete ProsodyAI recorded-audio response."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, SecretStr

from langchain_prosodyai.client import ProsodyClient

_SUPPORTED_AUDIO_EXTENSIONS = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}
_DEFAULT_MAX_AUDIO_BYTES = 50 * 1024 * 1024


class ProsodyAnalyzeAudioInput(BaseModel):
    """Input for a ProsodyAI analysis within the configured audio root."""

    audio_path: str = Field(
        description="Audio path relative to the tool's configured allowed_audio_root"
    )
    language: str = Field(default="en", description="BCP-47 language code, such as en or es")


class ProsodyAnalyzeAudioTool(BaseTool):
    """Return ProsodyAI's structured analysis for an application-owned audio file."""

    name: str = "prosody_analyze_audio"
    description: str = (
        "Analyze an audio file inside the configured audio root. Returns the complete "
        "structured ProsodyAI response: transcript, recording-local speakers and turns, "
        "ordered acoustic measurements, and same-speaker deltas. Input is a relative path."
    )
    args_schema: type[BaseModel] = ProsodyAnalyzeAudioInput

    api_key: SecretStr = Field(repr=False, exclude=True)
    allowed_audio_root: Path
    base_url: str = "https://api.prosodyai.app"
    session_id: str | None = None
    max_audio_bytes: int = Field(default=_DEFAULT_MAX_AUDIO_BYTES, gt=0)

    def _resolve_audio_path(self, audio_path: str) -> Path:
        try:
            root = self.allowed_audio_root.expanduser().resolve(strict=True)
        except OSError:
            raise ToolException(
                "The configured audio root does not resolve to an existing directory; "
                "fix allowed_audio_root where the tool is constructed."
            ) from None

        candidate = Path(audio_path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate

        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            raise ToolException(
                f"No file exists at {audio_path!r} inside the audio root; pass the "
                "relative path of an existing audio file."
            ) from None

        if not resolved.is_relative_to(root):
            raise ToolException(
                f"The requested path {audio_path!r} resolves outside the allowed audio "
                "root; pass a path inside the audio root."
            )
        if not resolved.is_file():
            raise ToolException(
                f"The requested path {audio_path!r} does not point to a regular file; "
                "pass the path of an audio file."
            )
        if resolved.suffix.lower() not in _SUPPORTED_AUDIO_EXTENSIONS:
            supported = ", ".join(sorted(_SUPPORTED_AUDIO_EXTENSIONS))
            raise ToolException(
                f"Unsupported audio type {resolved.suffix.lower()!r}. "
                f"Supported extensions: {supported}."
            )

        try:
            size = resolved.stat().st_size
        except OSError:
            raise ToolException(
                f"The audio file at {audio_path!r} could not be inspected; check "
                "filesystem permissions on the audio root."
            ) from None
        if size > self.max_audio_bytes:
            raise ToolException(
                f"The audio file is {size} bytes, which exceeds the configured size "
                f"limit of {self.max_audio_bytes} bytes; trim the recording or raise "
                "max_audio_bytes."
            )
        return resolved

    def _run(
        self,
        audio_path: str,
        language: str = "en",
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        """Analyze a validated audio file and preserve the complete API response."""
        del run_manager
        resolved = self._resolve_audio_path(audio_path)
        client: ProsodyClient | None = None

        try:
            client = ProsodyClient(
                api_key=self.api_key.get_secret_value(),
                base_url=self.base_url,
            )
            return client.analyze(
                audio=resolved,
                language=language,
                session_id=self.session_id,
                diarize=True,
            )
        except ToolException:
            raise
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            message = (
                f"ProsodyAI rejected the analysis request with HTTP {status}; check "
                "the API key and the audio file, then retry."
            )
            raise ToolException(message) from None
        except Exception:
            # Deliberately opaque: transport errors can carry URLs and tokens,
            # so none of the underlying detail reaches the calling agent.
            raise ToolException(
                "ProsodyAI could not analyze the requested audio file; verify the "
                "configured base_url is reachable and retry."
            ) from None
        finally:
            if client is not None:
                client.close()


# Compatibility with the pre-release repository import. New code should use
# ProsodyAnalyzeAudioTool so the exported class matches the action it performs.
ProsodyTool = ProsodyAnalyzeAudioTool

__all__ = [
    "ProsodyAnalyzeAudioInput",
    "ProsodyAnalyzeAudioTool",
    "ProsodyTool",
]
