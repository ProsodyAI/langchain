"""Small synchronous client for the ProsodyAI batch analysis contract."""

from __future__ import annotations

import mimetypes
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

AudioInput = bytes | str | Path

_OUTCOME_FIELDS = {
    "kpi_id",
    "scalar_value",
    "boolean_value",
    "category_value",
}


class ProsodyClient:
    """Call the authenticated ProsodyAI batch analysis endpoints."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.prosodyai.app",
        *,
        timeout: float | httpx.Timeout = 60.0,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if client is not None and transport is not None:
            raise ValueError("Pass either client or transport, not both")

        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout, transport=transport)

    @property
    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self._api_key}

    @staticmethod
    def _audio_part(audio: AudioInput) -> tuple[str, bytes, str]:
        if isinstance(audio, (str, Path)):
            path = Path(audio)
            content = path.read_bytes()
            filename = path.name or "audio.wav"
        elif isinstance(audio, bytes):
            content = audio
            filename = "audio.wav"
        else:
            raise TypeError("audio must be bytes or a filesystem path")

        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return filename, content, content_type

    def analyze(
        self,
        audio: AudioInput,
        *,
        language: str = "en",
        session_id: str | None = None,
        diarize: bool = True,
    ) -> dict[str, Any]:
        """Analyze a recording for transcript, turns, and measured acoustics."""
        filename, content, content_type = self._audio_part(audio)
        data = {
            "language": language,
            "diarize": "true" if diarize else "false",
        }
        if session_id:
            data["session_id"] = session_id

        response = self._client.post(
            f"{self.base_url}/v1/analyze/audio",
            headers=self._headers,
            files={"file": (filename, content, content_type)},
            data=data,
        )
        response.raise_for_status()
        return response.json()

    def analyze_base64(
        self,
        audio_base64: str,
        *,
        language: str = "en",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Analyze base64-encoded audio using the current base64 endpoint."""
        data: dict[str, Any] = {
            "audio_base64": audio_base64,
            "language": language,
        }
        if session_id:
            data["session_id"] = session_id

        response = self._client.post(
            f"{self.base_url}/v1/analyze/base64",
            headers=self._headers,
            json=data,
        )
        response.raise_for_status()
        return response.json()

    def extract_features(self, audio: AudioInput) -> dict[str, Any]:
        """Extract the API's non-classifying prosodic feature vector."""
        filename, content, content_type = self._audio_part(audio)
        response = self._client.post(
            f"{self.base_url}/v1/features/prosody",
            headers=self._headers,
            files={"file": (filename, content, content_type)},
        )
        response.raise_for_status()
        return response.json()

    def submit_correction(
        self,
        prediction_id: str,
        *,
        corrected_valence: float | None = None,
        corrected_arousal: float | None = None,
        corrected_dominance: float | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Submit a human correction using the current feedback schema."""
        if not any(
            value is not None
            for value in (
                corrected_valence,
                corrected_arousal,
                corrected_dominance,
                notes,
            )
        ):
            raise ValueError("At least one corrected value or note is required")

        data: dict[str, Any] = {"prediction_id": prediction_id}
        for key, value in {
            "corrected_valence": corrected_valence,
            "corrected_arousal": corrected_arousal,
            "corrected_dominance": corrected_dominance,
            "notes": notes,
        }.items():
            if value is not None:
                data[key] = value

        response = self._client.post(
            f"{self.base_url}/v1/feedback/correction",
            headers=self._headers,
            json=data,
        )
        response.raise_for_status()
        return response.json()

    def submit_session_outcome(
        self,
        session_id: str,
        outcomes: Sequence[Mapping[str, Any]],
        *,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Submit tenant-configured KPI outcomes for a completed conversation."""
        normalized = self._normalize_outcomes(outcomes)
        data: dict[str, Any] = {
            "session_id": session_id,
            "outcomes": normalized,
        }
        if notes is not None:
            data["notes"] = notes

        response = self._client.post(
            f"{self.base_url}/v1/feedback/session_outcome",
            headers=self._headers,
            json=data,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _normalize_outcomes(
        outcomes: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if not outcomes:
            raise ValueError("outcomes must contain at least one KPI outcome")

        normalized: list[dict[str, Any]] = []
        for outcome in outcomes:
            if not isinstance(outcome, Mapping):
                raise TypeError("Every outcome must be a mapping")
            unknown = set(outcome) - _OUTCOME_FIELDS
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"Unsupported outcome fields: {names}")

            kpi_id = outcome.get("kpi_id")
            if not isinstance(kpi_id, str) or not kpi_id.strip():
                raise ValueError("Every outcome requires a non-empty kpi_id")

            entry = {key: value for key, value in outcome.items() if key in _OUTCOME_FIELDS}
            entry["kpi_id"] = kpi_id.strip()
            normalized.append(entry)
        return normalized

    def close(self) -> None:
        """Close the internally owned HTTP client."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> ProsodyClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
