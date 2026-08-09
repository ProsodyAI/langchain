from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from langchain_prosodyai import ProsodyClient


def test_analyze_uses_current_multipart_contract(tmp_path: Path) -> None:
    audio = tmp_path / "call.mp3"
    audio.write_bytes(b"audio-bytes")
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return httpx.Response(200, json={"text": "hello", "turns": []})

    transport = httpx.MockTransport(handler)
    with ProsodyClient(
        api_key="test-api-key",
        base_url="https://api.test",
        transport=transport,
    ) as client:
        result = client.analyze(
            audio,
            language="es",
            session_id="session-1",
            diarize=True,
        )

    request = captured["request"]
    assert isinstance(request, httpx.Request)
    assert request.method == "POST"
    assert str(request.url) == "https://api.test/v1/analyze/audio"
    assert request.headers["X-API-Key"] == "test-api-key"
    assert b'filename="call.mp3"' in request.content
    assert b"Content-Type: audio/mpeg" in request.content
    assert b'name="language"\r\n\r\nes' in request.content
    assert b'name="session_id"\r\n\r\nsession-1' in request.content
    assert b'name="diarize"\r\n\r\ntrue' in request.content
    assert result["text"] == "hello"


def test_analyze_base64_uses_current_endpoint() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True})

    with ProsodyClient(
        api_key="test-api-key",
        base_url="https://api.test",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.analyze_base64("YWJj", language="en", session_id="session-1")

    assert requests[0].url.path == "/v1/analyze/base64"
    assert requests[0].headers["X-API-Key"] == "test-api-key"
    assert result == {"ok": True}


def test_submit_correction_uses_current_field_names() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "accepted"})

    with ProsodyClient(
        api_key="test-api-key",
        base_url="https://api.test",
        transport=httpx.MockTransport(handler),
    ) as client:
        client.submit_correction(
            "prediction-1",
            corrected_valence=-0.2,
            corrected_arousal=0.7,
            corrected_dominance=0.4,
            notes="reviewed",
        )

    assert payloads == [
        {
            "prediction_id": "prediction-1",
            "corrected_valence": -0.2,
            "corrected_arousal": 0.7,
            "corrected_dominance": 0.4,
            "notes": "reviewed",
        }
    ]


def test_submit_correction_rejects_empty_correction() -> None:
    client = ProsodyClient(
        api_key="test-api-key",
        transport=httpx.MockTransport(lambda _: httpx.Response(200)),
    )
    with pytest.raises(ValueError, match="At least one"):
        client.submit_correction("prediction-1")
    client.close()


def test_submit_session_outcome_uses_current_kpi_schema() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "accepted"})

    with ProsodyClient(
        api_key="test-api-key",
        base_url="https://api.test",
        transport=httpx.MockTransport(handler),
    ) as client:
        client.submit_session_outcome(
            "session-1",
            [
                {"kpi_id": "resolution_quality", "scalar_value": 4.0},
                {"kpi_id": "resolved", "boolean_value": True},
            ],
            notes="imported",
        )

    assert captured == {
        "path": "/v1/feedback/session_outcome",
        "payload": {
            "session_id": "session-1",
            "outcomes": [
                {"kpi_id": "resolution_quality", "scalar_value": 4.0},
                {"kpi_id": "resolved", "boolean_value": True},
            ],
            "notes": "imported",
        },
    }


def test_legacy_outcome_method_is_not_public() -> None:
    client = ProsodyClient(
        api_key="test-api-key",
        transport=httpx.MockTransport(lambda _: httpx.Response(200)),
    )
    assert not hasattr(client, "submit_outcome")
    client.close()


def test_injected_client_remains_caller_owned() -> None:
    injected = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"ok": True}))
    )
    client = ProsodyClient(api_key="test-api-key", client=injected)
    client.close()
    assert not injected.is_closed
    injected.close()
