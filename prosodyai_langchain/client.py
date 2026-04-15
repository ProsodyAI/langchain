"""ProsodyAI API client with session tracking and forward predictions."""

from typing import Any, Optional
import httpx


class ProsodyClient:
    """
    Client for ProsodyAI API.

    Supports session-based conversation tracking for forward-looking
    predictions (escalation risk, CSAT forecast, churn risk, etc.)
    and feedback submission for continuous model improvement.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.prosodyai.app",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            headers={"X-API-Key": api_key},
            timeout=60.0,
        )

    def analyze(
        self,
        audio: bytes | str,
        language: str = "en",
        vertical: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Analyze audio for emotion.

        Args:
            audio: Audio bytes or file path
            language: Language code
            vertical: Optional vertical for domain-specific analysis
            session_id: Optional session ID for conversation tracking.
                When provided, enables forward-looking predictions
                (escalation risk, CSAT forecast, churn risk, etc.)
                across multiple utterances in the same conversation.

        Returns:
            Analysis result with emotion, confidence, VAD scores,
            and forward_predictions (if session_id provided).
        """
        if isinstance(audio, str):
            with open(audio, "rb") as f:
                audio = f.read()

        files = {"file": ("audio.wav", audio, "audio/wav")}
        data: dict[str, str] = {"language": language}
        if vertical:
            data["vertical"] = vertical
        if session_id:
            data["session_id"] = session_id

        response = self._client.post(
            f"{self.base_url}/v1/analyze/audio",
            files=files,
            data=data,
        )
        response.raise_for_status()
        return response.json()

    def analyze_base64(
        self,
        audio_base64: str,
        language: str = "en",
        vertical: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Analyze base64-encoded audio.

        Args:
            audio_base64: Base64-encoded audio data
            language: Language code
            vertical: Optional vertical for domain-specific analysis
            session_id: Optional session ID for conversation tracking

        Returns:
            Analysis result with forward_predictions if session_id provided.
        """
        data: dict[str, Any] = {
            "audio_base64": audio_base64,
            "language": language,
        }
        if vertical:
            data["vertical"] = vertical
        if session_id:
            data["session_id"] = session_id

        response = self._client.post(
            f"{self.base_url}/v1/analyze/base64",
            json=data,
        )
        response.raise_for_status()
        return response.json()

    def extract_features(self, audio: bytes | str) -> dict:
        """Extract prosodic features from audio."""
        if isinstance(audio, str):
            with open(audio, "rb") as f:
                audio = f.read()

        files = {"file": ("audio.wav", audio, "audio/wav")}
        response = self._client.post(
            f"{self.base_url}/v1/features/prosody", files=files
        )
        response.raise_for_status()
        return response.json()

    # ============ Feedback API ============

    def submit_correction(
        self,
        prediction_id: str,
        correct_emotion: str,
        correct_valence: Optional[float] = None,
        correct_arousal: Optional[float] = None,
        correct_dominance: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> dict:
        """
        Submit a human correction for a prediction.

        This is the highest-quality training signal. Corrections are
        automatically included in the next fine-tuning batch.

        Args:
            prediction_id: ID from the original analysis response
            correct_emotion: The correct emotion label
            correct_valence: Optional corrected valence (-1 to 1)
            correct_arousal: Optional corrected arousal (0 to 1)
            correct_dominance: Optional corrected dominance (0 to 1)
            notes: Optional reviewer notes
        """
        data: dict[str, Any] = {
            "prediction_id": prediction_id,
            "correct_emotion": correct_emotion,
        }
        if correct_valence is not None:
            data["correct_valence"] = correct_valence
        if correct_arousal is not None:
            data["correct_arousal"] = correct_arousal
        if correct_dominance is not None:
            data["correct_dominance"] = correct_dominance
        if notes:
            data["notes"] = notes

        response = self._client.post(
            f"{self.base_url}/v1/feedback/correction",
            json=data,
        )
        response.raise_for_status()
        return response.json()

    def submit_outcome(
        self,
        prediction_id: str,
        vertical: str,
        outcome_correct: Optional[bool] = None,
        actual_csat: Optional[float] = None,
        deal_won: Optional[bool] = None,
        deal_value: Optional[float] = None,
        phq_score: Optional[int] = None,
    ) -> dict:
        """
        Submit a real-world outcome for a specific prediction.

        Used for per-utterance model fine-tuning with outcome-weighted loss.

        Args:
            prediction_id: ID from the original analysis response
            vertical: Vertical this outcome belongs to
            outcome_correct: Was the emotion prediction correct?
            actual_csat: Actual CSAT score (contact center)
            deal_won: Did the deal close? (sales)
            deal_value: Deal value if closed (sales)
            phq_score: PHQ-9 score (healthcare)
        """
        data: dict[str, Any] = {
            "prediction_id": prediction_id,
            "vertical": vertical,
        }
        if outcome_correct is not None:
            data["outcome_correct"] = outcome_correct
        if actual_csat is not None:
            data["actual_csat"] = actual_csat
        if deal_won is not None:
            data["deal_won"] = deal_won
        if deal_value is not None:
            data["deal_value"] = deal_value
        if phq_score is not None:
            data["phq_score"] = phq_score

        response = self._client.post(
            f"{self.base_url}/v1/feedback/outcome",
            json=data,
        )
        response.raise_for_status()
        return response.json()

    def submit_session_outcome(
        self,
        session_id: str,
        vertical: str,
        *,
        actual_csat: Optional[float] = None,
        escalated: Optional[bool] = None,
        churned: Optional[bool] = None,
        first_call_resolved: Optional[bool] = None,
        transferred: Optional[bool] = None,
        deal_won: Optional[bool] = None,
        deal_value: Optional[float] = None,
        days_to_close: Optional[int] = None,
        phq_score: Optional[int] = None,
        intervention_occurred: Optional[bool] = None,
        follow_up_scheduled: Optional[bool] = None,
        final_sentiment: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> dict:
        """
        Submit how a conversation actually ended.

        This is the primary training signal for forward-looking predictions.
        Each field maps to a ConversationPredictor head:
        - escalated -> will_escalate
        - actual_csat -> final_csat
        - churned -> churn_risk
        - first_call_resolved -> resolution_prob
        - deal_won -> deal_close_prob
        - intervention_occurred -> intervention_needed
        - final_sentiment -> sentiment_forecast

        Args:
            session_id: Session ID from the conversation
            vertical: Vertical this session belongs to
            actual_csat: Actual CSAT score (contact center, 1-5)
            escalated: Did the customer escalate?
            churned: Did the customer churn within 30 days?
            first_call_resolved: Was the issue resolved on first call?
            transferred: Was the call transferred?
            deal_won: Did the deal close? (sales)
            deal_value: Deal value (sales)
            days_to_close: Days from conversation to close (sales)
            phq_score: PHQ-9 score (healthcare, 0-27)
            intervention_occurred: Was clinical intervention needed?
            follow_up_scheduled: Was a follow-up scheduled?
            final_sentiment: Human-rated final sentiment (-1 to 1)
            notes: Optional notes
        """
        data: dict[str, Any] = {
            "session_id": session_id,
            "vertical": vertical,
        }
        for key, value in {
            "actual_csat": actual_csat,
            "escalated": escalated,
            "churned": churned,
            "first_call_resolved": first_call_resolved,
            "transferred": transferred,
            "deal_won": deal_won,
            "deal_value": deal_value,
            "days_to_close": days_to_close,
            "phq_score": phq_score,
            "intervention_occurred": intervention_occurred,
            "follow_up_scheduled": follow_up_scheduled,
            "final_sentiment": final_sentiment,
            "notes": notes,
        }.items():
            if value is not None:
                data[key] = value

        response = self._client.post(
            f"{self.base_url}/v1/feedback/session_outcome",
            json=data,
        )
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
