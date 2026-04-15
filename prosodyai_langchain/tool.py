"""LangChain tool for ProsodyAI with conversation tracking and forward predictions."""

from typing import Optional, Type
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from prosodyai_langchain.client import ProsodyClient


class ProsodyToolInput(BaseModel):
    """Input for ProsodyTool."""

    audio_path: str = Field(description="Path to audio file to analyze")
    language: str = Field(default="en", description="Language code (e.g., 'en', 'es')")


class ProsodyTool(BaseTool):
    """
    LangChain tool for speech emotion analysis with forward-looking predictions.

    Analyzes audio files to detect emotion, sentiment, and prosodic features.
    When session_id is set, tracks conversation state across multiple calls
    and provides forward-looking predictions (escalation risk, CSAT forecast,
    churn risk, recommended agent tone).

    Use this when you need to understand the emotional content of speech
    and predict conversation outcomes.
    """

    name: str = "prosody_emotion_analyzer"
    description: str = (
        "Analyzes speech audio to detect emotion, sentiment, and predict "
        "conversation outcomes. Input should be a path to an audio file. "
        "Returns emotion label, confidence, valence/arousal/dominance scores, "
        "and forward-looking predictions (escalation risk, CSAT forecast, "
        "recommended agent tone) when tracking a conversation session."
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
        """Analyze audio file for emotion and predict conversation outcomes."""
        client = ProsodyClient(api_key=self.api_key, base_url=self.base_url)

        try:
            result = client.analyze(
                audio=audio_path,
                language=language,
                vertical=self.vertical,
                session_id=self.session_id,
            )

            # Format for LLM consumption
            emotion_data = result.get("emotion", {})
            emotion = (
                emotion_data.get("primary", "unknown")
                if isinstance(emotion_data, dict)
                else emotion_data
            )
            confidence = (
                emotion_data.get("confidence", 0)
                if isinstance(emotion_data, dict)
                else 0
            )

            output = f"""Speech Analysis:
- Transcription: "{result.get('text', '')}"
- Emotion: {emotion} (confidence: {confidence:.0%})
- Valence: {result.get('valence', 0):+.2f} (negative to positive)
- Arousal: {result.get('arousal', 0):.2f} (calm to excited)
- Dominance: {result.get('dominance', 0):.2f} (submissive to dominant)"""

            # Add forward predictions if available
            fwd = result.get("forward_predictions")
            if fwd:
                escalation = fwd.get("will_escalate", 0)
                onset = fwd.get("escalation_onset", 0)
                csat = fwd.get("final_csat_predicted", 3.0)
                churn = fwd.get("churn_risk", 0)
                resolution = fwd.get("resolution_probability", 0.5)
                tone = fwd.get("recommended_tone", "professional")
                sentiment = fwd.get("sentiment_forecast", 0)
                pred_confidence = fwd.get("prediction_confidence", 0)
                utterances = fwd.get("utterances_seen", 0)

                output += f"""

Forward Predictions (based on {utterances} utterances):
- Escalation Risk: {escalation:.0%}"""

                if onset > 0.3:
                    output += f" [ONSET DETECTED: {onset:.0%}]"

                output += f"""
- Predicted Final CSAT: {csat:.1f}/5
- Churn Risk: {churn:.0%}
- Resolution Probability: {resolution:.0%}
- Sentiment Forecast: {sentiment:+.2f}
- Recommended Tone: {tone}
- Prediction Confidence: {pred_confidence:.0%}"""

                # Add urgent warnings for high-risk situations
                if escalation > 0.6:
                    output += (
                        "\n\nWARNING: High escalation risk. "
                        "Recommend de-escalation: acknowledge frustration, "
                        "apologize, focus on resolution."
                    )
                if onset > 0.5:
                    output += (
                        "\n\nALERT: Escalation onset detected NOW. "
                        "Immediate tone shift recommended."
                    )

            # Add vertical analysis if available
            va = result.get("vertical_analysis")
            if va:
                output += f"""

Vertical Analysis ({va.get('vertical', '')}):
- State: {va.get('state', 'unknown')}"""
                metrics = va.get("metrics", {})
                for key, value in metrics.items():
                    output += f"\n- {key}: {value}"

            # Store prediction_id for feedback
            prediction_id = result.get("prediction_id", "")
            if prediction_id:
                output += f"\n\n[prediction_id: {prediction_id}]"

            return output

        except Exception as e:
            return f"Error analyzing audio: {str(e)}"

        finally:
            client.close()
