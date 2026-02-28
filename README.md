# prosodyai-langchain

ProsodyAI integration for LangChain. Includes speech emotion analysis, forward-looking conversation predictions, and feedback for continuous model improvement.

## Installation

```bash
pip install prosodyai-langchain
```

## Usage

### As a LangChain Tool

```python
from prosodyai_langchain import ProsodyTool

tool = ProsodyTool(api_key="your-api-key")

# Use with an agent
result = tool.invoke({"audio_path": "./audio.wav"})
print(result)
```

### Conversation Tracking with Forward Predictions

Track a conversation across multiple utterances to get forward-looking predictions -- escalation risk, CSAT forecast, churn risk, and recommended agent tone.

```python
from prosodyai_langchain import ProsodyTool

# Set session_id to track conversation state
tool = ProsodyTool(
    api_key="your-api-key",
    vertical="contact_center",
    session_id="call-12345",
)

# Each invocation builds on the conversation history
result = tool.invoke({"audio_path": "./segment_1.wav"})
# Output includes:
# Forward Predictions (based on 1 utterances):
# - Escalation Risk: 12%
# - Predicted Final CSAT: 3.8/5
# - Recommended Tone: professional

result = tool.invoke({"audio_path": "./segment_2.wav"})
# Predictions sharpen with more context:
# Forward Predictions (based on 2 utterances):
# - Escalation Risk: 45%
# - Predicted Final CSAT: 2.6/5
# - Recommended Tone: empathetic
#
# WARNING: High escalation risk...
```

### Direct Client Usage

```python
from prosodyai_langchain import ProsodyClient

client = ProsodyClient(api_key="your-api-key")

# Analyze audio with session tracking
result = client.analyze(
    "./audio.wav",
    vertical="contact_center",
    session_id="call-12345",
)
print(result["emotion"])               # {'primary': 'frustrated', ...}
print(result["prediction_id"])         # 'pred-abc123'
print(result["forward_predictions"])   # {'will_escalate': 0.73, ...}

# Extract features only
features = client.extract_features("./audio.wav")
print(features["f0_mean"])
```

### Feedback for Continuous Improvement

Submit real-world outcomes to improve model predictions over time.

```python
from prosodyai_langchain import ProsodyClient

client = ProsodyClient(api_key="your-api-key")

# Correct a wrong prediction
client.submit_correction(
    prediction_id="pred-abc123",
    correct_emotion="angry",
)

# Submit conversation outcome (trains forward predictions)
client.submit_session_outcome(
    session_id="call-12345",
    vertical="contact_center",
    actual_csat=2.0,
    escalated=True,
    first_call_resolved=False,
)

# Submit per-prediction outcome
client.submit_outcome(
    prediction_id="pred-abc123",
    vertical="contact_center",
    actual_csat=2.0,
)
```

### With LangChain Agents

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from prosodyai_langchain import ProsodyTool

llm = ChatOpenAI(model="gpt-4")
tool = ProsodyTool(
    api_key="your-api-key",
    vertical="contact_center",
    session_id="call-12345",
)

agent = create_openai_tools_agent(llm, [tool], prompt)
executor = AgentExecutor(agent=agent, tools=[tool])

response = executor.invoke({
    "input": "Analyze the emotion in this audio: ./customer_call.wav"
})
```
