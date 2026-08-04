"""ProsodyAI integration for LangChain."""

from langchain_prosodyai.client import ProsodyClient
from langchain_prosodyai.tool import ProsodyAnalyzeAudioTool, ProsodyTool

__version__ = "0.3.0"

__all__ = ["ProsodyAnalyzeAudioTool", "ProsodyClient", "ProsodyTool", "__version__"]
