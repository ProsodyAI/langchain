from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_prosodyai import ProsodyAnalyzeAudioTool


class TestProsodyAnalyzeAudioToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ProsodyAnalyzeAudioTool

    @property
    def tool_constructor_params(self) -> dict[str, object]:
        return {
            "api_key": "test-api-key",
            "allowed_audio_root": Path("."),
        }

    @property
    def tool_invoke_params_example(self) -> dict[str, object]:
        return {"audio_path": "call.wav", "language": "en"}
