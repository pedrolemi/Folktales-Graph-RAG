from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel
from .tool_factory import BaseToolFactory

class EmptyInput(BaseModel):
    """No input required."""
    pass

class TerminalToolFactory(BaseToolFactory):
    _GREETING_RESPONSE = (
        "Hello! I’m a knowledge assistant focused on folktales from all around the world. You can ask me about their characters, relationships, themes or how they are structured."
    )

    _OUT_OF_SCOPE_RESPONSE = (
        "This question is outside my scope."
    )

    _SKILLS_RESPONSE = (
        "I can answer questions about folktales: their origins, characters, themes, narrative structures, cultural contexts and the relationships between characters."
    )

    def get_tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = [
            StructuredTool.from_function(
                name="greeting",
                description=(
                    "Handle conversational messages like greetings, farewells, thanks or capability questions."
                ),
                args_schema=EmptyInput,
                func=self._greeting,
                return_direct=True,
                metadata={"terminal": True}
            ),
            StructuredTool.from_function(
                name="out_of_scope",
                description=(
                    "Handle questions unrelated to folktales."
                ),
                args_schema=EmptyInput,
                func=self._out_of_scope,
                return_direct=True,
                metadata={"terminal": True}
            ),
            StructuredTool.from_function(
                name="skills",
                description="Explain system capabilities.",
                args_schema=EmptyInput,
                func=self._skills,
                return_direct=True,
                metadata={"terminal": True}
            )
        ]

        tools.extend(self.custom_tools.values())

        self._build_tool_map(tools)
        return tools

    def _greeting(self) -> str:
        return self._GREETING_RESPONSE
    
    def _out_of_scope(self) -> str:
        return self._OUT_OF_SCOPE_RESPONSE

    def _skills(self) -> str:
        return self._SKILLS_RESPONSE
        