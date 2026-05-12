from langchain_core.tools import BaseTool

class BaseToolFactory:
    def __init__(self):
        self.custom_tools: dict[str, BaseTool] = {}
        self.tool_map: dict[str, BaseTool] = {}

    def register_custom_tool(self, tool: BaseTool, terminal: bool = False):
        metadata = tool.metadata or {}
        metadata["terminal"] = terminal
        tool.metadata = metadata

        self.custom_tools[tool.name] = tool

    def _build_tool_map(self, tools: list[BaseTool]):
        self.tool_map = {tool.name: tool for tool in tools}
        
    def is_terminal_tool(self, name: str) -> bool:
        tool = self.tool_map.get(name)

        if tool is None:
            return False
        
        metadata = getattr(tool, "metadata", None)
        
        if not isinstance(metadata, dict):
            return False

        return bool(metadata.get("terminal", False))
    
    def get_tools(self) -> list[BaseTool]:
        raise NotImplementedError
        