from .agentic_rag import AgenticRAG
from neo4j_manager import Neo4jManager
from .terminal_tool_factory import TerminalToolFactory
from .agent_system import AgentSystem

class MultiAgentSystem(AgentSystem):
    def __init__(self, neo4j_manager: Neo4jManager):
        super().__init__()

        self.rag = AgenticRAG(neo4j_manager)

        tool_manager = TerminalToolFactory()
        tool_manager.register_custom_tool(self.rag.as_tool(max_iterations=2))

        system_prompt=(
            "You coordinate specialized sub-agents. "
            "Available agents:\n"
            "- greeting\n"
            "- skills\n"
            "- out_of_scope\n"
            "- rag_tool\n"
            "Use the rag_tool to delegate work."
        )
        
        self._build_agent(tool_manager, system_prompt)

    