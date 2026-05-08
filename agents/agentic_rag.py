from typing import Any
from langchain_core.tools import StructuredTool
from neo4j_manager import Neo4jManager
from .rag_tool_factory import RAGToolFactory
from pydantic import BaseModel, Field
from .agent_system import AgentSystem
from typing import Optional
import uuid

class RAGToolInput(BaseModel):
    queries: list[str] = Field(..., description="User questions to anser using the RAG Ssytem.")

class AgenticRAG(AgentSystem):
    def __init__(self, neo4j_manager: Neo4jManager, embedding_index: str, fulltext_index: str, node_label: str, text_property: str, system_prompt: Optional[str] = None, tool_name: str = "rag_tool", description: str = "Answers questions using RAG over a knowledge base."):
        super().__init__()

        tool_manager = RAGToolFactory(
            neo4j_manager, 
            embedding_index, 
            fulltext_index, 
            node_label, 
            text_property
        )

        self.system_prompt = system_prompt or """You are a concise question-answering assistant.

STRICT RULES:
1. Answer ONLY from the provided context. Do not use prior knowledge.
2. Start your reply with the answer itself. Never begin with "I", "Let me", "Based on", "The context", or any preamble.
3. Do not explain your reasoning or thinking process. Only state facts.
4. Keep the answer short: one or two sentences maximum.
5. Add an inline citation after each fact.
6. If the context does not contain the answer, reply: "This information is not in the knowledge base.
"""

        self._build_agent(tool_manager, self.system_prompt)

        self.tool_name = tool_name
        self.tool_description = description
    
    def as_tool(self, max_iterations: int = 2) -> StructuredTool:
        def _run(queries: list[str]) -> dict[str, Any]:
            question = "\n".join(queries)

            result = self.answer(
                question=question,
                thread_id=f"subagent-{uuid.uuid4()}",
                max_iterations=max_iterations
            )

            iterations = result.get("iterations", [])

            last_context = []
            if iterations:
                last_context = iterations[-1].get("context", [])

            return {
                "answer": result.get("answer"),
                "context": last_context,
            }

        return StructuredTool.from_function(
            name=self.tool_name,
            description=self.tool_description,
            func=_run,
            args_schema=RAGToolInput,
        )
