from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field
from retrievers.vector_retriever import VectorRetriever
from retrievers.hybrid_retriever import HybridRetriever
from graph.neo4j_manager import Neo4jManager
from .terminal_tool_factory import BaseToolFactory
from typing import Any

class QueryInput(BaseModel):
	query: str = Field(..., description="User question expressed in natural language.")

class RAGToolFactory(BaseToolFactory):
	def __init__(self, neo4j_manager: Neo4jManager, vector_index: str, fulltext_index: str, node_label: str, text_property: str, return_projection: dict[str, str], extra_match = ""):
		super().__init__()
		self.neo4j = neo4j_manager
		self.vector_retriever = VectorRetriever(
			neo4j_manager, 
			vector_index, 
			return_projection=return_projection,
			extra_match=extra_match,
			include_score=False
		)
		self.hybrid_retriever = HybridRetriever(
			neo4j_manager, 
			vector_index, 
			fulltext_index, 
			node_label, 
			text_property,
			return_projection=return_projection,
			extra_match=extra_match,
			include_score=False
		)
		
	def get_tools(self) -> list[BaseTool]:
		tools: list[BaseTool] = [
			StructuredTool.from_function(
				name="vector_search",
				description="Use it for semantic search over folktale knowledge when the user asks open-ended, "
				"conceptual or descriptive questions. Ideal for themes, meanings, summaries, "
				"story interpretation, narrative elements or sysbolism."
				"This tool performs pure vector similarity search and returns the most relevant "
				"text passages ranked by semantic relevance.",
				args_schema=QueryInput,
				func=self._vector_search,
			),
			StructuredTool.from_function(
				name="hybrid_search",
				description="Use when the query contains BOTH semantic intent and explicit signals such as "
				"named folktales, characters, places, cultures objects, or partial keywords. "
				"Best for mixed queries that require combining keyword matching with semantic retrieval, "
				"such as specific story searches with context or partial information. "
				"Returns relevant passages using a combination of vector similarity and keyword filtering.",
				args_schema=QueryInput,
				func=self._hybrid_search,
			)
		]

		tools.extend(self.custom_tools.values())

		self._build_tool_map(tools)
		return tools
	
	def _format_record(self, record: dict[str, Any]) -> str:
		return "\n".join(
			f"{key}: {value}"
			for key, value in record.items()
			if value is not None
		)

	def _vector_search(self, query: str) -> list[str]:
		results = self.vector_retriever.retrieve(query, top_k=1)

		return [
			self._format_record(r)
			for r in results
		]
		
	def _hybrid_search(self, query: str) -> list[str]:
		results = self.hybrid_retriever.retrieve(query, top_k=1)

		return [
			self._format_record(r)
			for r in results
		]
	
