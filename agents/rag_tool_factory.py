from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field
from retrievers.vector_retriever import VectorRetriever
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.text2cyper import Text2CypherRetriever
from neo4j_manager import Neo4jManager
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
		self.text2cypher = Text2CypherRetriever(neo4j_manager)
		self._add_few_shots()

	def _add_few_shots(self):
		self.text2cypher.add_few_shot_example(
			"What happens in Momotaro, in order, and which characters are involved in each event?",
			"""
MATCH (f:Folktale {url: "https://fairytalez.com/momotaro/"})-[:HAS_EVENT]->(e:Event)
OPTIONAL MATCH (e)-[:HAS_AGENT]->(a:Agent)
RETURN e.order AS order,
	e.name AS event,
	collect(a.name) AS agents
ORDER BY order
			"""
		)

		self.text2cypher.add_few_shot_example(
			"Which characters are most important in the story based on how many events they appear in?",
			"""
MATCH (a:Agent)
OPTIONAL MATCH (a)<-[:HAS_AGENT]-(e:Event)
RETURN a.name AS agent,
	count(DISTINCT e) AS event_count
ORDER BY event_count DESC
LIMIT 10
			"""
		)

		self.text2cypher.add_few_shot_example(
			"How many folktales come from Japan?",
			"""
MATCH (f:Folktale)-[:FROM_NATION]->(n:Nation {name: "Japan"})
RETURN count(f) AS total_folktales
			"""
		)

		self.text2cypher.add_few_shot_example(
			"Who are Momotaro's friends?",
			"""
MATCH (m:Agent {name: "Momotaro"})-[r:RELATIONSHIP]->(a:Agent)
WHERE r.type = "friend"
RETURN a.name AS friend
			"""
		)

		self.text2cypher.add_few_shot_example(
			"What happens after the specific event where Momotaro gives the dumpling to the dog?",
			"""
MATCH (e:Event {name: "Momotaro gives dog a dumpling."})-[:POST_EVENT]->(next:Event)
RETURN e.name AS current_event,
	next.name AS next_event
			"""
		)

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
			),
			StructuredTool.from_function(
				name="text2cypher",
				description="Use for structured queries that require structural information about the graph. "
				"This includes questions about relationships between entities, counts, filtering, "
				"comparisons, rankings, graph traversal, paths between nodes, temporal or event ordering "
				"and queries at the type level (e.g., categories or entity classes). "
				"Returns both the generated Cypher query and structured graph results.",
				args_schema=QueryInput,
				func=self._text2cypher
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
		results = self.vector_retriever.retrieve(query)

		return [
			self._format_record(r)
			for r in results
		]
		# return [r["description"] for r in results]
	
	def _hybrid_search(self, query: str) -> list[str]:
		results = self.hybrid_retriever.retrieve(query)

		return [
			self._format_record(r)
			for r in results
		]
		# return [r["description"] for r in results]

	def _text2cypher(self, query: str) -> dict[str, Any]:
		cypher, results = self.text2cypher.retrieve(query)
		return {
			"cypher": cypher,
			"context": [str(r) for r in results],
		}

