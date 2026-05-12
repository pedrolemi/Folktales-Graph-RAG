from typing import Any
from pydantic import BaseModel, Field
from graph.neo4j_manager import Neo4jManager
from utils.models import get_llm
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate, 
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from .base_retriever import BaseRetriever
from typing import cast, Optional
from langchain_core.tools import StructuredTool

class CypherQuery(BaseModel):
    cypher: str = Field(
		...,
		description=(
			"Cypher query to execute against the Neo4j knowledge graph. "
			"Must be a valid Cypher statement using the folktale schema "
			"(e.g., nodes like Folktale, Event, Character, Nation and relationships "
			"such as HAS_EVENT, HAS_CHARACTER, FROM_NATION, POST_EVENT)."
		),
	)

class QueryInput(BaseModel):
	query: str = Field(..., description="User question expressed in natural language.")

class Text2CypherRetriever(BaseRetriever):
    def __init__(self, neo4j_manager: Neo4jManager):
        super().__init__(neo4j_manager, "")
        self.model = get_llm(0.0)
        self.few_shot_examples = []

        schema = self.neo4j.get_schema()
        self.schema_str = self.neo4j.format_schema(schema)
        self.schema_str = self.schema_str.replace("{", "{{").replace("}", "}}")

        self._build_chain()

    def add_few_shots(self):
#         self.add_few_shot_example(
#             "What happens in the folktale 'Momotaro', in order, and which characters are involved in each event?",
#             """
# MATCH (f:Folktale)-[:HAS_EVENT]->(e:Event)
# WHERE toLower(f.title) CONTAINS "momotaro"
# OPTIONAL MATCH (e)-[:HAS_CHARACTER]->(a:Character)
# RETURN e.order AS order,
#     e.name AS event,
#     collect(a.name) AS agents
# ORDER BY order
#     """
#         )

        self.add_few_shot_example(
            "Which characters are most important based on how on the number of events they appear in?",
            """
MATCH (a:Character)
OPTIONAL MATCH (a)<-[:HAS_CHARACTER]-(e:Event)
RETURN a.name AS character,
count(DISTINCT e) AS event_count
ORDER BY event_count DESC
LIMIT 10
    """
        )

        self.add_few_shot_example(
            "How many folktales come from Japan?",
            """
MATCH (f:Folktale)-[:FROM_NATION]->(n:Nation)
WHERE toLower(n.name) CONTAINS "japan"
RETURN count(f) AS total_folktales
    """
        )

        self.add_few_shot_example(
            "Who are Momotaro's friends?",
            """
MATCH (m:Character)-[r:RELATIONSHIP]->(a:Character)
WHERE toLower(m.name) CONTAINS "momotaro"
AND toLower(r.type) CONTAINS "friend"
RETURN a.name AS friend
    """
        )

        self.add_few_shot_example(
            "What happens after the event where Momotaro gives the dumpling to the dog?",
            """
MATCH (f:Folktale)-[:HAS_EVENT]->(e1:Event)-[:POST_EVENT]->(e2:Event)
WHERE toLower(e1.description) CONTAINS "momotaro"
AND toLower(e1.description) CONTAINS "dumpling"
AND toLower(e1.description) CONTAINS "dog"
RETURN e2.description AS ground_truth
LIMIT 1
    """
        )

        self.add_few_shot_example(
            "What is the genre of the 'The Birdcatcher' folktale?",
            """
MATCH (f:Folktale)-[:HAS_GENRE]->(g:Genre)
WHERE toLower(f.title) CONTAINS "birdcatcher"
RETURN g.name AS genre
    """
    )
        
        self.add_few_shot_example(
            "What are some characteristics of the tailor's wife, such as race, gender, and age group?",
            """
MATCH (a:Character)
WHERE toLower(a.name) CONTAINS "tailor's wife"

RETURN
    a.race AS race,
    a.gender AS gender,
    a.ageGroup AS ageGroup
LIMIT 1
"""
        )

        self.add_few_shot_example(
            "Where does Momotaro live?",
            """
MATCH (a:Character)-[:LIVES_IN]->(p:Place)
WHERE toLower(a.name) CONTAINS "momotaro"
RETURN p.name AS name
"""
        )

        self.add_few_shot_example(
    "What is the role of the tiger?",
    """
MATCH (a:Character)-[:HAS_ROLE]->(r)
WHERE toLower(a.name) CONTAINS "tiger's son"
RETURN r.name AS role
"""
        )


    def add_few_shot_example(self, question: str, cypher: str):
        """Añade un ejemplo few-shot."""
        self.few_shot_examples.append({
            "question": question,
            "cypher": cypher
        })

        self._build_chain()

    def _build_chain(self):
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{question}"),
                ("ai", "{cypher}"),
            ]
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=self.few_shot_examples,
        )

        # cypher_query = """
        # MATCH (f:Folktale)
        # RETURN f.title as title,
        #     f.url as url
        # """

        # results = self.neo4j.execute_query(cypher_query)

        # folktales = "\n".join(
        #     f"- {row['title']}: {row['url']}"
        #     for row in results
        # )

        system_prompt = f"""You are an expert at converting natural language questions into Cypher queries for Neo4j.

GRAPH SCHEMA:
{self.schema_str}

RULES:
1. Use only the node labels, relationship types and properties shown in the schema.
2. Output ONLY the Cypher query in the 'cypher' field. Do not include explanations or comments.
3. The query must be syntactically correct Neo4j Cypher.
"""        

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(template=system_prompt),
                few_shot_prompt,
                HumanMessagePromptTemplate.from_template(template="{question}")
            ]
        )
        self.chain = prompt | self.model.with_structured_output(CypherQuery)

    def generate_cypher(self, question: str) -> str:
        """
        Genera una query Cypher a partir de una pregunta en lenguaje natural.
        Uses structured output so the model is constrained to return only the
        Cypher string — no preamble, reasoning, or markdown wrapping.
        """
        result = self.chain.invoke({"question": question})
        result = cast(CypherQuery, result)

        cypher = result.cypher.strip()

        if not cypher.lower().startswith(("match", "call", "with")):
            raise ValueError(f"Invalid Cypher generated: {cypher}")
        
        return cypher

    def retrieve(self, question: str) -> tuple[str, list[dict[str, Any]]]:
        """
        Genera una query Cypher y la ejecuta.

        Returns:
            Tupla con (cypher_query, results)
        """
        cypher = self.generate_cypher(question)

        try:
            results = self.neo4j.execute_query(cypher)
            return cypher, results
        except Exception as e:
            print(f"Error executing Cypher: {e}")
            print(f"Generated query: {cypher}")
            return cypher, []
        
    def _remove_embedding(self, obj):
        if isinstance(obj, dict):
            return {
                k: self._remove_embedding(v)
                for k, v in obj.items()
                if k != "embedding"
            }
        elif isinstance(obj, list):
            return [self._remove_embedding(item) for item in obj]
        else:
            return obj
        
    def _run_tool(self, query: str) -> dict[str, Any]:
        cypher, results = self.retrieve(query)

        cleaned_results = self._remove_embedding(results)

        return {
            "cypher": cypher,
            "context": [str(r) for r in cleaned_results],
        }

    def as_tool(
        self,
        name: str = "text2cypher",
        description: Optional[str] = None,
        args_schema: type[BaseModel] = QueryInput,
    ) -> StructuredTool:
        return StructuredTool.from_function(
            name=name,
            description=description or (
                "Use this tool when the user asks factual, structured, or "
                "relationship-based questions about folktales stored in the "
                "Neo4j knowledge graph. Ideal for entities, connections, events, "
                "characters, nations, or explicit graph traversal queries. "
                "The tool converts natural language into Cypher, executes it, "
                "and returns both the generated query and query results."
            ),
            args_schema=args_schema,
            func=self._run_tool,
        )

