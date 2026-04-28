from typing import Any
from pydantic import BaseModel
from neo4j_manager import Neo4jManager
from utils.models import get_llm
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate, 
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from .base_retriever import BaseRetriever
from typing import cast

class CypherQuery(BaseModel):
    cypher: str


class Text2CypherRetriever(BaseRetriever):
    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j = neo4j_manager
        self.model = get_llm(0.0)
        self.few_shot_examples = []

        schema = self.neo4j.get_schema()
        self.schema_str = self.neo4j.format_schema(schema)
        self.schema_str = self.schema_str.replace("{", "{{").replace("}", "}}")

        print(self.schema_str)

        self._build_chain()

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

        system_prompt = (
            "You are an expert at converting natural language questions into Cypher queries for Neo4j.\n\n"
            f"Graph Schema:\n{self.schema_str}\n\n"
            "Rules:\n"
            "1. Use only the node labels, relationship types, and properties shown in the schema.\n"
            "2. Output ONLY the Cypher query in the 'cypher' field — no explanations, no markdown.\n"
            "3. The query must be syntactically correct Neo4j Cypher."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(template=system_prompt),
                # few_shot_prompt,
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
        