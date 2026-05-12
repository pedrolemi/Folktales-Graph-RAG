from typing import cast
from utils.models import get_llm
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

class QuestionDecomposition(BaseModel):
    sub_questions: list[str] = Field(
        default_factory=list,
        description="A list of atomic, self-contained sub-questions that are directly derived from the original question."
    )
    should_decompose: bool = Field(
        ...,
        description="True if the question should be broken into sub-questions."
    )

class QuestionDecomposer:
    def __init__(self):
        client = get_llm(0.0)

        system_prompt = """You are an expert at determining whether a folktale knowledge graph question requires decomposition into multiple dependent graph-query steps.

Your task is to decide whether the question requires breaking into sequential sub-queries over the graph structure.

1. should_decompose:

Set to true ONLY if:
- the question requires building an intermediate result set that is reused in a later step, OR
- the query requires chained constraints across multiple graph traversals that cannot be expressed as a single direct retrieval, OR
- the task involves filtering -> transformation -> re-filtering across different entity types or relationships

Examples of true decomposition:
- find a subset of characters -> then analyze their traits.
- retrieve events -> then order/filter based on derived property not directly stored.
- identify entities -> then compute relationships among those entities.

Set to false if ANY of the following apply:
- the question is a direct lookup of nodes or relationships (even if it returns many results).
- the question is a simple relational traversal (1-2 hops in the graph).
- the question is a COUNT, LIST or MATCH query over a single pattern.
- the question corresponds to a single Cypher query in the knowledge base.
- the question is about a single folktale narrative (events, characters, places, objects).
- the question asks to "list", "show", "retrieve", or "find" elements from one graph pattern.

IMPORTANT:
Do NOT decompose narrative or extraction queries, even if they return multiple results.

Example (DO NOT DECOMPOSE):
- "List all events in the folktale 'The Thieves and the Cock'".
- "Which characters appear in 'Momotaro'?".
- "What places are in 'Sharing Joy and Sorrow'?".

These are single graph retrieval operations.

2. sub_questions:

If should_decompose = true:
- produce ONLY minimal atomic graph sub-queries.
- each sub-question must represent a distinct dependency step in graph traversal.
- preserve original meaning.
- do NOT add reasoning, explanation or reformulation.
- do NOT split single Cypher patterns into artificial steps.

If should_decompose = false:
- return []

3. KEY PRINCIPLE (VERY IMPORTANT):

A folktale question is usually ONE query unless:
it explicitly requires intermediate reasoning over extracted graph results.

Do NOT assume complexity from multiple entities being returned.

4. OUTPUT FORMAT:

Return ONLY valid JSON:

{{
  "should_decompose": boolean,
  "sub_questions": list
}}
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            HumanMessagePromptTemplate.from_template("""
QUESTION:
{question}

Decide whether this folktale graph question requires decomposition into dependent sub-questions.
""")
        ])

        self.chain = prompt | client.with_structured_output(QuestionDecomposition)

    def run(self, question: str) -> list[str]:
        result = self.chain.invoke({"question": question})
        result = cast(QuestionDecomposition, result)

        if result.should_decompose:
            return result.sub_questions

        return [question]