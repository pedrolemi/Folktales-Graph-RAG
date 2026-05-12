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
        client = get_llm(0.1)

        system_prompt = """You are an expert at detecting whether a question contains MULTIPLE INDEPENDENT INFORMATION REQUESTS that require sequential reasoning.
    
Your goal is to DECOMPOSE ONLY WHEN STRICTLY NECESSARY.

A question should be decomposed ONLY if:
- answering it requires FIRST identifying/filtering/selecting a set of entities/items,
AND THEN
- performing a SECOND DISTINCT operation on that resulting set.

GOOD examples (should_decompose = true):

Question:
"Among users who signed up this week, who has the highest score?"
Sub-questions:
- "Which users signed up this week?"
- "Which of those users has the highest score?"

Question:
"For books written by Tolkien, which one is the shortest?"
Sub-questions:
- "Which books were written by Tolkien?"
- "Which of those books is the shortest?"

Question:
"List the planets and explain Mars."
Sub-questions:
- "What are the planets?"
- "Explain Mars."

Question:
"Find the fastest algorithm and explain why it is fast."
Sub-questions:
- "Which algorithm is the fastest?"
- "Why is that algorithm fast?"

BAD examples (should_decompose = false):

- "What is Python?"
- "Explain recursion."
- "What is the capital of France?"
- "Why is the sky blue?"
- "How many users are there?"
- "Who invented the telescope?"

RULES:

1. should_decompose:
Set to true ONLY if:
- the question contains MULTIPLE DISTINCT OPERATIONS,
AND
- at least one operation depends on the output of another.

Otherwise set to false.

2. sub_questions:
If should_decompose = True:
- produce ONLY the MINIMAL set of atomic, self-contained sub-questions required to answer the original question.
- each sub-question must represent a necessary information dependency or distinct operation.
- preserve the original wording whenever possible.
- do not introduce new entities, constraints, assumptions or terminology.
- do not add explanations, reasoning steps, definitions or clarifications.
- do not number items.

If should_decompose = false:
- return an empty list.

4. OUTPUT FORMAT:
Return ONLY valid JSON that matches the schema.
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            HumanMessagePromptTemplate.from_template("""
QUESTION:
{question}

Determine whether the question requires decomposition.
""")
        ])

        self.chain = prompt | client.with_structured_output(QuestionDecomposition)

    def run(self, question: str) -> list[str]:
        result = self.chain.invoke({"question": question})
        result = cast(QuestionDecomposition, result)

        if result.should_decompose:
            return result.sub_questions

        return [question]