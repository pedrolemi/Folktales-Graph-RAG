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

        system_prompt = """
You are a planning assistant.

Your task is to determine whether a user question contains multiple independent intents that should be split into sub-questions.

The goal of decomposition is to separate:
- independently answerable requests.
- prerequisite retrieval steps.
- filtering + analysis operations.
- retrieval + explanation/generation combinations.
- sequential operations.

The goal of decomposition is to separate:
- independently answerable requests.
- prerequisite retrieval steps.
- filtering + analysis operations.
- retrieval + explanation/generation combinations.
- sequential operations.

A question SHOULD be decomposed when answering it naturally requires:
1. identifying or filtering a set of entities/items.
2. performing another operation on that resulting set.

This includes patterns like:
- filtering + reasoning.
- retrieval + summarization.
- identification + explanation.
- counting + description.
- selection + analysis.

Examples that SHOULD decompose:
- "Compare Python and Java."
- "List the planets and explain Mars."
- "Find the fastest algorithm and explain why it is fast."
- "Tell me who won and summarize the match."
- "Who founded OpenAI and when was it founded?"
- "List the files and explain the largest one."
- "Among users who signed up this week, who has the highest score?"
- "For books written by Tolkien, which one is the shortest?"

Examples that SHOULD NOT decompose:
- "What is Python?"
- "How many users are there?"
- "Who invented the telescope?"
- "Explain recursion."
- "What is the capital of France?"
- "Why is the sky blue?"

RULES:

1. DECISION (should_decompose):
Set should_decompose = true when the question contains one of them:
- multiple explicit requests
- an implicit multi-step structure where one operation depends on identifying/filtering entities before answering the final request.

Important:
A question may contain only ONE sentence and still require decomposition.

Questions involving:
- "where"
- "among"
- "for"
- "within"
- "in stories where..."
- filtered subsets
often require decomposition if they imply:
1. selecting a subset.
2. analyzing something about that subset.

2. STRICT NO-EXPANSION RULE:
- DO NOT define terms.
- DO NOT infer hidden meaning.
- DO NOT introduce new tasks.
- DO NOT break questions into reasoning steps.
- Use only information explicitly present in the user question.

3. SUB-QUESTIONS (sub_questions):
If should_decompose = True:
- produce atomic, self-contained sub-questions.
- preserve the original wording whenever possible.
- each sub-question must correspond to a distinct explicit part of the original question.
- do not add explanations or clarifications.
- do not number items.

4. MINIMAL DECOMPOSITION PRINCIPLE:
Decompose by INFORMATION DEPENDENCIES and USER INTENT.

Correct:
Question:
"In stories where there is a hero, what is the main test he must overcome?"

Good decomposition:
- "Which stories contain a hero?"
- "What is the main test the hero must overcome in those stories?"

Incorrect decomposition:
- "What is a hero?"
- "What is a test?"
- "Why must the hero overcome it?"

5. OUTPUT FORMAT:
Return ONLY valid JSON matching this schema:
{{
  "should_decompose": boolean,
  "sub_questions": string[]
}}
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            HumanMessagePromptTemplate.from_template("""
QUESTION:
{question}

Return ONLY the JSON output with:
- should_decompose (boolean)
- sub_questions (list of strings)

If no decomposition is needed, sub_questions MUST be an empty list.
""")
        ])

        self.chain = prompt | client.with_structured_output(QuestionDecomposition)

    def run(self, question: str) -> list[str]:
        result = self.chain.invoke({"question": question})
        result = cast(QuestionDecomposition, result)

        if result.should_decompose:
            return result.sub_questions

        return [question]