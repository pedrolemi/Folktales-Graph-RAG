from typing import cast
from utils.models import get_llm
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

class Critique(BaseModel):
    """Structured evaluation of a response in relation to a question and its supporting context."""
    is_complete: bool = Field(description="True if the answer fully addresses every part of the question.")
    is_faithful: bool = Field(description="True if each statement in the answer is explicitly supported by the provided context.")
    missing_info: list[str] = Field(default_factory=list, description="A list of specific missing elements necessary to fully and precisely answer the question.")

class AnswerCritic:
    def __init__(self):
        client = get_llm(0.0)

        system_prompt = """You are an expert at evaluating answers to questions based on provided context.

You must evaluate an ANSWER using ONLY the provided CONTEXT.

Your are NOT ALLOWED to use outside knowledge.

EVALUATION RULES:

1. COMPLETENESS (is_complete)
- True ONLY if the answer fully addresses ALL the parts of the question.
- False if any sub-question, restriction or fact is missing.

2. FAITHFULNESS (is_faithful)
- True ONLY if each statement in the answer is explicitly supported by the context.
- False if:
    - the answer more information than that suggested by the context.
    - the answer paraphrases beyond what is supported.

3. MISSING INFORMATION (missing_info)
- List ONLY what is necessary to complete the answer.
- Express it as brief and specific missing elements.
- If nothing is missing, return an empty list.

CRITICAL RULES:
- Do NOT use external knowledge.
- Do NOT infer missing facts.
- Be strict and conservative in judgments.

OUTPUT FORMAT:
Return ONLY valid JSON matching the schema."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            HumanMessagePromptTemplate.from_template(template="""
QUESTION: 
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Evaluate strictly according to the rules.
""")
        ])

        self.chain = prompt | client.with_structured_output(Critique)


    def critique(self, question: str, context: list[str], answer: str) -> Critique:
        context_str = "\n\n".join(
            f"[{i + 1}] {c}" for i, c in enumerate(context)
        )

        result = self.chain.invoke({
            "question": question,
            "context": context_str,
            "answer": answer,
        })

        return cast(Critique, result)
    