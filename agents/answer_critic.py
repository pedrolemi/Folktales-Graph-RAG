from typing import cast
from utils.models import get_llm
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

class CritiqueResult(BaseModel):
    is_complete: bool = Field(description="Whether the answer fully addresses the question")
    is_faithful: bool = Field(description="Whether the answer is supported by the context")
    missing_info: list[str] = Field(default_factory=list, description="Missing information or follow-up questions")
    feedback: str = Field(description="Short explanation of the evaluation")

class AnswerCritic:
    def __init__(self):
        client = get_llm(0.0)

        system_prompt = """You are an expert at evaluating answers to questions based on provided context.

Your task is to determine:
1. Is the answer complete? (Does it fully address all parts of the question?)
2. Is the answer faithful? (Is it supported by the provided context?)
3. What information is missing, if any?

Respond ONLY with valid JSON in this format:
{{
    "is_complete": true/false,
    "is_faithful": true/false,
    "missing_info": ["additional question 1", "additional question 2"],
    "feedback": "brief explanation"
}}

If the answer is complete and faithful, missing_info should be an empty list."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            HumanMessagePromptTemplate.from_template(template="""Question: {question}

Context:
{context}

Answer: {answer}

Evaluate this answer.
""")
        ])

        self.chain = prompt | client.with_structured_output(CritiqueResult)


    def critique(self, question: str, context: list[str], answer: str) -> CritiqueResult:
        context_str = "\n\n".join(
            f"[{i + 1}] {c}" for i, c in enumerate(context)
        )

        result = self.chain.invoke({
            "question": question,
            "context": context_str,
            "answer": answer,
        })

        return cast(CritiqueResult, result)
    