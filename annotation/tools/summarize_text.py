from typing import cast
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate



class SummaryResponse(BaseModel):
    """Model output for summarization."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid"
    )

    summary: str = Field(
        ...,
        description=(
            "Clean summarized text without duplicated or repeated information."
        ),
    )


summary_system_prompt = SystemMessagePromptTemplate.from_template(
    template="""
You are an expert summarization assistant.

Your task is to summarize the provided text while removing redundancy.

INSTRUCTIONS:
- Remove duplicated information.
- Remove repetitive sentences and ideas.
- Preserve the essential information and chronology.
- Keep the summary concise and coherent.
- Do NOT invent new information.
- Merge semantically equivalent statements into a single statement.

OUTPUT FORMAT:
{{
    "summary": "Clean summarized text"
}}
"""
)

summary_human_prompt = HumanMessagePromptTemplate.from_template(
    """
Text:
{text}
"""
)

summary_prompt = ChatPromptTemplate.from_messages([
    summary_system_prompt,
    summary_human_prompt,
])


def _summarize_text(
    model: BaseChatModel,
    text: str,
):
    """
    Generates a clean summary removing duplicated and repetitive content.

    Args:
        model (BaseChatModel):
            Language model used for summarization.

        text (str):
            Input text to summarize.

    Returns:
        str:
            Clean summarized text.
    """

    summary_chain = summary_prompt | model.with_structured_output(
        SummaryResponse
    )

    response = summary_chain.invoke({
        "text": text
    })

    response = cast(SummaryResponse, response)

    return response.summary