from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from models.event import StorySegments, MAX_EVENTS
from typing import cast
from loguru import logger


event_prompt = ChatPromptTemplate.from_messages(
	[
		SystemMessagePromptTemplate.from_template(template='''You are an AI designed to break down a folktale into distinct parts, each corresponding to a different event or progression in the story. Your task is to carefully analyze the story, identify key events, and divide it into sections, while maintaining the integrity of the original content.

Each section must be SUMMARY of a major event or progression in the story. Do NOT add or omit essential details in the story, only divide and condense the original content into meaningful segments. Each segment must represent a complete event or progression, and when combined, all the segments must make up the full story without any loss of information.

The sections must follow a logical and sequential order. For instance, the first part should introduces an event, and each subsequent section should progress naturally from the previous one, maintaining a smooth narrative flow. Each part should be distinct but also contribute to the overall coherence and development of the story, ensuring that the progression of events feels continuous and well-structured.

Focus on the key elements within each section, explicitly naming the characters involved, the significant objects to the development of the event and the setting.
'''),

		HumanMessagePromptTemplate.from_template(template='''Given the following folktale, divide it into at most {max_events} parts, with each part is a summary of a key event or progression. Do not to omit or alter any information in any section.

Folktale:
{folktale}
''')
	]
)

def extract_story_segments(model: BaseChatModel, folktale: str):
	"""
	Extrae los segmentos de una historia (eventos).

	Args:
		model (BaseChatModel): Modelo de lenguaje utilizado para la extracción de eventos.
		folktale (str): Texto completo del cuento o relato del cual se extraen los segmentos.

	Returns:
		list[str]: Lista de segmentos de la historia extraídos por el modelo.

	"""

	event_chain = event_prompt | model.with_structured_output(StorySegments)

	events = event_chain.invoke({
		"folktale": folktale,
		"max_events": MAX_EVENTS
	})

	events = cast(StorySegments, events)
	
	for i, event in enumerate(events.segments):
		logger.debug(f"Event {i+1}: {event}")

	return events.segments