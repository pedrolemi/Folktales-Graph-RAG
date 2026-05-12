from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from utils.models import get_llm
from langchain_core.messages import HumanMessage, AIMessage
from schemas.event import EntitesLLM
from pydantic import ValidationError
from schemas.object import Object
from schemas.place import Place
from typing import cast
from loguru import logger

extractor_prompt = ChatPromptTemplate.from_messages(
	[
		SystemMessagePromptTemplate.from_template(template='''You are an expert information extraction system specialized in narrative analysis.

Your task is to extract structured narrative elements from a folktale segment with high precision.

You will be given:
- The title of the folktale.
- A segment of the story.
- A list of candidate objects.
- A list of candidate locations.

EXTRACTION RULES:
   
1. Objects:
	- Select only objects that are explicitly mentioned or clearly used in the segment.
	- Do NOT include irrelevant or background objects.
	- If no objects are present, return an empty list.

3. Location:
	- Select the SINGLE most relevant location where the segment occurs.
	- If multiple locations are mentioned, choose the primary setting.
	- If the location is unclear, choose the best option from context.
			
STRICT CONSTRAINTS:						
- You MUST ONLY select elements from the provided lists.
- You MUST return indices (integers), NOT names.
- Each index can appear ONLY ONCE.
- Do NOT invent or assume elements not present in the lists.
- If uncertain, prefer excluding rather than guessing.
'''),

		HumanMessagePromptTemplate.from_template(template='''Given a segment from the folktale and the complete list of objects from the story, select only the elements that are explicitly mentioned or clearly relevant to this segment.

Folktale title:
{title}

Objects:
{objects}

Locations:
{places}

Story segment:
{story_segment}
										   
Extract:
- Relevant objects (indices)
- Location (single index)

Return only the indices.
''')		
	]
)

def extract_event_elements(title: str, story_segment: str, objects: list[Object], places: list[Place], max_attempts: int = 5):
	"""
	Extrae los elementos de un evento a partir de un segmento de historia.

	Args:
		model (BaseChatModel): Modelo de lenguaje que realizará la extracción.
		event (EventMetadata): datos de evento.
		examples (list[EventExample]): Lista de ejemplos few-shot para guiar al modelo.
		max_attempts (int, optional): Número máximo de intentos para obtener una extracción válida.

	Returns:
		EventElements: Objeto con los elementos extraídos del evento.

	Raises:
		RuntimeError: Si después de `max_attempts` no se puede extraer un conjunto válido
		de elementos para el evento.
	"""

	model = get_llm(0.0)

	elements_prompt = ChatPromptTemplate.from_messages(
		extractor_prompt.messages + [
			MessagesPlaceholder(variable_name="messages")
		]
	)

	messages = []

	elements_chain = elements_prompt | model.with_structured_output(EntitesLLM)

	formatted_objects = "\n".join(f"{i}. {place.name}" for i, place in enumerate(objects))
	formatted_places = "\n".join(f"{i}. {place.name}" for i, place in enumerate(places))

	for attempt in range(max_attempts):
		try:
			logger.debug(f"Attempt {attempt + 1}/{max_attempts} for segment")

			logger.info(elements_prompt.format(
				title=title,
				objects=formatted_objects,
				places=formatted_places,
				story_segment=story_segment,
				messages=messages
			))

			elements = elements_chain.invoke({
				"title": title,
				"objects": formatted_objects,
				"places": formatted_places,
				"story_segment": story_segment,
				"messages": messages
			})

			elements = cast(EntitesLLM, elements)

			error_message = elements.validate_indices(objects, places)

			if not error_message:
				logger.debug(f"Valid extraction on attempt {attempt + 1}")

				return elements.get_ids(objects, places)
			
			logger.warning(error_message)

			messages.append(AIMessage(content=str(elements.model_dump())))
			messages.append(HumanMessage(content=error_message))

		except ValidationError as e:
			logger.warning(f"Validation error: {e}")
			messages.append(HumanMessage(content=str(e)))

	raise RuntimeError(
        f"Failed to extract valid EventElements after {max_attempts} attempts"
    )

