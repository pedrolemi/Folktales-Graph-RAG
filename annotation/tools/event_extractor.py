from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, BaseMessage
from models.event import StorySegments, MAX_EVENTS, EventElements
from utils.format_utils import format_agents, format_objects, format_places
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic_core import ValidationError
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

human_prompt = HumanMessagePromptTemplate.from_template(template='''Given the following segment from the folktale **{title}** and the corresponding list of characters, objects and places in the entire story, select the most appropriate ones for this segment.

Characters (Agents):
{characters}

Objects:
{objects}

Locations (Places):
{places}

Story segment:
{story_segment}
										   
Your task is to identify:
1. Which characters (agents) are involved in this segment?
2. Which objects are relevant to this segment (if any)?
3. Which location does this event take place in?

Please return the index of the selected characters, objects and place from the lists provided above.
''')

system_prompt = SystemMessagePromptTemplate.from_template(template='''You are an AI designed to extract narrative elements from a segment of a folktale. Your task is to identify the characters, objects and location that are part of the segment provided.

For this task, you will be provided with:
- The full title of the folktale.
- A summary of the segment you should analyze to extract the elements.
- A complete list of possible characters, objects and places from the entire story.
											
Your job is to carefully analyze the segment and extract the correct elements based on the following instructions:
											
1. Characters (Agents):
	- Review the list of all characters in the story and identify the ones who are involved in this specific segment. There must be at least one character.
   
2. Objects:
   - Examine the list of all objects in the story and select the ones that are relevant to this segment. It’s possible that no object is mentioned in the event. If so, return an empty list.

3. Location (Place):
   - From the list of available locations, select the one where the segment occurs, based on the context.
											
Important Guidelines:
- Each element (character, object and location) can appear only once and should be represented by its index from the corresponding list.
- Be thorough in your analysis and ensure that you select only the relevant elements based on the context of the segment.
''')

def extract_event_elements(model: BaseChatModel, event: EventMetadata, examples: list[EventExample], max_attempts: int = 5):
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
	few_shot_examples: list[BaseMessage] = []

	for i, example in enumerate(examples):
		output_dict = example.output.model_dump()

		id = str(i + 1)

		human_message = human_prompt.format(
			title=example.title,
			characters=format_agents(example.agents),
			objects=format_objects(example.objects),
			places=format_places(example.places),
			story_segment=example.story_segment,
		)

		human_message = HumanMessage(human_message.content, name=f"example_user")
		few_shot_examples.append(human_message)

		ai_message = AIMessage(
			"",
			name="example_assistant",
			tool_calls=[{
				"name": EventElements.__name__,
				"args": output_dict,
				"id": id
			}]
		)
		few_shot_examples.append(ai_message)

		tool_message = ToolMessage("", tool_call_id=id)
		few_shot_examples.append(tool_message)

	# for ex in few_shot_examples:
	# 	ex.pretty_print()

	elements_prompt = ChatPromptTemplate.from_messages(
		[
			system_prompt,	
			MessagesPlaceholder(variable_name="few_shot_examples"),
			human_prompt,
			MessagesPlaceholder(variable_name="messages")
		]
	)

	messages: list[BaseMessage] = []

	elements_chain = elements_prompt | model.bind_tools([EventElements], tool_choice="any")

	formatted_agents = format_agents(event.agents)
	formatted_objects = format_objects(event.objects)
	formatted_places = format_places(event.places)

	for _ in range(max_attempts):
		logger.info(
			elements_prompt.format(
				story_segment=event.story_segment,
				places=formatted_places,
				objects=formatted_objects,
				characters=formatted_agents,
				title=event.title,
				few_shot_examples=few_shot_examples,
				messages=messages
			)
		)

		ai_message = elements_chain.invoke({
			"story_segment": event.story_segment,
			"places": formatted_places,
			"objects": formatted_objects,
			"characters": formatted_agents,
			"title": event.title,
			"few_shot_examples": few_shot_examples,
			"messages": messages
		})

		# print(ai_message)

		messages.append(ai_message)

		error_message = None

		if not ai_message.tool_calls:
			error_message = f"Respond using the {EventElements.__name__} function."
			logger.warning(error_message)
			human_message = HumanMessage(error_message)
			messages.append(human_message)
		else:
			args = ai_message.tool_calls[0]["args"]

			logger.debug(f"Event: {event.story_segment}\nElements: {args}")

			try:
				elements = EventElements.model_validate(args)
			except ValidationError as e:
				error_message = f"Validation error: {e}"

			if error_message is None:
				# elements.place = -1
				error_message = elements.validate_indices(event.agents, event.objects, event.places)

			if error_message:
				logger.warning(error_message)
				tool_message = ToolMessage(
					error_message,
					tool_call_id=ai_message.tool_calls[0]["id"]
				)
				messages.append(tool_message)
			else:
				logger.debug(f"Event: {event.story_segment}\nFinal elements: {elements}")
				return elements
			
	raise RuntimeError(
		f"Failed to extract event elements after {max_attempts} attempts. "
        f"Last error: {error_message} | "
        f"Segment: {event.story_segment}"
    )
