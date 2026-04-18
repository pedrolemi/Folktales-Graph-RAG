from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from schemas.object import ObjectsLLM, Object
from langchain_core.language_models.chat_models import BaseChatModel
from typing import cast
from loguru import logger

object_prompt = ChatPromptTemplate.from_messages(
	[
		SystemMessagePromptTemplate.from_template(template='''You are an AI that extracts relevant objects to the story of a folktale.

Your task is to identify the minimum set of essential for the story to make sense (e.g., Cinderella's glass slipper). An object is any entity that is neither a character nor a place, but that plays a significant role in the story.

DO NOT extract:
- Humans or human-like characters.
- Anthropomorphic animals (animals that speak, reason or behave like humans).
- Locations, places, settings or environments (e.g., castles, forests, villages). 

Each object MUST include:
- Exactly one 'type' chosen from the allowed list.
- One appropriate 'name', that describes the object.

ALLOWED 'type' VALUES:
{objects}

CLASS SELECTION RULES:
1. ALWAYS choose the MOST SPECIFIC class available.
	- Example: If the object is a glass slipper, use 'crafted_object'.
	- Example: If the object is an apple, use 'natural_object'.

2. Each object MUST have exactly ONE 'type'.
	- Do NOT combine multiple classes.
	- Do NOT repeat fields.

NAME RULES:
- Be descriptive but concise.
						
DESCRIPTION RULES:
- 'description' must be a short sentence describing the object and its role in the story.
- Use only information explicitly stated or clearly implied.
- Do NOT invent new details.										
'''),

		HumanMessagePromptTemplate.from_template(template='''Given the folktale below, identify the essential objects that are required for the story to function or progress.

Rules:
- Include only objects that directly affect the plot, conflict or resolution.
- Exclude humans or human-like beings.
- Exclude anthropomorphic animals.
- Exclude locations, places, settings or environments.

Folktale:
{folktale}
''')
	]
)

def extract_objects(model: BaseChatModel, folktale: str, objects_dict: dict):
	"""
	Extrae los objetos presentes en un texto.

	Args:
		model (BaseChatModel): Modelo de lenguaje utilizado para la extracción de objetos.
		folktale (str): Texto completo del cuento o relato del cual se extraen los objetos.
		object_hierarchy (dict): Diccionario que define la jerarquía de objetos.

	Returns:
		list[str]: Lista de objetos.

	"""

	formatted = "\n".join(f"- '{k}': {v}" for k, v in objects_dict.items())

	object_chain = object_prompt | model.with_structured_output(ObjectsLLM)

	print(object_prompt.format(
		folktale=folktale,
		objects=formatted
	))

	objects = object_chain.invoke({
		"folktale": folktale,
		"objects": formatted
	})
	
	
	logger.debug(f"Objects: {objects}")

	objects = cast(ObjectsLLM, objects)

	objects = [
		Object.from_llm(obj)
		for obj in objects.objects
	]

	return objects