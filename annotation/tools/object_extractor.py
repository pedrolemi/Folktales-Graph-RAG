from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from models.object import Objects
from langchain_core.language_models.chat_models import BaseChatModel
from typing import cast
from loguru import logger
from utils.format_utils import format_hierarchy, format_classes

object_prompt = ChatPromptTemplate.from_messages(
	[
		SystemMessagePromptTemplate.from_template(template='''You are an AI that extracts relevant objects to the story of a folktale.

Your task is to identify the minimum set of essential for the story to make sense (e.g., Cinderella's glass slipper). An object is any entity that is neither a character nor a place, but that plays a significant role in the story.

DO NOT extract:
- Humans or human-like characters.
- Anthropomorphic animals (animals that speak, reason or behave like humans).
- Locations, places, settings or environments (e.g., castles, forests, villages). 

Each object MUST include:
- Exactly one 'class_name' chosen from the allowed list.
- One appropriate 'instance_name', that describes the object.

ALLOWED 'class_name' VALUES:
- {objects}

HIERARCHY (FOR REASONING):
{object_hierarchy}

CLASS SELECTION RULES:
1. ALWAYS choose the MOST SPECIFIC class available.
	- Example: If the location is a glass slipper, use 'crafted_object' rather than 'inanimate_object'.
	- Example: If the location is an apple, use 'natural_obejct' rather than 'inanimate_object'.

2. Each object MUST have exactly ONE 'class_name'.
	- Do NOT combine multiple classes.
	- Do NOT repeat fields.

INSTANCE NAME RULES:
- 'instance_name' must be written in snake_case.
- Use lowercase letters and underscores only.
- Be descriptive but concise.
- Do NOT include spaces, hyphens, or punctuation.
- Examples: 'glass_slipper', 'ball_gown', 'oak_tree'.
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

def extract_objects(model: BaseChatModel, folktale: str, object_hierarchy: dict):
	"""
	Extrae los objetos presentes en un texto.

	Args:
		model (BaseChatModel): Modelo de lenguaje utilizado para la extracción de objetos.
		folktale (str): Texto completo del cuento o relato del cual se extraen los objetos.
		object_hierarchy (dict): Diccionario que define la jerarquía de objetos.

	Returns:
		list[str]: Lista de objetos.

	"""
	formatted_hierarchy = format_hierarchy(object_hierarchy)
	formatted_classes = format_classes(object_hierarchy)

	object_chain = object_prompt | model.with_structured_output(Objects)
	objects = object_chain.invoke({
		"folktale": folktale,
		"object_hierarchy": formatted_hierarchy,
		"objects": formatted_classes
	})
	
	# logger.info(
	#    object_prompt.format(
	#       folktale = folktale,
	#       object_hierarchy = formatted_hierarchy,
	#       objects = formatted_classes
	#    )
	# )
	
	logger.debug(f"Objects: {objects}")

	objects = cast(Objects, objects)

	return objects.objects