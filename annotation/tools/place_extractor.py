from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from models.place import Places, MAX_PLACES
from langchain_core.language_models.chat_models import BaseChatModel
from typing import cast
from loguru import logger
from utils.format_utils import format_hierarchy, format_classes

place_prompt = ChatPromptTemplate.from_messages(
	[
		SystemMessagePromptTemplate.from_template(template='''You are an AI that extracts locations from a folktale.

Your task is to identify each location and assign it exactly one 'class_name' from the allowed list below and invent a suitable 'instance_name'.

Your output MUST:
- Be valid JSON.
- Match the 'Places' schema exactly.
- Contain no extra text, comments or explanations.

ALLOWED 'class_name' VALUES:
- {places}

HIERARCHY (FOR REASONING):
{place_hierarchy}

CLASS SELECTION RULES:
1. ALWAYS choose the MOST SPECIFIC class available.
   - Example: If the location is a castle, use 'castle' rather than 'dwelling'.
   - Example: If the location is a village, use 'village' rather than 'settlement'.
   - Example: If the location is a forest, use 'forest' rather than 'natural_place'.

2. Use a PARENT class ONLY if:
   - The text refers to a place generically.
   - No specific subtype is stated or implied.

3. Each place MUST have exactly ONE 'class_name'.
   - Do NOT combine multiple classes.
   - Do NOT repeat fields.

INSTANCE NAME RULES:
- 'instance_name' must be written in snake_case.
- Use lowercase letters and underscores only.
- Be descriptive but concise.
- Do NOT include spaces, hyphens, or punctuation.
- Examples: 'hero_house', 'ogres_castle', 'nearby_forest', 'small_village'.

SELECTION RULES:
- Include ONLY locations explicitly mentioned in the story.
- Do NOT infer or assume locations.
- Choose the MOST SPECIFIC allowed 'class_name'.
- List each place only once.
'''),

		HumanMessagePromptTemplate.from_template(template='''Extract all locations explicitly mentioned in the folktale below.

For each location:
- Select the MOST SPECIFIC allowed 'class_name'.
- Create a concise, descriptive 'instance_name' written in snake_case.

Do not include more than {max_places} places.                                      

Folktale:
{folktale}
''')
	]
)

def extract_places(model: BaseChatModel, folktale: str, place_hierarchy: dict):
   """
   Extrae los lugares presentes en un cuento.

   Args:
	  model (BaseChatModel): Modelo de lenguaje utilizado para la extracción de lugares.
	  folktale (str): Texto completo del cuento o relato del cual se extraen los lugares.
	  place_hierarchy (dict): Diccionario que define la jerarquía de lugares.

   Returns:
	  list[str]: Lista de lugares.

   """
   formatted_hierarchy = format_hierarchy(place_hierarchy)
   formatted_classes = format_classes(place_hierarchy)

   place_chain = place_prompt | model.with_structured_output(Places)
   places = place_chain.invoke({
	  "folktale": folktale,
	  "max_places": MAX_PLACES,
	  "place_hierarchy": formatted_hierarchy,
	  "places": formatted_classes
   })
   
#    logger.info(
# 	  place_prompt.format(
# 		 folktale = folktale,
# 		 max_places = MAX_PLACES,
# 		 place_hierarchy = formatted_hierarchy,
# 		 places = formatted_classes
# 	  )
#    )
   
   logger.debug(f"Places: {places}")

   places = cast(Places, places)

   return places.places