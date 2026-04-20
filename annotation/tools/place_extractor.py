from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from schemas.place import PlacesLLM, Place, MAX_PLACES
from langchain_core.language_models.chat_models import BaseChatModel
from typing import cast
from loguru import logger

place_prompt = ChatPromptTemplate.from_messages(
	[
		SystemMessagePromptTemplate.from_template(template='''You are an AI that extracts locations from a folktale.

Your task is to identify each location and assign it exactly one 'type' from the allowed list below and provide a suitable 'name' and 'description.

Your output MUST:
- Be valid JSON.
- Match the 'PlacesLLM' schema exactly.
- Contain no extra text, comments or explanations.

ALLOWED 'type' VALUES:
{places}

CLASS SELECTION RULES:
1. ALWAYS choose the MOST SPECIFIC type available.

3. Each place MUST have exactly ONE 'type'.
   - Do NOT combine multiple classes.
   - Do NOT repeat fields.

NAME RULES:
- Be descriptive but concise.
                                            
DESCRIPTION RULES:
- 'description' must be a short sentence describing the place as presented in the story.
- Use only information explicitly stated or clearly implied.
- Do NOT invent new details.

SELECTION RULES:
- Include ONLY locations explicitly mentioned in the story.
- Do NOT infer or assume locations.
- Choose the MOST SPECIFIC allowed 'type'.
- List each place only once.
'''),

		HumanMessagePromptTemplate.from_template(template='''Extract all locations explicitly mentioned in the folktale below.

Do not include more than {max_places} places.                                      

Folktale:
{folktale}
''')
	]
)

def extract_places(model: BaseChatModel, folktale: str, places_dict: dict):
   """
   Extrae los lugares presentes en un cuento.

   Args:
	  model (BaseChatModel): Modelo de lenguaje utilizado para la extracción de lugares.
	  folktale (str): Texto completo del cuento o relato del cual se extraen los lugares.
	  place_hierarchy (dict): Diccionario que define la jerarquía de lugares.

   Returns:
	  list[str]: Lista de lugares.

   """

   formatted = "\n".join(f"- '{k}': {v}" for k, v in places_dict.items())

   place_chain = place_prompt | model.with_structured_output(PlacesLLM)

   logger.info(place_prompt.format(
      folktale=folktale,
      max_places=MAX_PLACES,
      places=formatted
   ))

   places = place_chain.invoke({
      "folktale": folktale,
	   "max_places": MAX_PLACES,
	   "places": formatted
   })
   
   
   logger.debug(f"Places: {places}")

   places = cast(PlacesLLM, places)

   places = [
		Place.from_llm(place)
		for place in places.places
	]

   return places