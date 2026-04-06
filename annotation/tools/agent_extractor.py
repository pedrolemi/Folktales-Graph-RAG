from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from models.agent import Agents, Agent
from models.place import Place
from langchain_core.language_models.chat_models import BaseChatModel
from utils.format_utils import format_hierarchy, format_places
from loguru import logger
import json
from typing import cast

agent_prompt = ChatPromptTemplate.from_messages(
	[
		SystemMessagePromptTemplate.from_template(template='''You are an AI that extracts characters from a folktale. For each character mentioned, extract the relevant details as described below. Each character must be described with the following attributes:

Exactly, you have to assign this information to each character:
- 'class_name': the agent type.
- 'instance_name': create a unique identifier that describes the character type, written in snake_case.
- 'age_category': the character's age group.
- 'gender': the character's gender, which can be either 'feminine' or 'masculine'.
- 'has_personality': a list of character traits based on the Big 5 personality traits theory.
- 'name' (optional): the character's name, if available.
- 'has_role': the character's role based on Prop's theory of the five spheres of action.
- 'lives_in' (optional): the place where the character lives, selected from the list below. Omit if the character has no specified location.

FIELD DEFINITIONS:

1. 'class_name':
Select the most specific type of character, from the following:
-  'individual_agent', 'human_being', 'anthropomorphic_animal', 'magical_creature', 'group_of_agents'.

2. 'instance_name':
- A unique identifier written in snake_case.
- Do not include spaces, hyphens, or punctuation.
- Descriptive but concise.
- Examples: 'main_hero', 'villain_minion'.

3. 'age_category':
Choose one: 'children', 'young', 'adult', 'senior'.

4. 'gender':
Choose either: 'feminine' or 'masculine'.

5. 'has_personality':
A list of character traits that reflect the character's behavior and personality, selected from the following options:
- 'sociable', 'joy', 'active', 'assertive', 'anxious', 'depressive', 'tense', 'aggressive', 'cold', 'egotism', 'impersonal', 'impulsive'.

6. 'name':
The character's name. Use title case, capitalizing the first letter of each word (e.g., 'Cinderella', 'Lady Tremaine', 'Haze'). If no name is provided, leave this field blank.

7. 'has_role':
The role the character plays, inspired by Prop's theory of the five spheres of action. Specify both:
- 'class_name': the type of role the character plays within the story, based on the hierarchy outlined below.
- 'instance_name': a more specific instance of the role, written in snake_case (e.g., 'main_hero', 'villain_minion'). Multiple characters can share the same 'instance_name' if they fulfill the exact same role within the story.

Role hierarchy:
{role_hierarchy}

8. 'lives_in':
Each character may live in one of the following locations:
{places}
If no location is specified, omit this field.

OUTPUT FORMAT:
Your output must be a valid JSON array, with one object per character. Each object must adhere strictly to the "Agent" schema.

Ensure your output contains NO extra text, comments, or explanations. Here's the required format:

{example}
'''),

		HumanMessagePromptTemplate.from_template(template='''Extract all characters and their characteristics explicitly mentioned in the folktale below. For each character, assign one role and one agent type based on the provided rules.

Folktale:
{folktale}
''')
	]
)

def extract_agents(model: BaseChatModel, folktale: str, example: list[Agent], places: list[Place], role_hierarchy: dict):
    """
    Extrae los agentes de un cuento utilizando un modelo de lenguaje con salida estructurada.

    Args:
        model (BaseChatModel):
            Modelo de lenguaje utilizado para la extracción de agentes.
        folktale (str):
            Texto del cuento o relato del cual se extraen los agentes.
        example (list[Agent]):
            Lista de agentes de ejemplo utilizada como referencia para el modelo.
        places (list[Place]):
            Lista de lugares válidos donde pueden residir los agentes.
        role_hierarchy (dict):
            Diccionario que define la jerarquía de roles entre los agentes.

    Returns:
        list[Agent]:
            Lista de agentes extraídos.
    """
    example_json = json.dumps(
        [agent.model_dump(mode="json") for agent in example],
        indent=4
    )
    formatted_places = format_places(places)
    formatted_hierarchy = format_hierarchy(role_hierarchy)
    
    agent_chain = agent_prompt | model.with_structured_output(Agents)
    agents = agent_chain.invoke({
        "folktale": folktale,
        "example": example_json,
        "places": formatted_places,
        "role_hierarchy": formatted_hierarchy
    })
    
    agents = cast(Agents, agents).agents

    n_places = len(places)
    for agent in agents:
        if agent.lives_in is not None and agent.lives_in >= n_places:
            logger.warning(f"In agents: {agent.instance_name}. 'lives_in': {agent.lives_in} is out of bounds. It must be less than {n_places}.")
            agent.lives_in = None
    
    logger.debug(f"Characters: {agents}")

    return agents