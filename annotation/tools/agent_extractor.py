from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from models.agent import AgentsLLM, Agent
from models.place import Place
from langchain_core.language_models.chat_models import BaseChatModel
from utils.format_utils import format_hierarchy
from loguru import logger
from typing import cast

agent_prompt = ChatPromptTemplate.from_messages(
	[
		SystemMessagePromptTemplate.from_template(template='''You are an AI that extracts characters from a folktale and returns structured data STRICTLY matching a predefined schema.
                                            
Your task is to identify ALL explicitly mentioned characters and describe them using the exact fields and formats below.
                                            
You MUST follow the schema exactly. Do not invent fields. Do not omit required fields.

FIELD DEFINITIONS:
                                        
1. 'race' (string):
Type of entity.

2. 'name' (sgtring):
The name or label used to refer to the character within the story.

3. 'age_group':
Choose one: 
- 'children', 'young', 'adult', 'senior'.

4. 'gender' (string):
- 'feminine' or 'masculine'.
- Omit if unknown.
                                            
5. 'personality':
Big Five traits as integers from 0 to 100:
- openness
- conscientiousness
- extraversion
- agreeableness
- neuroticism

Use reasonable estimates based on behavior:
- Heroes: high agreeableness, high conscientiousness
- Villains: low agreeableness, higher neuroticism or low conscientiousness

7. 'role':
Must be EXACTLY one of the following values:
{roles}

8. 'lives_in' (optional):
- Integer index of the place from the list below:
{places}
- If unknown, omit.

IMPORTANT RULES:
                                            
- Output MUST be a valid JSON object matching the schema.
- No explanations, no comments, no extra text.
- All fields must match expected types exactly.
- Ensure all numeric personality values are integers between 0 and 100.
- Do NOT hallucinate characters not explicitly mentioned.
                                            
OUTPUT FORMAT:

Return a JSON object with this structure:

{{
  "agents": [
    {{
      "race": "...",
      "name": "...",
      "age_group": "...",
      "gender": "...",
      "description": "...",
      "personality": {{
        "openness": 0,
        "conscientiousness": 0,
        "extraversion": 0,
        "agreeableness": 0,
        "neuroticism": 0
      }},
      "role": "...",
      "lives_in": 0
    }}
  ]
}}
'''),

		HumanMessagePromptTemplate.from_template(template='''Extract all characters and their characteristics explicitly mentioned in the folktale below.

Folktale:
{folktale}
''')
	]
)

def extract_agents(model: BaseChatModel, folktale: str, places: list[Place], role_dict: dict, personality_dict):
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
    formatted_places = "\n".join(f"- {i}. {place.name}" for i, place in enumerate(places))
    formatted_roles = format_hierarchy(role_dict)
    formatted_personality = "\n".join(f"- '{k}': {v}" for k, v in personality_dict.items())
    
    agent_chain = agent_prompt | model.with_structured_output(AgentsLLM)

    print(agent_prompt.format(
        folktale=folktale,
        places=formatted_places,
        roles=formatted_roles,
        personality=formatted_personality
    ))

    agents = agent_chain.invoke({
        "folktale": folktale,
        "places": formatted_places,
        "roles": formatted_roles,
        # "personality": formatted_personality
    })
    
    agents = cast(AgentsLLM, agents).agents

    n_places = len(places)
    for agent in agents:
        if agent.lives_in is not None and agent.lives_in >= n_places:
            logger.warning(f"In agents: {agent.name}. 'lives_in': {agent.lives_in} is out of bounds. It must be less than {n_places}.")
            agent.lives_in = None

    agents = [
		Agent.from_llm(obj, places)
		for obj in agents
	]
    
    logger.debug(f"Characters: {agents}")

    return agents