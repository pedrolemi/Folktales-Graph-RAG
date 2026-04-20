from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from schemas.agent import AgentsLLM, Agent, PersonalityLLM
from schemas.place import Place
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
The ontological or categorical type of the character.
											
2. 'name' (string):
The name or label used to refer to the character within the story.

3. 'age_group' (string):
Must be exactly one of: 
- 'children', 'young', 'adult', 'senior'.

4. 'gender' (string):
Must be exactly one of:
- 'male' or 'female'.
											
5. 'description' (string):
A concise summary of the character's role, behavior and narrative function in the story.

6. 'role' (string):
Must be EXACTLY one of the following values:
{roles}

7. 'lives_in' (integer, optional):
- Integer index of the place from the list below:
{places}
- Omit this field if unknown or not explicitly stated in the text.

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

personality_prompt = ChatPromptTemplate.from_messages(
	[
		SystemMessagePromptTemplate.from_template(template='''You are an AI that assigns personality traits to a character based on a folktale.

Your job is to infer their personality using the Big Five model.

TRAITS (integer values from 0 to 100):
{traits}

GUIDELINES:
- Base your estimates ONLY on behaviors, actions, and descriptions present in the folktale.
- Do NOT invent traits not supported by the text.
- If evidence is limited, assign moderate values (40-60).
- Use extreme values (0-20 or 80-100) ONLY when strongly justified.

HEURISTICS:
- Heroes → higher agreeableness and conscientiousness.
- Villains → lower agreeableness, higher neuroticism or lower conscientiousness.
- Brave or active characters → higher extraversion.
- Reserved or passive characters → lower extraversion.
'''),

		HumanMessagePromptTemplate.from_template(template='''Assign Big Five personality traits to the following character based on the folktale.

Folktale:
{folktale}

Character:
Name: {character}
Description: {description}
''')
	]
)


def extract_agents(model: BaseChatModel, folktale: str, places: list[Place], role_dict: dict, traits_dict: dict):
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
	formatted_places = "\n".join(f"{i}. {place.name}" for i, place in enumerate(places))
	formatted_roles = format_hierarchy(role_dict)
	formatted_traits = "\n".join(f"- '{k}': {v}" for k, v in traits_dict.items())
	
	agent_chain = agent_prompt | model.with_structured_output(AgentsLLM)

	logger.info(agent_prompt.format(
		folktale=folktale,
		places=formatted_places,
		roles=formatted_roles,
	))

	agents_llm = agent_chain.invoke({
		"folktale": folktale,
		"places": formatted_places,
		"roles": formatted_roles,
	})
	
	agents_llm = cast(AgentsLLM, agents_llm)

	logger.info(agents_llm)

	personality_chain = personality_prompt | model.with_structured_output(PersonalityLLM)

	agents = []

	n_places = len(places)
	for agent_llm in agents_llm.agents:
		logger.info(personality_prompt.format(
			traits=formatted_traits,
			folktale=folktale,
			character=agent_llm.name,
			description=agent_llm.description
		))

		personality = personality_chain.invoke({
			"traits": formatted_traits,
			"folktale": folktale,
			"character": agent_llm.name,
			"description": agent_llm.description
		})

		personality = cast(PersonalityLLM, personality)

		if agent_llm.lives_in is not None and agent_llm.lives_in >= n_places:
			logger.warning(f"In agents: {agent_llm.name}. 'lives_in': {agent_llm.lives_in} is out of bounds. It must be less than {n_places}.")
			agent_llm.lives_in = None

		agent = Agent.from_llm(agent_llm, personality, places)
		agents.append(agent)
	
		logger.debug(f"Character: {agent}")

	return agents