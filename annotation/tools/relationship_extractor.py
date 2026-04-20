from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from schemas.agent import Agent
from schemas.relationship import RelationshipLLM, Relationship
from langchain_core.language_models.chat_models import BaseChatModel
from itertools import combinations
from loguru import logger
from typing import cast

relationship_system_prompt = SystemMessagePromptTemplate.from_template(template='''You are an expert AI system that identifies relationships between two characters in a folktale.

You must classify the relationship into EXACTLY ONE of the following types:
- 'knows': The characters are acquaintances, but not close friends or enemies.
- 'friend': The characters share a close bond and help each other whenever needed.
- 'enemy': The characters have a rivalry or antagonism, typically involving conflict.
- 'family_member': The characters are related by blood, marriage or adoption, regardless of the quality of their relationship.
- 'none': The characters do not meet or interact in the story.

You must also provide:
- A short description of their relationship based only the sotry.
- A strength score of their relationship from 1 to 5:
	1: very weak / minimal interaction
	3: moderate interaction
	5: very strong / central relationship in the story

IMPORTANT RULES:
- Only use evidence from the folktale.
- Do NO infer relationships that are not explicitly or clearly implied.
- If the characters never interact or are unrelated in the story, use "none".
- If the evidence is ambiguous or weak, default to "none".
''')

context_human_prompt = HumanMessagePromptTemplate.from_template(template='''Extract ONLY factual interaction evidence between two characters.

Characters:
- {agent_1}: {agent_1_description}
- {agent_2}: {agent_2_description}

Folktale:
{folktale}

Return:
- direct interactions
- shared events
- co-occurrences

If none exist, say: "no interaction found"
''')

relationship_context_prompt = ChatPromptTemplate.from_messages([
	relationship_system_prompt,
	context_human_prompt
])

structured_prompt = HumanMessagePromptTemplate.from_template(template='''Based on the folktale and the extracted interaction context below, determine the relationship between two characters.

Characters:
- {agent_1}: {agent_1_description}
- {agent_2}: {agent_2_description}

Folktale:
{folktale}

Interaction context:
{context}

Return ONLY a structured JSON result.
No explanations.
No extra text.
''')

relationship_structured_prompt = ChatPromptTemplate.from_messages([
	relationship_system_prompt,
	structured_prompt
])

def extract_relationships(model: BaseChatModel, folktale: str, agents: list[Agent]):
	"""
	Extrae las relaciones entre agentes de un cuento.

	Args:
		model (BaseChatModel): Modelo de lenguaje utilizado para la extracción de relaciones.
		folktale (str): Texto completo del cuento.
		agents (list[Agent]): Lista de agentes presentes en el cuento.

	Returns:
		list[Relationship]: Lista de relaciones identificadas entre los agentes.
	"""

	context_chain = relationship_context_prompt | model
	classification_chain = relationship_structured_prompt | model.with_structured_output(RelationshipLLM)

	relationships = []
	for i, j in combinations(range(len(agents)), 2):
		agent_i = agents[i]
		agent_j = agents[j]

		ai_message = context_chain.invoke({
			"folktale": folktale,
            "agent_1": agent_i.name,
            "agent_2": agent_j.name,
            "agent_1_description": agent_i.description,
            "agent_2_description": agent_j.description
		})

		logger.info(ai_message)

		relationship = classification_chain.invoke({
			"folktale": folktale,
            "agent_1": agent_i.name,
            "agent_2": agent_j.name,
            "agent_1_description": agent_i.description,
            "agent_2_description": agent_j.description,
            "context": ai_message.content
		})
		
		relationship = cast(RelationshipLLM, relationship)

		if relationship.type != "none":
			relationship = Relationship.from_llm(relationship, agents, i, j)
			relationships.append(relationship)
			
			logger.debug(
                f"Relationship: {agent_i.name} -> {agent_j.name} | "
                f"type={relationship.type} | "
                f"strength={relationship.strength} | "
                f"description={relationship.description}"
            )

	return relationships
