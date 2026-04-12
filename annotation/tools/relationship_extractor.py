from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from models.agent import Agent
from models.relationship import RelationshipLLM, Relationship
from langchain_core.language_models.chat_models import BaseChatModel
from itertools import combinations
from loguru import logger
from typing import cast

relationship_prompt = ChatPromptTemplate.from_messages(
	[
		SystemMessagePromptTemplate.from_template(template='''You are an AI tasked with identifying the relationships between two characters in a folktale. 
											
You must classify the relationship into EXACTLY ONE of the following types:
- 'knows': The characters are acquaintances, but not close friends or enemies.
- 'friend': The characters share a close bond and help each other whenever needed.
- 'enemy': The characters have a rivalry or antagonism, typically involving conflict.
- 'family_member': The characters are related by blood, marriage, or adoption, regardless of the quality of their relationship.
- 'none': The characters do not meet or interact in the story.

IMPORTANT RULES:
- Only use evidence from the folktale.
- Do NOT infer beyond what is explicitly or clearly implied.
- If the characters never interact or are unrelated in the story, use "none".
- If ambiguous, default to "knows".

You must also provide:
- A short justification based only on the story.
- A strength score from 1 to 5:
	1 = very weak / minimal interaction
	3 = moderate interaction
	5 = very strong / central relationship in the story
		
OUTPUT REQUIREMENTS (STRICT):
You MUST output structured data matching:
- type: one of ["knows", "friend", "enemy", "family_member", "none"]
- description: a concise explanation of evidence from the story
- strength: integer between 1 and 5
'''),

		HumanMessagePromptTemplate.from_template(template='''Based on the folktale below, determine the relationship between "{agent_1}" and "{agent_2}".

Folktale:
{folktale}

Return ONLY the structured result.
''')
	]
)

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
	relationship_chain = relationship_prompt | model.with_structured_output(RelationshipLLM)

	relationships = []
	for i, j in combinations(range(len(agents)), 2):
		agent_i = agents[i]
		agent_j = agents[j]

		relationship  = relationship_chain.invoke({
			"folktale": folktale,
			"agent_1": agent_i.name,
			"agent_2": agent_j.name
		})
		
		relationship = cast(RelationshipLLM, relationship)
		relationship = Relationship.from_llm(relationship, agents, i, j)
		relationships.append(relationship)
		
		logger.debug(
			f"Relationship: {agent_i.name} -> {agent_j.name} = {relationship.type} (strength={relationship.strength}, description={relationship.description})"
		)

	return relationships
