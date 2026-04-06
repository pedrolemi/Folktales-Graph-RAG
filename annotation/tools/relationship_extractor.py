from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from models.agent import Agent, Relationship
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger
from utils.regex_utils import relationship_regex
import re

relationship_prompt = ChatPromptTemplate.from_messages(
	[
		SystemMessagePromptTemplate.from_template(template='''You are an AI tasked with identifying the relationships between two characters in a folktale. Your task is to determine the most accurate relationship between the two characters based on the interactions described in the folktale. The relationship types you can choose from are:

- 'knows': The characters are acquaintances, but not close friends or enemies.
- 'friend': The characters share a close bond and help each other whenever needed.
- 'enemy': The characters have a rivalry or antagonism, typically involving conflict.
- 'family_member': The characters are related by blood, marriage, or adoption, regardless of the quality of their relationship.
- 'none': The characters do not meet or interact in the story.

KEY GUIDELINES:
- Focus on the nature of the interactions between the characters. Are they working together as allies, or is there tension between them?
- If there is ambiguity or if their relationship is unclear, default to 'none'.
- Remember that the context of the story is key, so consider only the interactions presented in the folktale.
'''),

		HumanMessagePromptTemplate.from_template(template='''Based on the following folktale, identify the relationships between '{agent_1}' and '{agent_2}'. Choose the most accurate relationship type, considering the context of their interactions in the story. Please follow the steps below to think through your answer, and then respond.

Think step by step:
1. Review the interactions: Carefully analyze the moments where '{agent_1}' and '{agent_2}' interact. What is the nature of their relationship at those moments?
2. Assess the relationship depth: Consider the strength of their connection. Are they just acquaintances or do they share a deeper bond such as friendship, family or rivalry?
4. Context matters: Think about their role within the story. Is their relationship central to the plot, or do they only have a brief interaction?
5. Make your decision: Based on the previous stepts, choose the most fitting relationship from these options: 'knows', 'friend', 'enemy', 'family_member' or 'none'.

Final answer format:
"Therefore, their relationship is: [chosen relationship]."

Folktale:
{folktale}
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
	relationship_chain = relationship_prompt | model

	relationships = []
	for i, agent_1 in enumerate(agents):
		start = i + 1
		for j, agent_2 in enumerate(agents[start:], start=start):
			# print(i, j)
			
			# logger.info(
			# 	relationship_prompt.format(
			# 		folktale = folktale,
			# 		agent_1 = agent_1.instance_name,
			# 		agent_2 = agent_2.instance_name
			# 	)
			# )

			ai_messsage = relationship_chain.invoke({
				"folktale": folktale,
				"agent_1": agent_1.instance_name,
				"agent_2": agent_2.instance_name
			})
			
			content = ai_messsage.content
			match = re.search(relationship_regex, content)

			relationship_type = "none"
			if match:
				relationship_type = match.group(1)

				if relationship_type in {"knows", "friend", "enemy", "family_member"}:
					relationship = Relationship(
						agent=i,
						other=j,
						relationship=relationship_type
					)

					relationships.append(relationship)
				else:
					relationship_type = "none"
			
			logger.debug(f"Relationship between '{agent_1.instance_name}' and '{agent_2.instance_name}': {relationship_type}\nReasoning: {content}")

	return relationships