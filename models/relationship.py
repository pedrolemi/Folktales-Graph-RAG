from pydantic import BaseModel, Field
from typing import Literal, Optional
from .agent import Agent
from utils.regex_utils import agent_regex

class RelationshipLLM(BaseModel):
	'''Represents the relationship between two characters in a folktale.'''
	type: Literal["knows", "friend", "enemy", "family_member", "none"] = Field(..., description="Type of relationship between two characters.")
	description: str = Field(..., description="Brief justification based on evidence from the folktale")
	strength: int = Field(..., ge=0, le=5, description="Strength of the relationship.")

class Relationship(RelationshipLLM):
	type: Optional[str] = None
	source: str
	source_id: str = Field(..., pattern=agent_regex)
	target: str
	target_id: str = Field(..., pattern=agent_regex)
	strength: float

	@classmethod
	def from_llm(cls, llm_obj: RelationshipLLM, agents: list[Agent], agent_1: int, agent_2: int):
		source_agent = agents[agent_1]
		target_agent = agents[agent_2]		

		return cls(
			type=llm_obj.type,
			description=llm_obj.description,
			source=source_agent.name,
			source_id=source_agent.id,
			target=target_agent.name,
			target_id=target_agent.id,
			strength=llm_obj.strength/5.0
		)
