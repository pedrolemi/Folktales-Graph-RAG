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
	source_id: str = Field(..., pattern=agent_regex)
	target_id: str = Field(..., pattern=agent_regex)

	@classmethod
	def from_llm(cls, llm_obj: RelationshipLLM, agents: list[Agent], agent_1: int, agent_2: int):

		agent_map = {i: agent.id for i, agent in enumerate(agents)}

		source_id = agent_map[agent_1]
		target_id = agent_map[agent_2]
		
		type_lit = llm_obj.type
		resolved_type = None if type_lit == "none" else type_lit

		return cls(
			type=resolved_type,
			description=llm_obj.description,
			strength=llm_obj.strength,
			source_id=source_id,
			target_id=target_id,
		)