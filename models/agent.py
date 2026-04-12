from pydantic import ConfigDict, Field, BaseModel
from .place import Place
from utils.regex_utils import agent_regex, place_regex
from typing import Literal, Optional
from enum import StrEnum
import uuid

class Role(StrEnum):
	'''Enumeration used to classify the roles of characters within a story.

	This classification aligns with the spheres of action from Propp's narrative theory, categorizing roles
	into primary, secondary, and tertiary types based on their importance and function in the story.'''

	# Primary characters
	PRIMARY_CHARACTER = "primary_character"
	MAIN_CHARACTER = "main_character"
	HERO = "hero"
	ANTAGONIST = "antagonist"
	VILLAIN = "villain"
	FALSE_HERO = "false_hero"

	# Secondary characters
	SECONDARY_CHARACTER = "secondary_character"
	HELPER = "helper"
	MAGICAL_HELPER = "magical_helper"
	PRISONER = "prisoner"
	PRINCESS = "princess"
	QUEST_GIVER = "quest_giver"
	HERO_FAMILY = "hero_family" 

	# Tertiary characters
	TERTIARY_CHARACTER = "tertiary_character"

class Personality(BaseModel):
	"""Big Five personality traits scored from 0 (low) to 100 (high)."""

	openness: int = Field(..., ge=0, le=100, description="Creativity, curiosity, openness to experience.")
	conscientiousness: int = Field(..., ge=0, le=100, description="Discipline, organization, reliability.")
	extraversion: int = Field(..., ge=0, le=100, description="Sociability, energy, assertiveness.")
	agreeableness: int = Field(..., ge=0, le=100, description="Kindness, cooperation, empathy.")
	neuroticism: int = Field(..., ge=0, le=100, description="Emotional instability, anxiety, reactivity.")
	
class AgentLLM(BaseModel):
	'''A character that appears within the folktale.'''

	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

	race: str = Field(..., description="The ontological category of the character.")

	name: str = Field(..., description="A unique identifier for the character as referenced in the story.")	
	
	age_group: Literal["children", "young", "adult", "senior"] = Field(..., description="Approximate age category of the character.")

	gender: Optional[Literal["male", "female"]] = Field(None, description="Biological or narrative gender of the character, if explicitly stated.")

	description: str = Field(..., description="A concise summary of the character's role, behavior and narrative function.")

	personality: Personality = Field(..., description="A list of personality traits based on the Big 5 personality traits theory. These traits define the character's personality.")
	
	role: Role = Field(..., description="The narrative function of the character (e.g., hero, villain, helper).")

	lives_in: Optional[int] = Field(None, description="Index of the place in the story where the agent resides, referring to the provided places list. Must be omitted if unknown or not explicitly stated.")
	  
class AgentsLLM(BaseModel):
	"""A collection of the characters that appear within the folktale."""

	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

	agents: list[AgentLLM] = Field(
		..., 
		description="List of all the characters that appear in the story."
	)
	
class Agent(AgentLLM):
	id: str = Field(
		default_factory=lambda: f"agent_{uuid.uuid4().hex}",
		pattern=agent_regex
	)

	lives_in: Optional[str] = Field(None, pattern=place_regex)

	@classmethod
	def from_llm(cls, llm_obj: AgentLLM, places: list[Place]):
		data = llm_obj.model_dump()
		
		lives_in_idx = data.pop("lives_in", None)

		place_map = {i: place.id for i, place in enumerate(places)}

		lives_in_uuid = None
		if lives_in_idx is not None and lives_in_idx in place_map:
			lives_in_uuid = place_map[lives_in_idx]

		return cls(
			**data,
			lives_in=lives_in_uuid,
		)