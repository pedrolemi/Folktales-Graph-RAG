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

class PersonalityLLM(BaseModel):
	"""Big Five personality traits scored from 0 (low) to 100 (high)."""

	openness: int = Field(..., ge=0, le=100, description="Creativity, curiosity, openness to experience.")
	conscientiousness: int = Field(..., ge=0, le=100, description="Discipline, organization, reliability.")
	extraversion: int = Field(..., ge=0, le=100, description="Sociability, energy, assertiveness.")
	agreeableness: int = Field(..., ge=0, le=100, description="Kindness, cooperation, empathy.")
	neuroticism: int = Field(..., ge=0, le=100, description="Emotional instability, anxiety, reactivity.")

class Personality(BaseModel):
	openness: float
	conscientiousness: float
	extraversion: float
	agreeableness: float
	neuroticism: float
	
class AgentLLM(BaseModel):
	'''A character that appears within the folktale.'''

	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

	race: str = Field(..., description="The ontological category of the character.", min_length=1)

	name: str = Field(..., description="A unique identifier for the character as referenced in the story.", min_length=1)	
	
	age_group: Literal["children", "young", "adult", "senior"] = Field(..., description="Approximate age category of the character.")

	gender: Literal["male", "female"] = Field(..., description="Gender of the character.")

	description: str = Field(..., description="A concise summary of the character's role, behavior and narrative function.")
	
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

	personality: Personality

	lives_in: Optional[str] = Field(None, pattern=place_regex)

	@classmethod
	def from_llm(cls, llm_obj: AgentLLM, llm_personality: PersonalityLLM, places: list[Place]):
		personality = Personality(
			**{
				trait: getattr(llm_personality, trait) / 100.0
				for trait in type(llm_personality).model_fields
			}
		)

		lives_in_id = None
		if llm_obj.lives_in is not None:
			lives_in_id = places[llm_obj.lives_in].id

		return cls(
			race=llm_obj.race,
			name=llm_obj.name,
			age_group=llm_obj.age_group,
			gender=llm_obj.gender,
			description=llm_obj.description,
			role=llm_obj.role,
			personality=personality,
			lives_in=lives_in_id,
		)
