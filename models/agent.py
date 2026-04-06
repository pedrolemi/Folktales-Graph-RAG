from typing import Literal, Optional
from models.role import Role
from pydantic import ConfigDict, Field, BaseModel
from utils.regex_utils import snake_case_regex
from enum import StrEnum

RelationshipClass = Literal["knows", "friend", "enemy", "family_member"]

class Relationship(BaseModel):
	agent: int
	other: int
	relationship: RelationshipClass

class AgentClass(StrEnum):
	'''Enumeration to classify characters in the context of a folktale. This defines whether a character is an individual or a group, as well as its possible subcategories.'''
	
	# Individual agents
	INDIVIDUAL_AGENT = "individual_agent"
	HUMAN_BEING = "human_being"
	ANTHROPOMORPHIC_ANIMAL = "anthropomorphic_animal"
	MAGICAL_CREATURE = "magical_creature"
	
	# Group of agents
	GROUP_OF_AGENTS = "group_of_agents"
	
class Agent(BaseModel):
	'''A character that appears within the folktale.'''

	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
	
	class_name: AgentClass = Field(..., description="The type of character, whether it is an individual or a group, and its subcategories.")

	instance_name: str = Field(
		..., 
		description="A descriptive, unique identifier for the agent in snake_case format. This name must clearly identify the location within the context of the folktale.",
		examples=["cinderella", "stepmother", "stepsister_one", "stepsister_one", "prince", "fairy_godmother", "tortoise", "hare", "fox", "pig_one", "pig_two", "pig_three", "big_bad_wolf"],
		pattern=snake_case_regex)
	
	age_category: Literal["children", "young", "adult", "senior"] = Field(..., description="The age group to which the character belongs.")
	gender: Literal["male", "female"] = Field(..., description="The gender of the character, either 'male' or 'female'.")
	has_personality: list[Literal["sociable", "joyful", "active", "assertive", "eager", "depressive", "tense", "aggressive", "cold", "egocentric", "impersonal", "impulsive"]] = Field(
		..., 
		description="A list of personality traits based on the Big 5 personality traits theory. These traits define the character's personality.")
	name: Optional[str] = Field(
		None,
		description="The character's name.",
		examples=["Cinderella", "Lady Tremaine", "Anastasia", "Drizella", "Tortoise", "Haze", "Fox"],
		min_length=1
		# pattern=name_regex
	)
	
	has_role: Role = Field(..., description="The characters's role in the story, such as protagonist, antagonist or supporting character.")
	lives_in: Optional[int] = Field(None, description="The index of the place in the story where the agent resides, referring to the places array, if applicable.")
	  
class Agents(BaseModel):
	"""A collection of the characters that appear within the folktale."""

	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

	agents: list[Agent] = Field(..., description="List of all the characters that appear in the story.")
	