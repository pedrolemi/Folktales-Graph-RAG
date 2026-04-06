from pydantic import BaseModel, ConfigDict, Field, field_validator
from utils.regex_utils import snake_case_regex
from enum import StrEnum
from typing import Optional
from models.agent import Agent
from models.object import Object
from models.place import Place
from utils.format_utils import format_agents, format_objects, format_places

class EventClass(StrEnum):
	EVENT = "event"
	
	# Move
	MOVE = "move"

	# Setup
	SETUP = "setup"

	# Conflict
	CONFLICT = "conflict"
	INITIAL_SITUATION = "initial_situation"
	HERO_INTERDICTION = "hero_interdiction"
	VILLAINY = "villainy"
	FALSE_MATRIMONY = "false_matrimony"
	EXPULSION = "expulsion"
	KIDNAPPING = "kidnapping"
	MURDER = "murder"
	LACK = "lack"
	LACK_OF_BRIDE = "lack_of_bride"
	LACK_OF_MONEY = "lack_of_money"
	HERO_DEPARTURE = "hero_departure"
	STRUGGLE = "struggle"
	FIGHT = "fight"
	BRANDING = "branding"
	RECEIVE_MARK = "receive_mark"
	RECEIVE_INJURY = "receive_injury"
	CONNECTIVE_INCIDENT = "connective_incident"
	CALL_FOR_HELP = "call_for_help"
	DEPARTURE_DECISION = "departure_decision"
	VILLAIN_GAINS_INFORMATION = "villain_gains_information"

	# Preparation
	PREPARATION = "preparation"
	ABSENTATION = "absentation"
	BREAKING_INTERDICTION = "breaking_interdiction"
	ACQUISITION = "acquisition"
	GET_PRESENT = "get_present"
	GUIDANCE = "guidance"
	RETURN = "return"
	MAKE_CONTACT_WITH_ENEMY = "make_contact_with_enemy"
	MEDIATION = "mediation"
	TRICKERY = "trickery"

	# Beginning of counteraction
	BEGINNING_OF_COUNTERACTION = "beginning_of_counteraction"

	# Helper move
	HELPER_MOVE = "helper_move"
	RECEIPT_OBJECT = "receipt_object"
	LIQUIDATION_OF_LACK = "liquidation_of_lack"
	RELEASE_FROM_CAPTIVITY = "release_from_captivity"
	PURSUIT_AND_RESCUE = "pursuit_and_rescue"

	# False hero make unfounded claim
	FALSE_HERO_MAKE_UNFOUNDED_CLAIM = "false_hero_make_unfounded_claim"

	# Attempt at reconnaissance
	ATTEMPT_AT_RECONNAISSANCE = "attempt_at_reconnaissance"

	# Resolution
	RESOLUTION = "resolution"

	# Victory
	VICTORY = "victory"
	VILLAIN_DEFEATED = "villain_defeated"

	# Arrival
	ARRIVAL = "arrival"
	UNRECOGNISED_ARRIVAL = "unrecognised_arrival"
	HOME_ARRIVAL = "home_arrival"

	# Difficult task with soluiton
	DIFFICULT_TASK_WITH_SOLUTION = "difficult_task_with_solution"
	DIFFICULT_TASK = "difficult_task"
	SOLUTION_DIFFICULT_TASK = "solution_difficult_task"

	# Resolution function
	RESOLUTION_FUNCTION = "resolution_function"
	RECOGNITION = "recognition"
	PUNISHMENT = "punishment"
	REWARD = "reward"

	# Exposure fo villain
	EXPOSURE_OF_VILLAIN = "exposure_of_villain"

	# Transfiguration
	TRANSFIGURATION = "transfiguration"
	PHYSICAL_TRANSFORMATION = "physical_transformation"
	PSYCHOLOGICAL_TRANSFORMATION = "psychological_transformation"

	# Wedding or throne
	WEDDING_OR_THRONE = "wedding_or_throne"
	WEDDING = "wedding"
	GET_THRONE = "get_throne"

class Event(BaseModel):
	class_name: EventClass
	instance_name: str = Field(..., pattern=snake_case_regex)
	description: Optional[str] = None

	agents: list[int] = Field(default_factory=list)
	objects: list[int] = Field(default_factory=list)
	place: int

MIN_EVENTS = 3
MAX_EVENTS = 15

class StorySegments(BaseModel):
	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

	'''A collection representing the individual segments into which the story is divided.'''
	segments: list[str] = Field(
		..., 
		description="A list of textual segments, each representing a distinct event or part of the story.",
		max_length=MAX_EVENTS)
	
class EventElements(BaseModel):
	'''It represents the narrative elements involved in a specific segment of the story.'''

	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

	agents: list[int] = Field(
		..., 
		description="Indeces of the characters involved in the segment of the story.",
		min_length=1
	)
	objects: list[int] = Field(default_factory=list, description="Indices of the objects that play a relevant role in this segment of the story.")
	place: int = Field(..., description="Index of the location where this segment of the story takes place.")

	@field_validator('agents', mode='after')  
	@classmethod
	def validate_agents(cls, value: list[int]) -> list[int]:
		return list(set(value))
	
	@field_validator('objects', mode='after')  
	@classmethod
	def validate_objects(cls, value: list[int]) -> list[int]:
		return list(set(value))
	
	def validate_indices(self, agents: list[Agent], objects: list[Object], places: list[Place]):
		formatted_agents = format_agents(agents)
		formatted_objects = format_objects(objects)
		formatted_places = format_places(places)

		n_agents = len(agents)
		n_objects = len(objects)
		n_places = len(places)

		for agent_idx in self.agents:
			if agent_idx < 0 or agent_idx >= n_agents:
				content = (
					f"Agent index {agent_idx} is out of bounds. It must be between 0 and {n_agents - 1}. "
					f"Please choose a valid character from the list below, or remove it if no longer needed:\n"
					f"{formatted_agents}"
				)
				return content

		for object_idx in self.objects:
			if object_idx < 0 or object_idx >= n_objects:
				content = (
					f"Object index {object_idx} is out of bounds. It must be between 0 and {n_objects - 1}. "
					f"Please choose a valid object from the list below, or remove it if no longer needed:\n"
					f"{formatted_objects}"
				)
				return content

		if self.place < 0 or self.place >= n_places:
			content = (
				f"Place index is out of bounds. It must be between 0 and {n_places - 1}. "
				f"Please choose a valid place from the list below:\n"
				f"{formatted_places}"
			)
			return content
		
		return None

class EventMetadata(BaseModel):
	title: str
	agents: list[Agent]
	objects: list[Object]
	places: list[Place]
	story_segment: str

class EventExample(EventMetadata):
	output: EventElements