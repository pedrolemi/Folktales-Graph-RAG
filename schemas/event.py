from pydantic import BaseModel, ConfigDict, Field, field_validator
from utils.regex_utils import place_regex, object_regex, agent_regex, event_regex
from typing import Annotated
from .object import Object
from .place import Place
from .agent import Agent
from enum import StrEnum
import uuid

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

class EventAgentLLM(BaseModel):
	'''Represent an agent involved in a specific event within a folktale, along with their actions and relative importance.'''

	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

	id: int = Field(..., description="Index of the agent within the list of the agent for the entire folktale.")

	actions: list[Annotated[str, Field(min_length=1)]] = Field(
		...,
		description="List of actions, behaviors, or roles performed by the agent in this event. Each item is a single character.",
		min_length=1
	)

	importance: int = Field(
		..., 
		description="Integer score between 0 (minimal relevance) and 10 (critical role) indicating the importance of the agent in the event.",
		ge=1,
		le=10
	)

class EventAgentsLLM(BaseModel):
	'''Collection of all agents involved in a single event within a folktale.'''

	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

	agents: list[EventAgentLLM] = Field(
		...,
		description="List of all the agents participating in the event.",
		min_length=1,
	)

class EventAgent(EventAgentLLM):
	id: str = Field(..., pattern=agent_regex)
	importance: float

	@classmethod
	def from_llm(cls, llm_obj: EventAgentLLM, agents: list[Agent]):
		agent_id = agents[llm_obj.id].id

		return cls(
			id=agent_id,
			actions=llm_obj.actions,
			importance=llm_obj.importance/10
		)

class Event(BaseModel):
	id: str = Field(
        default_factory=lambda: f"event_{uuid.uuid4().hex}",
		pattern=event_regex
    )
	type: EventClass
	name: str
	description: str
	thoughts: list[str]

	agents: list[EventAgent] = Field(default_factory=list)
	objects: list[Annotated[str, Field(pattern=object_regex)]] = Field(default_factory=list)
	place: str = Field(..., pattern=place_regex)

MIN_EVENTS = 3
MAX_EVENTS = 15

class EntitesLLM(BaseModel):
	'''It represents the narrative elements involved in a specific segment of the story.'''

	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

	objects: list[int] = Field(default_factory=list, description="Indices of the objects that play a relevant role in this segment of the story.")
	place: int = Field(..., description="Index of the location where this segment of the story takes place.")

	@field_validator('objects', mode='after')
	@classmethod
	def validate_objects(cls, value: list[int]) -> list[int]:
		return list(set(value))

	def validate_indices(self, objects: list[Object], places: list[Place]):
		formatted_objects = "\n".join(f"{i}. {obj.name}" for i, obj in enumerate(objects))
		formatted_places = "\n".join(f"{i}. {place.name}" for i, place in enumerate(places))

		n_objects = len(objects)
		n_places = len(places)

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

	def get_ids(self, objects: list[Object], places: list[Place]):
		place_id = places[self.place].id

		objects_ids = [
			objects[idx].id
			for idx in self.objects
		]

		return objects_ids, place_id
