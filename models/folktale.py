from pydantic import BaseModel, Field, model_validator
from models.agent import Agent, Relationship
from models.place import Place
from models.object import Object
from models.event import Event
from enum import StrEnum
from typing import Optional
from typing_extensions import Self

class GenreClass(StrEnum):
	'''Enumeration of common genres of folktales, classified by typical themes, characters and narrative conventions.'''
	FABLE = "fable"
	FAIRY_TALE = "fairy_tale"
	LEGEND = "legend"
	MYTH = "myth"

class Genre(BaseModel):
	'''The genre classification of a foltkale, based on its theme, characters and narrative structure.'''
	genre: GenreClass = Field(..., description="The genre assigned to the folktale, chosen from a set of predefined categories.")

class AnnotatedFolktale(BaseModel):
	uri: Optional[str] = None
	nation: Optional[str] = None
	title: str
	has_genre: GenreClass
	
	agents: list[Agent] = Field(default_factory=list)
	relationships: list[Relationship] = Field(default_factory=list)
	places: list[Place] = Field(default_factory=list)
	objects: list[Object] = Field(default_factory=list)
	events: list[Event] = Field(default_factory=list)

	@model_validator(mode='after')
	def check_folktale(self) -> Self:
		n_agents = len(self.agents)
		n_objects = len(self.objects)
		n_places = len(self.places)

		for idx, relationship in enumerate(self.relationships):	
			if relationship.agent < 0 or relationship.agent >= n_agents:
				raise ValueError(f"In relationships[{idx}]: agent index {relationship.agent} is out of bounds.")

			if relationship.other < 0 or relationship.other >= n_agents:
				raise ValueError(f"In relationships[{idx}]: other index {relationship.other} is out of bounds.")

		for idx, agent in enumerate(self.agents):
			if agent.lives_in is not None:
				if agent.lives_in < 0 or agent.lives_in >= n_places:
					raise ValueError(f"In agents[{idx}] ({agent.instance_name}): lives_in index {agent.lives_in} is out of bounds.")
			
		for event_idx, event in enumerate(self.events):
			for agent_idx in event.agents:
				if agent_idx < 0 or agent_idx >= n_agents:
					raise ValueError(f"In events[{event_idx}] ({event.instance_name}): agent index {agent_idx} is out of bounds.")
				
			for object_idx in event.objects:
				if object_idx < 0 or object_idx >= n_objects:
					raise ValueError(f"In events[{event_idx}] ({event.instance_name}): object index {object_idx} is out of bounds.")

			place_idx = event.place	
			if place_idx < 0 or place_idx >= n_places:
				raise ValueError(f"In events[{event_idx}] ({event.instance_name}): place index {place_idx} is out of bounds.")

		return self