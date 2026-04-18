from pydantic import BaseModel, Field
from schemas.agent import Agent
from schemas.relationship import Relationship
from schemas.place import Place
from schemas.object import Object
from schemas.event import Event
from enum import StrEnum

class GenreClass(StrEnum):
	'''Enumeration of common genres of folktales, classified by typical themes, characters and narrative conventions.'''
	FABLE = "fable"
	FAIRY_TALE = "fairy_tale"
	LEGEND = "legend"
	MYTH = "myth"

class Genre(BaseModel):
	'''The genre classification of a foltkale, based on its theme, characters and narrative structure.'''
	genre: GenreClass = Field(..., description="The genre assigned to the folktale, chosen from a set of predefined categories.")

class Folktale(BaseModel):
	url: str
	nation: str
	title: str
	genre: GenreClass
	
	agents: list[Agent] = Field(default_factory=list)
	relationships: list[Relationship] = Field(default_factory=list)
	places: list[Place] = Field(default_factory=list)
	objects: list[Object] = Field(default_factory=list)
	events: list[Event] = Field(default_factory=list)
