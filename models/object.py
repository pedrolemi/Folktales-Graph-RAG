from pydantic import Field, BaseModel, ConfigDict
from utils.regex_utils import snake_case_regex
from enum import StrEnum

class ObjectClass(StrEnum):
	'''Enumeration of object categories that may appear in a folktale.'''
	
	# Inanimate objects
	MAGICAL_OBJECT = "magical_object"
	NATURAL_OBJECT = "natural_object"
	CRAFTED_OBJECT = "crafted_object"

class Object(BaseModel):
	'''A distinct object that plays a role in the narrative of a folktale.'''

	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
	
	class_name: ObjectClass = Field(..., description="The classification of the object, indicating its type.")
	instance_name: str = Field(
		...,
		description="A descriptive, unique identifier for the object in snake_case. This name must clearly distinguish this object within the context of the folktale.",
		pattern=snake_case_regex,
		examples=["glass_slipper", "carriage", "ball_gown", "straw", "stick", "brick", "magic_sword", "ancient_parchment"]
	)

class Objects(BaseModel):
	'''Collection of all the relevant objects that appear within the folktale.'''

	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

	objects: list[Object] = Field(
		...,
		description="List of all the objects necessary required for the coherent development and representation of the folktale.")