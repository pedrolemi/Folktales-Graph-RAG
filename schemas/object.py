from pydantic import Field, BaseModel, ConfigDict
from enum import StrEnum
from utils.regex_utils import object_regex
import uuid

class ObjectType(StrEnum):
	'''Enumeration of object categories that may appear in a folktale.'''

	MAGICAL_OBJECT = "magical_object"
	NATURAL_OBJECT = "natural_object"
	CRAFTED_OBJECT = "crafted_object"

class ObjectLLM(BaseModel):
	'''A distinct object that plays a role in the narrative of a folktale.'''

	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
	
	type: ObjectType = Field(..., description="The classification of the object, indicating its type.")

	name: str = Field(..., description="A descriptive name for the object, which identifies it within the folktale.")

	description: str = Field(..., description="A short sentence describing the object and its role in the folktale, using only information explicitly stated or clearly implied.")

class ObjectsLLM(BaseModel):
	'''Collection of all the relevant objects that appear within the folktale.'''

	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

	objects: list[ObjectLLM] = Field(
		...,
		description="List of all the objects necessary required for the coherent development and representation of the folktale."
	)

class Object(ObjectLLM):
    id: str = Field(
        default_factory=lambda: f"object_{uuid.uuid4().hex}",
		pattern=object_regex
    )

    @classmethod
    def from_llm(cls, llm_obj: ObjectLLM):
        return cls(
            **llm_obj.model_dump(),
        )