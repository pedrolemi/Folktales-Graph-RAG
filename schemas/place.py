from pydantic import Field, BaseModel, ConfigDict
from utils.regex_utils import place_regex
from enum import StrEnum
import uuid

class PlaceType(StrEnum):
	'''Enumeration representing different types of places, organized within a hierarchical structure.'''
	
	MOUNTAIN = "mountain"
	FOREST = "forest"
	FIELD = "field"
	RIVER = "river"
	CASTLE = "castle"
	PALACE = "palace"
	HOUSE = "house"
	HUT = "hut"
	FARMHOUSE = "farmhouse"
	TOWER = "tower"
	COMMUNITY_BUILDING = "community_building"
	SHOP = "shop"
	SCHOOL = "school"
	TAVERN = "tavern"
	VILLAGE = "village"
	TOWN = "town"
	CITY = "city"
	KINGDOM = "kingdom"

class PlaceLLM(BaseModel):
	'''A location that appears within the folktale.'''
	
	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

	type: PlaceType = Field(..., description="The classification of the place, indicating its type.")
	
	name: str = Field(..., description= "A descriptive name for the place, which identifies it within the folktale.")

	description: str = Field(..., description="A short sentence describing the place as it appears in the folktale, using only information explicitly stated or clearly implied.")
	
MAX_PLACES = 6
	
class PlacesLLM(BaseModel):
	"Collection of locations that appear within the folktale."

	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

	places: list[PlaceLLM] = Field(
		...,
		description="List of all locations in the folktale.",
		max_length=MAX_PLACES
	)

class Place(PlaceLLM):
    id: str = Field(
        default_factory=lambda: f"place_{uuid.uuid4().hex}",
		pattern=place_regex
    )

    @classmethod
    def from_llm(cls, llm_obj: PlaceLLM):
        return cls(
            **llm_obj.model_dump(),
        )