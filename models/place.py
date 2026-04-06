from pydantic import Field, BaseModel, ConfigDict
from utils.regex_utils import snake_case_regex
from enum import StrEnum

class PlaceClass(StrEnum):
	'''Enumeration representing different types of places, organized within a hierarchical structure.'''
	
	# Natural places
	NATURAL_PLACE = "natural_place"
	MOUNTAIN = "mountain"
	FOREST = "forest"
	FIELD = "field"
	RIVER = "river"

	# Buildings
	BUILDING = "building"
	
	DWELLING = "dwelling"
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

	# Settlements
	SETTLEMENT = "settlement"
	VILLAGE = "village"
	TOWN = "town"
	CITY = "city"
	KINGDOM = "kingdom"

# PlaceClass = Annotated[Literal["natural_place", "mountain", "forest", "field", "dwelling", "castle", "palace", "house", "hut", "farmhouse", "tower", "community_building", "shop", "school", "tavern", "settlement", "village", "town", "city", "kingdom"], 
# 					   Field(..., description="The category of place this instant represents. Must be one of the predifined types.")]

class Place(BaseModel):
	'''A location that appears within the folktale.'''
	
	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

	class_name: PlaceClass = Field(..., description="The classification of the place, indicating its type.")
	
	instance_name: str = Field(
		...,
		description= "A descriptive, unique name for the place in snake_case format. This name must clearly identify the location within the context of the folktale.",
		examples=["hero_house", "royal_ballroom", "race_track", "straw_house", "stick_house", "brick_house", "near_forest"],
		pattern=snake_case_regex)
	
MAX_PLACES = 6
	
class Places(BaseModel):
	"Collection of locations that appear within the folktale."

	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

	places: list[Place] = Field(
		...,
		description="List of all locations explicitly mentioned in the folktale.",
		max_length=MAX_PLACES
	)