from pydantic import BaseModel, Field, ConfigDict
from enum import StrEnum
from utils.regex_utils import snake_case_regex

class RoleClass(StrEnum):
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

class Role(BaseModel):
    '''A role that a character plays within the folktale.'''

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    class_name: RoleClass = Field(..., description="The role that a character fulfills in the story. This could be a protagonist, antagonist, or supporting role based on the character's function in the narrative.")
    instance_name: str = Field(
        ..., 
        description="A unique, descriptive identifier that further specifies the role within its class. This name should describe the role more precisely (e.g., 'main_hero', 'villain_minion'). It is not the character's name, but a categorization of their narrative function.",
        examples=["main_hero", "main_villain", "villain_minion", "hero_soulmate", "fairy_godmother_helper", "hero_rival", "hero_brother"],
        pattern=snake_case_regex
    )
