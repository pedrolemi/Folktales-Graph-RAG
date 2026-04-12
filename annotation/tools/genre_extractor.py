from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from models.folktale import Genre
from langchain_core.language_models.chat_models import BaseChatModel
from utils.format_utils import format_agents
from typing import cast
from loguru import logger

genre_prompt = ChatPromptTemplate.from_messages(
	[
		SystemMessagePromptTemplate.from_template(template='''
You are an AI tasked with identifying the genre of a given folktale. Analyze the folktale in detail and categorize it into one of the following genres:

You have the following options to choose from:
{genres}
'''),

		HumanMessagePromptTemplate.from_template(template='''Read the following folktale carefully and determine its genre. Consider the characters, themes, and story structure before making a decision.

Folktale:
{folktale}
''')
	]
)

def extract_genre(model: BaseChatModel, folktale: str, genres: dict):
	"""
	Extrae el género de un cuento.

	Args:
		model (BaseChatModel): Modelo de lenguaje utilizado para la extracción del género.
		folktale (str): Texto completo del cuento o relato cuyo género se desea identificar.

	Returns:
		str: Género.

	"""

	formatted = "\n".join(f"- '{k}': {v}" for k, v in genres.items())

	genre_chain = genre_prompt | model.with_structured_output(Genre)

	print(genre_prompt.format(
		folktale=folktale,
		genres=formatted
	))

	genre = genre_chain.invoke({
		"folktale": folktale,
		"genres": formatted
	})

	
	logger.debug(f"Genre: {genre}")

	genre = cast(Genre, genre)

	return genre.genre