from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from models.folktale import Genre
from langchain_core.language_models.chat_models import BaseChatModel
from typing import cast
from loguru import logger

genre_prompt = ChatPromptTemplate.from_messages(
	[
		SystemMessagePromptTemplate.from_template(template='''
You are an AI tasked with identifying the genre of a given folktale. Analyze the folktale in detail and categorize it into one of the following genres:

You have the following options to choose from:
- 'fable': a short story, usually featuring animals as characters, that conveys a moral lesson.
- 'fairy_tale': a fantastical narrative involving magical elements, often with fairies, witches or magical creatures, and typically having a happy ending.
- 'legend': a traditional story that has it roots in history or is based on real events, often featuring heroic figures.
- 'myth': a story involving gods, goddesses or supernatural beings, usually explaining natural phenomena or cultural practices.
'''),

		HumanMessagePromptTemplate.from_template(template='''Read the following folktale carefully and determine its genre. Consider the characters, themes, and story structure before making a decision.

Folktale:
{folktale}
''')
	]
)

def extract_genre(model: BaseChatModel, folktale: str):
	"""
	Extrae el género de un cuento.

	Args:
		model (BaseChatModel): Modelo de lenguaje utilizado para la extracción del género.
		folktale (str): Texto completo del cuento o relato cuyo género se desea identificar.

	Returns:
		str: Género.

	"""
	genre_chain = genre_prompt | model.with_structured_output(Genre)
	genre = genre_chain.invoke({"folktale": folktale})
	
	logger.debug(f"Genre: {genre}")

	genre = cast(Genre, genre)

	return genre.genre