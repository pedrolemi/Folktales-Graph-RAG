from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from schemas.event import MAX_EVENTS
from loguru import logger

class ChunkParser(BaseOutputParser[list[str]]):
	def parse(self, text: str) -> list[str]:
		chunks = [c.strip() for c in text.split("<<<CHUNK>>>") if c.strip()]

		if not chunks:
			raise OutputParserException("No chunks found in output")

		return chunks
	
	@property
	def _type(self) -> str:
		return "chunk_parser"

event_prompt = ChatPromptTemplate.from_messages(
	[
		SystemMessagePromptTemplate.from_template(template='''You are an AI that divides a folktale into contiguous narrative segments.

Your goal is NOT to analyze or restructure the story, but to SPLIT it into meaningful parts.

DEFINITION OF AN EVENT:
An event is a meaningful change, action or development in the story involving characters, goals or situations.

INSTRUCTIONS:
- Split the story into AT MOST {max_events} events.
- Each part must preserve the ORIGINAL wording as much as possible.
- Do NOT rewrite, summarize, or reinterpret the text.
- Do NOT omit any information.
- Do NOT add any new information.
- Each part must be a CONTIGUOUS chunk of the original text.
- Together, all parts must reconstruct the FULL story exactly.

OUTPUT FORMAT:
Return only the chunks separated by the delimiter:

<<<CHUNK>>>

Do not include this delimiter inside any chunk.

<text chunk>
<<<CHUNK>>>
<text chunk>
<<<CHUNK>>>
<text chunk>

'''),

		HumanMessagePromptTemplate.from_template(template='''Folktale:
{folktale}
''')
	]
)

def extract_story_segments(model: BaseChatModel, folktale: str):
	"""
	Extrae los segmentos de una historia (eventos).

	Args:
		model (BaseChatModel): Modelo de lenguaje utilizado para la extracción de eventos.
		folktale (str): Texto completo del cuento o relato del cual se extraen los segmentos.

	Returns:
		list[str]: Lista de segmentos de la historia extraídos por el modelo.

	"""

	parser = ChunkParser()
	event_chain = event_prompt | model | parser

	chunks = event_chain.invoke({
		"folktale": folktale,
		"max_events": MAX_EVENTS
	})
	
	for i, event in enumerate(chunks):
		logger.debug(f"Event {i+1}: {event}")

	return chunks
