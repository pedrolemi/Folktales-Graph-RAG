from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from schemas.agent import Agent
from schemas.event import EventAgentsLLM, EventAgent
from pydantic import BaseModel, Field, ConfigDict
from collections import Counter
from typing import Optional, cast
from loguru import logger

class Response(BaseModel):
	'''Model output for a single-option classification task.'''

	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
	
	thinking: str = Field(
		...,
		description=(
			"Brief justification explaining why the selected option best matches the input text."
		),
		examples=[
			"The text sets up the hero's normal life and environment before any conflict occurs.",
			"The antagonist performs harmful acts that create obstacles for the hero.",
			"The hero is forced out of a safe place due to circumstances beyond their control.",
			"There is a direct confrontation involving physical combat."
		]
	)

	response: int = Field(
		...,
		description=(
			"Zero-based index of the selected option from the provided list."
		),
		ge=0
	)

type_system_prompt = SystemMessagePromptTemplate.from_template(template="""You are an expert narrative analysis assistant. Your task is to classify the text of a narrative evento into the most specific type within a closed list of options.

INSTRUCTIONS:
- Select exactly ONE option from the list.
- Each option is identified by its index number (0, 1, 2, ...).
- The answer MUST be a valid index within the bounds of the provided list of options.
- NEVER return an index < 0 or >= number of options.

DECISION GUIDELINES:
- Prefer the MOST SPECIFIC and CONCRETE option available.
- Only choose general options if no specific match exists.
- Use all available context to improve classification accuracy.
																
REASONING GUIDELINES:
- Keep "thinking" concise.
- Base the justification strictly on the evidence from the text and context.

AVAILABLE OPTIONS:
{options}

PREVIOUS THOUGHTS:
{previous_thought}

OUTPUT FORMAT:
{{
	"thinking": "Brief justification based on the text",
  	"response": int
}}
""")

type_human_prompt = HumanMessagePromptTemplate.from_template(
	"""Event text:
{event}

Past story (previous narrative context):
{past_story}

Event position in the narrative:
This fragment is event {event_index}/{total_events}.
Use this to understand narrative progression (e.g., setup, escalation, climax, resolution).
"""
)

type_prompt = ChatPromptTemplate.from_messages([
	type_system_prompt,
	type_human_prompt,
])

def _build_options_prompt(node: dict, self_name: Optional[str] = None):
	"""
	Construye un prompt de opciones basado en un nodo y sus hijos, 
	devolviendo tanto la representación en texto como la lista de opciones.

	Args:
		node (dict): Diccionario que representa un nodo.
		self_name (Optional[str]): Nombre opcional para incluir el nodo actual como una opción adicional.

	Returns:
		tuple[str, list]: 
			- str: Texto con cada opción numerada y su descripción, lista para mostrar al usuario.
			- list: Lista de tuplas (node_id, description) correspondientes a cada opción.
	"""
	options = []
	
	for child_id, info in node.get("children", {}).items():
		options.append((child_id, info["description"]))

	if self_name:
		options.append((self_name, node["description"]))

	lines = [
		f"{idx}. {node_id}: {description}" 
		for idx, (node_id, description) in enumerate(options)
	]

	return "\n".join(lines), options

def _build_options_prompt_by_list(options: list[tuple]):
	"""
	Construye un prompt de opciones a partir de una lista de tuplas (node_id, description).
	Args:
		options (list[tuple[str, str]]): Lista de tuplas donde cada tupla contiene:
			- node_id (str): Identificador de la opción
			- description (str): Descripción de la opción

	Returns:
		str: Texto con cada opción numerada y su descripción, separadas por saltos de línea.
	"""
	lines = [
		f"{idx}. {node_id}: {description}" 
		for idx, (node_id, description) in enumerate(options)
	]
	return "\n".join(lines)

def _extract_event(model: BaseChatModel, folktale_event: str, past_story: str, event_index: int, total_events: int, options: str, previous_thoughts: list[str]):
	"""
	Extrae un evento relevante de exto utilizando un modelo de lenguaje.

	Args:
	model (BaseChatModel): Modelo de lenguaje utilizado para la extracción del evento.
	folktale_event (str): Texto.
	options (str): Lista de opciones formateadas
	previous_thought (str, opcional): Pensamientos o contexto previo del modelo

	Returns:
	tuple[str, str]:
		- response (str): Evento seleccionado o generado por el modelo.
		- thinking (str): Razonamiento del modelo sobre la selección.

	"""
	type_chain = type_prompt | model.with_structured_output(Response)

	# print(type_prompt.format(
	# 	options=options,
	# 	event=folktale_event,
	# 	past_story=past_story,
	# 	event_index=event_index,
	# 	total_events=total_events,
	# 	previous_thought="\n".join(f"- {thought}" for thought in previous_thoughts)
	# ))

	response = type_chain.invoke({
		"options": options,
		"event": folktale_event,
		"past_story": past_story,
		"event_index": event_index,
		"total_events": total_events,
		"previous_thought": "\n".join(f"- {thought}" for thought in previous_thoughts)
	})

	response = cast(Response, response)

	return response.response, response.thinking

def hierarchical_event_classification(model: BaseChatModel, event_index: int, story_segments: list[str], taxonomy_tree: dict, n_rounds: int = 3, verbose: bool = False):
	"""
	Clasifica un evento usando una taxonomía jerárquica con descripciones.

	Args:
		model: Instancia de BaseChatModel.
		folktale_event: Texto del evento a clasificar.
		taxonomy_tree: Diccionario de taxonomía.
		n_rounds: Número de veces a preguntar al LLM por cada nivel.

	Returns:
		tuple[str, str]: (evento final elegido, justificación final)
	"""

	def log(msg: str, indent_level: int = 0):
		if verbose:
			indent = " " * indent_level
			print(f"{indent}{msg}")

	event_text = story_segments[event_index]
	past_story = "\n".join(story_segments[:event_index])
	total_events = len(story_segments)

	current_nodes = taxonomy_tree["function"]["children"]
	previous_event = None
	final_thoughts = []

	options_str, options_list = _build_options_prompt(taxonomy_tree["function"])
	level = 0

	result_event = None
	finished = False

	log("=== Hierarchical classification start ===")
	log(f"Event: {event_text}")

	while current_nodes and not finished:
		votes = []
		thoughts = []

		log(f"\n--- Nivel {level} ---", level)
		log("Options:", level)
		log(options_str, level + 1)

		# Preguntar al LLM n_rounds veces
		for i in range(n_rounds):
			event, thinking = _extract_event(
				model=model,
				folktale_event=event_text,
				past_story=past_story,
				event_index=event_index,
				total_events=total_events,
				options=options_str,
				previous_thoughts=final_thoughts
			)

			if 0 <= event < len(options_list):
				votes.append(event)
				thoughts.append(thinking)

				log(f"\nRound ({i + 1}/{n_rounds})", level + 1)
				log(f"Proposed: {options_list[event]}", level + 2)
				log(f"Thought: {thinking}", level + 2)
			else:
				log(f"\nRound ({i + 1}/{n_rounds})", level + 1)
				log(f"Proposed: OUT_OF_RANGE")
				log(f"Thought: {thinking}", level + 2)

		if not votes:
			log("No valid votes. Ending.", level)
			finished = True
			result_event = previous_event
			continue
		
		# Voto por mayoría
		vote_count = Counter(votes)
		max_freq = max(vote_count.values())
		most_frequent = [v for v, c in vote_count.items() if c == max_freq]

		log("Vote distribution:", level)
		for v, c in vote_count.items():
			log(f"- {options_list[v]}: {c}", level + 1)

		winning_event_idx = most_frequent[0]

		if len(most_frequent) > 1:
			log("Tie detected", level)

			selected = [options_list[i] for i in most_frequent]
			selected_str = _build_options_prompt_by_list(selected)

			log("Tie options:", level)
			log(selected_str, level + 1)

			event, thinking = _extract_event(
				model=model,
				folktale_event=event_text,
				past_story=past_story,
				event_index=event_index,
				total_events=total_events,
				options=selected_str,
				previous_thoughts=final_thoughts
			)

			log("Tie break decision:", level)
			log(f"Index: {event}", level + 1)
			log(f"Thought: {thinking}", level + 1)

			if not (0 <= event < len(selected)):
				log("Invalid tie-break. Ending.", level)
				result_event = previous_event
				finished = True
				continue
			
			winning_event_idx = most_frequent[event]

		final_thoughts.extend(
			thoughts[i] for i, v in enumerate(votes) if v == winning_event_idx
		)
		
		winning_event = options_list[winning_event_idx][0]

		log(f"Winning event: {winning_event}", level)

		# Se termina si el evento ganador es el padre
		if winning_event == previous_event:
			log("Repetead event detected. Ending.", level)
			result_event = winning_event
			finished = True
			continue
		
		# Se termina si se trata de un evento que se encuentra en un nodo hoja
		if not current_nodes[winning_event]["children"]:
			log("Leaf node reached. Stopping.", level)
			result_event = winning_event
			finished = True
			continue

		log(f"Descending into: {winning_event}", level)	
		
		options_str, options_list = _build_options_prompt(
			current_nodes[winning_event], 
			winning_event
		)

		current_nodes = current_nodes[winning_event]["children"]
		previous_event = winning_event
		level += 1

	log("\n=== Classification finished ===")
	log(f"Final event: {result_event}")
	log("Supporting thoughts:")
	for t in final_thoughts:
		log(f"- {t}", 2)

	return previous_event, final_thoughts

name_system_prompt = SystemMessagePromptTemplate.from_template(
	template="""You are an expert narrative annotation assistant.
Your task is to generate a concise for a narrative event.

INSTRUCTIONS:
- Produce a SHORT label (3-6 words).
- The name must capture the core action or outcome of the event.
- Prefer "verb + object" or "key event description" style.
- Do NOT repeat or paraphrase the full event text.
- Do NOT include unnecessary details.
- Do NOT include explanations.

OUTPUT FORMAT:
Return ONLY the event name as a single line.
"""
)

name_human_prompt = HumanMessagePromptTemplate.from_template(
	template="""Event type:
{event_type}

Event text:
{event_text}

Reasoning:
{thinking}
"""
)

name_prompt = ChatPromptTemplate.from_messages([
	name_system_prompt,
	name_human_prompt,
])

def extract_event_name(model: BaseChatModel, event_type: str, event_text: str, thoughts: list[str]):
	"""
	Extrae el nombre de la instancia de un evento a partir de su tipo y descripción.

	Args:
		model (BaseChatModel): Modelo de lenguaje utilizado para la extracción.
		event_type (str): Tipo del evento.
		event_text (str): Texto del evento del folktale.
		thinking (str, optional): Razonamiento previo que puede guiar la selección de la instancia. 

	Returns:
		str: Nombre de la instancia del evento identificada por el modelo.
	"""
	parser = StrOutputParser()
	name_chain = name_prompt | model | parser

	print(name_prompt.format(
		event_type=event_type,
		event_text=event_text,
		thinking="\n".join(f"- {thought}" for thought in thoughts)
	))

	result = name_chain.invoke({
		"event_type": event_type,
		"event_text": event_text,
		"thinking": "\n".join(f"- {thought}" for thought in thoughts)
	})

	# print(result)
	
	logger.debug(f"Event name: {result}")

	return result

event_agent_system_prompt = SystemMessagePromptTemplate.from_template(
	template="""You are an expert information extraction system specialized in identifying which agents participate in a specific event within a folktale.

Your task is to determine which agents from a given candidate list are involved in the event described.

For each participating agent, you must extract:
- The agent's index in the provided list.
- The actions they perform in the event.
- Their importance in the event (0-10 scale)

IMPORTANCE SCALE:
- 0-3: minor or background presence.
- 4-6: relevant participation.
- 7-8: central participant.
- 9.10: critical to the event outcome.

INPUTS YOU WILL RECEIVE:
- Folktale title.
- Event text.
- List of candidate agents (with indices).
- Reasoning/thoughts from another model about the event.

IMPORTANT RULES:
- Only include agents that clearly participate in the event.
- Do NOT invent agents not present in the candidate list.
- The "id" must be the index of the agent in the provided list.
- Each agent must include at least one action.
- Actions must be short, concrete descriptions of what the agent did in the event.

OUTPUT FORMAT:
{{
	"id": int,
	"actions": [string],
	"importance": int
}}
"""
)

event_agent_human_prompt = HumanMessagePromptTemplate.from_template(
	template="""Identify which agents participate in the event and describe their actions and importance.

Folktale title:
{title}
	
Event text:
{event_text}

Other model's thoughts:
{thinking}

Candidate agents:
{agents}

Return:
From the candidate agent above, select only those who participate in the event. For each selected agent, specify:
- Their index (id).
- Their actions in this event.
- Their importance (0-10).
"""
)

event_agent_prompt = ChatPromptTemplate.from_messages([
	event_agent_system_prompt,
	event_agent_human_prompt,
])

def extract_event_agents(model: BaseChatModel, event_text: str, thoughts: list[str], agents: list[Agent], title: str):
	formatted_agents = "\n".join(f"{i}. {agent.name}: {agent.description}" for i, agent in enumerate(agents))

	event_agent_chain = event_agent_prompt | model.with_structured_output(EventAgentsLLM)

	print(event_agent_prompt.format(
		title=title,
		event_text=event_text,
		thinking="\n".join(f"- {thought}" for thought in thoughts),
		agents=formatted_agents
	))

	result = event_agent_chain.invoke({
		"title": title,
		"event_text": event_text,
		"thinking": "\n".join(f"- {thought}" for thought in thoughts),
		"agents": formatted_agents
	})

	result = cast(EventAgentsLLM, result)

	event_agents = []
	for event_agent_llm in result.agents:
		agent_id = event_agent_llm.id

		if 0 <= agent_id < len(agents):
			event_agent = EventAgent.from_llm(event_agent_llm, agents)
			event_agents.append(event_agent)
			
			logger.debug(
				f"Agent={agents[agent_id].name} | "
				f"actions={event_agent.actions} | "
				f"importance={event_agent.importance}"
			)
		else:
			logger.warning(f"Agent index out of range: {agent_id} (valid range: 0-{len(agents) - 1})")

	return event_agents