from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field, ConfigDict
from collections import Counter
from typing import Optional, cast
from utils.regex_utils import snake_case_regex

class Response(BaseModel):
	'''Model output for a single-option classification task.'''

	model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
	
	thinking: str = Field(
		...,
		description=(
			"Brief justification explaining why the selected option best matches "
			"the input text. This field is for interpretability only."
		),
		examples=[
			"The text sets up the hero's normal life and environment before any conflict occurs.",
			"The antagonist performs harmful acts that create obstacles for the hero.",
			"The hero is forced out of a safe place due to circumstances beyond their control.",
			"There is a direct confrontation involving physical combat."
		],
		min_length=1 # No permite strings vacíos
	)

	response: int = Field(
		...,
		description=(
			"Zero-based index of the selected option from the provided list. "
			"The value must correspond to exactly one available option."
		),
		ge=0  # Mayor o igual que 0
	)

class_system_prompt = SystemMessagePromptTemplate.from_template(template="""You are an expert narrative analysis assistant. Your task is to classify a text fragment by selecting the most specific narrative event from a closed list of options.

Instructions:
- Select exactly ONE option from the list.
- Each option is identified by its index number (0, 1, 2, ...).
- The value returned in "response" MUST be a valid index within the bounds of the provided options list.
- Do NOT return an index that is negative or greater than or equal to the number of available options.
- Return ONLY the index number as an integer in "response".
- Do NOT invent options or return natural language in "response".
- If multiple options are applicable, select the one that is the most detailed and specific.
- Abstract or general options should only be chosen if no specific option applies.
Available options:
\"\"\"{options}\"\"\"

{previous_thought}
Output format:
{{
  "thinking": "Brief justification based on the text",
  "response": int
}}
""")

class_human_prompt = HumanMessagePromptTemplate.from_template(
	"""Text to classify:
\"\"\"{event}\"\"\""""
)

class_prompt = ChatPromptTemplate.from_messages([
	class_system_prompt,
	class_human_prompt,
])

def _build_options_prompt(node: dict, self_name: Optional[str] = None) -> tuple[str, list]:
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

def _build_options_prompt_by_list(options: list) -> str:
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

def _extract_event(model: BaseChatModel, folktale_event: str, options: str, previous_thought: str = ""):
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
	class_chain = class_prompt | model.with_structured_output(Response)
	response = class_chain.invoke({
		"options": options,
		"event": folktale_event,
		"previous_thought": previous_thought
	})

	response = cast(Response, response)

	return response.response, response.thinking

def hierarchical_event_classification(model: BaseChatModel, folktale_event: str, taxonomy_tree: dict, n_rounds: int = 3, verbose: bool = False):
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

	current_nodes = taxonomy_tree["event"]["children"]
	previous_event = None
	final_thinking = []
	options_str,options_list = _build_options_prompt(taxonomy_tree["event"])
	level = 0
	final_thinking_str = ""

	if verbose:
		print("=== Inicio de clasificación jerárquica ===")
		print(f"Evento a clasificar: {folktale_event}")

	while current_nodes:
		votes = []
		thoughts = []

		if verbose:
			print(f"\n--- Nivel {level} ---")
			print("Opciones disponibles:")
			print(options_str)

		# Preguntar al LLM n_rounds veces
		for i in range(n_rounds):
			event, thinking = _extract_event(
				model=model,
				folktale_event=folktale_event,
				options=options_str,
				previous_thought=final_thinking_str
			)

			# if 0 <= event < len(options_list):
			if event >= 0 and event < len(options_list):
				votes.append(event)
				thoughts.append(thinking)

				if verbose:
					print(f"\nLlamada al modelo ({i + 1}/{n_rounds})")
					print(f"  Evento propuesto: {options_list[event]}")
					print(f"  Justificación: {thinking}")
			else:
				if verbose:
					print("OUT OF RANGE")
					print(f"\nLlamada al modelo ({i + 1}/{n_rounds})")
					print(f"  Índice del evento: {event}")
					print(f"  Justificación: {thinking}")

		if not votes:
			return previous_event, final_thinking

		if verbose:
			print("\n---\n")
		
		# Voto por mayoría
		vote_count = Counter(votes)
		max_freq = max(vote_count.values())

		#indice de evento 
		most_frequent = [v for v, c in vote_count.items() if c == max_freq]

		winning_event = most_frequent[0]

		if len(most_frequent) > 1:
			selected = [options_list[i] for i in most_frequent]
			selected_str = _build_options_prompt_by_list(selected)
			# event: indice de evento de lista de most_frequent
			event, thinking = _extract_event(
				model=model,
				folktale_event=folktale_event,
				options=selected_str,
				previous_thought=final_thinking_str
			)

			if verbose:
				print(f" Empate:")
				print(selected_str)
				print(f"  Índice del evento: {event}")
				print(f"  Justificación: {thinking}")

			if event >= 0 and event < len(selected):
				return previous_event, final_thinking
			
			winning_event = most_frequent[event]

		final_thinking.extend(
			thoughts[i] for i, v in enumerate(votes) if v == winning_event
		)
		
		winning_event = options_list[winning_event][0]

		if verbose:
			print(f"  Evento propuesto: {winning_event}")

		# Si el evento se repite o no tiene hijos
		if winning_event == previous_event:
			if verbose:
				print("Evento repetido. Finalizando clasificación.")
			return winning_event, final_thinking
		
		if not current_nodes[winning_event]["children"]:
			if verbose:
				print("El evento ganador no tiene hijos. Finalizando clasificación.")
			return winning_event, final_thinking		
		
		options_str,options_list = _build_options_prompt(current_nodes[winning_event],winning_event)

		current_nodes = current_nodes[winning_event]["children"]
		previous_event = winning_event

		final_thinking_str = "Previous decision or reasoning to consider:\n" + "\n".join(final_thinking)

		if verbose:
			print(f"Descendiendo a los hijos de: {winning_event}")
			level += 1

	# if verbose:
	print("\n=== Fin de clasificación ===")
	print(f"  Evento propuesto: {previous_event}")
	print(f"  Justificación: {final_thinking}")

	return previous_event, final_thinking

class EventInstanceName(BaseModel):
	"""
	Model output for generating a generic snake_case narrative event identifier.
	"""

	model_config = ConfigDict(
		str_strip_whitespace=True,
		extra="forbid"
	)

	instance_name: str = Field(
		...,
		description=(
			"Generic narrative event identifier in snake_case. "
			"Must correspond to the event type and contain no story-specific details."
		),
		examples=[
			"antagonist_rushes_to_finish_line",
			"hero_passing_antagonist",
			"loss_of_safe_space",
			"magical_helper_grants_wish",
			"hero_works_hard"
		],
		pattern=snake_case_regex,
		min_length=1
	)

instance_system_prompt = SystemMessagePromptTemplate.from_template(
	template="""You are an expert narrative annotation assistant.
Your task is to generate a GENERIC instance name for a narrative event.

You will be given:
- The event type
- The event text
- A brief reasoning explaining why this is that event type

Instructions:
- Output ONLY a generic instance name.
- Do NOT include specific characters, places, objects, or story-specific details.
- The name must be abstract and reusable.
- Use snake_case.
- Do NOT repeat or summarize the input fields.
- Do NOT output explanations or additional fields.

Output format:
{{
	"instance_name": "generic_event_name"
}}
"""
)

instance_human_prompt = HumanMessagePromptTemplate.from_template(
	template="""Event type:
{event_type}

Event text:
\"\"\"{event_text}\"\"\"

Reasoning:
{thinking}
"""
)

instance_prompt = ChatPromptTemplate.from_messages([
	instance_system_prompt,
	instance_human_prompt,
])

def extract_event_instance_name(model: BaseChatModel, event_type: str, event_text: str, thinking: str = ""):
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
	instance_chain = instance_prompt | model.with_structured_output(EventInstanceName)
	response = instance_chain.invoke({
		"event_type": event_type,
		"event_text": event_text,
		"thinking": thinking
	})
	response = cast(EventInstanceName, response)
	return response.instance_name