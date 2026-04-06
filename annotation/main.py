from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from models.event import EventElements, EventMetadata, EventExample, Event, EventClass
from utils.loader import load_folktale_csv, load_json_folder, save_annotated_folktale, data_dir
from utils.regex_utils import title_case_to_snake_case, clean_regex
from .tools.place_extractor import extract_places
from .tools.agent_extractor import extract_agents
from .tools.genre_extractor import extract_genre
from .tools.event_extractor import extract_story_segments, extract_event_elements
from .tools.event_classifier import hierarchical_event_classification, extract_event_instance_name
from .tools.object_extractor import extract_objects
from .tools.relationship_extractor import extract_relationships
from models.folktale import AnnotatedFolktale
from pandas import DataFrame
from loguru import logger
import uuid
import os
import re

def get_model(temperature: float) -> BaseChatModel:
	model = ChatOllama(
		base_url=os.environ.get("OLLAMA_HOST"),
		model="llama3.1:8b",
		num_gpu=-1,
		validate_model_on_init=True,
		temperature=temperature
	)
	return model

def get_event_example(folktale: AnnotatedFolktale, event_index: int):
	n_events = len(folktale.events)

	if event_index < n_events:
		event = folktale.events[event_index]
		if event.description:
			example = EventExample(
				title=folktale.title,
				agents=folktale.agents,
				objects=folktale.objects,
				places=folktale.places,
				story_segment=event.description,
				output=EventElements(
					agents=event.agents,
					objects=event.objects,
					place=event.place
				)
			)
			return example
	return None

def get_folktales_by_count(folktales_df: DataFrame, start_index: int, n_folktales: int):
	n_folktales_df = len(folktales_df)
	end_index = min(start_index + n_folktales, n_folktales_df)

	selected_folktales_df = folktales_df.iloc[start_index:end_index]

	return selected_folktales_df

def setup_logging(log_dir: str):
	os.makedirs(log_dir, exist_ok=True)

	run_id = uuid.uuid4().hex[:8]

	log_format = (
		"{time:YYYY-MM-DD HH:mm:ss} | "
        "{level} | "
        f"run={run_id} | "
        "{message}"
	)

	logger.add(
        f"{log_dir}/exceptions.log",
        level="ERROR",
		format=log_format,
        backtrace=True,
        diagnose=True,
        encoding="utf-8",
    )

def main():
	log_dir = "./logs"
	setup_logging(log_dir)

	load_dotenv()

	model = get_model(0.5)

	hierarchies = load_json_folder(f"{data_dir}/hierarchies")

	examples = load_json_folder(f"{data_dir}/examples/annotated")
	cinderella = AnnotatedFolktale(**examples["cinderella"])

	event_examples = []
	cinderella_hero_works_hard = get_event_example(cinderella, 0)
	event_examples.append(cinderella_hero_works_hard)	

	event_hierarchy = hierarchies["event"]
	place_hierarchy = hierarchies["place"]
	role_hierarchy = hierarchies["role"]
	object_hierarchy = hierarchies["object"]

	folktales_df = load_folktale_csv()
	selected_folktales_df = get_folktales_by_count(folktales_df, 5, 300)

	for idx, row in selected_folktales_df.iterrows():
		text = row["text"]
		uri = row["source"].rstrip('/')
		nation = row["nation"]
		if isinstance(nation, str):
			nation = nation.lower()
		else:
			nation = None
		title = row["title"]
		
		logger.debug(f"Starting annotation for title '{title}' (index {idx})...")

		try:
			genre = extract_genre(model, text)

			objects = extract_objects(model, text, object_hierarchy)

			places = extract_places(model, text, place_hierarchy)

			agents = extract_agents(model, text, cinderella.agents, places, role_hierarchy)

			relationships = extract_relationships(model, text, agents)

			story_segments = extract_story_segments(model, text)

			events = []
			for segment in story_segments:
				event_metada = EventMetadata(
					title=title,
					agents=agents,
					objects=objects,
					places=places,
					story_segment=segment
				)

				elements = extract_event_elements(model, event_metada, event_examples)

				event_type, thinking = hierarchical_event_classification(
					model=model,
					folktale_event=segment,
					taxonomy_tree=event_hierarchy,
					n_rounds=3,
					verbose=False
				)
				
				if event_type is None:
					event_type = EventClass.EVENT
				instance_name = extract_event_instance_name(model, event_type, segment, "\n".join(thinking))
				
				event = Event(
					class_name=EventClass(event_type),
					instance_name=instance_name,
					description=segment,
					agents=elements.agents,
					objects=elements.objects,
					place=elements.place
				)

				events.append(event)

			folktale = AnnotatedFolktale(
				uri=uri,
				nation=nation,
				has_genre=genre,
				title=title,
				relationships=relationships,
				agents=agents,
				places=places,
				objects=objects,
				events=events
			)
			filename = re.sub(clean_regex, "", title)
			filename = title_case_to_snake_case(filename)
			save_annotated_folktale(folktale, f"{idx}_{filename}")

		except Exception:
			logger.exception(
        		"Error processing folktale | title={} | index={}",
        		title,
				idx
    		)
			with open(f"{log_dir}/failed_indexes.log", "a", encoding="utf-8") as f:
				f.write(f"{idx}\n")

if __name__ == "__main__":
	main()