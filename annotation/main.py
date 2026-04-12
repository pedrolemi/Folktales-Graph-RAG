from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
# from models.event import EventElements, Event, EventClass
from utils.loader import load_csv, load_json_folder
# from utils.regex_utils import title_case_to_snake_case, clean_regex
from .tools.place_extractor import extract_places
from .tools.agent_extractor import extract_agents
from .tools.genre_extractor import extract_genre
from .tools.event_extractor import extract_story_segments
# from .tools.event_classifier import hierarchical_event_classification, extract_event_instance_name
from .tools.object_extractor import extract_objects
from .tools.relationship_extractor import extract_relationships
# from models.folktale import AnnotatedFolktale
from pandas import DataFrame
from config import get_settings
from loguru import logger
import uuid
import os
import re

def get_model(temperature: float) -> BaseChatModel:
	settings = get_settings()

	model = ChatOllama(
		base_url=settings.ollama_base_url,
		model=settings.ollama_model,
		num_gpu=-1,
		validate_model_on_init=True,
		temperature=temperature
	)
	return model

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
	# log_dir = "./logs"
	# setup_logging(log_dir)

	model = get_model(0.5)

	# hierarchies = load_json_folder(f"{data_dir}/hierarchies")
	
	# event_hierarchy = hierarchies["event"]
	# place_hierarchy = hierarchies["place"]
	# role_hierarchy = hierarchies["role"]
	# object_hierarchy = hierarchies["object"]

	data_dir = "./data"
	processed_dir = os.path.join(data_dir, "processed")
	folktales_path = os.path.join(processed_dir, "folk_tales_deduplicated.csv")
	folktales_df = load_csv(folktales_path)
	selected_folktales_df = get_folktales_by_count(folktales_df, 0, 1)

	metadata_dir = "./metadata"
	# print(os.path.exists(metadata_dir))
	structure_dir = os.path.join(metadata_dir, "structure")
	collections_dir = os.path.join(metadata_dir, "collections")
	entities_dir = os.path.join(metadata_dir, "entities")

	structures = load_json_folder(structure_dir)
	collections = load_json_folder(collections_dir)
	entities = load_json_folder(entities_dir)

	for idx, row in selected_folktales_df.iterrows():
		text = row["text"]
		uri = row["source"].rstrip('/')
		nation = row["nation"]
		title = row["title"]

		# print(title)
		
		logger.debug(f"Starting annotation for title '{title}' (index {idx})...")

		# try:
		# genre = extract_genre(model, text, collections["genre"])

		# objects = extract_objects(model, text, entities["object"])
		# print(objects)

		places = extract_places(model, text, entities["place"])

		agents = extract_agents(model, text, places, structures["role"], collections["trait"])

		relationships = extract_relationships(model, text, agents)

		# story_segments = extract_story_segments(model, text)

		# events = []
		# for segment in story_segments:
		# 	event_metada = EventMetadata(
		# 		title=title,
		# 		agents=agents,
		# 		objects=objects,
		# 		places=places,
		# 		story_segment=segment
		# 	)

		# 	elements = extract_event_elements(model, event_metada, event_examples)

		# 	event_type, thinking = hierarchical_event_classification(
		# 		model=model,
		# 		folktale_event=segment,
		# 		taxonomy_tree=event_hierarchy,
		# 		n_rounds=3,
		# 		verbose=False
		# 	)
			
		# 	if event_type is None:
		# 		event_type = EventClass.EVENT
		# 	instance_name = extract_event_instance_name(model, event_type, segment, "\n".join(thinking))
			
		# 	event = Event(
		# 		class_name=EventClass(event_type),
		# 		instance_name=instance_name,
		# 		description=segment,
		# 		agents=elements.agents,
		# 		objects=elements.objects,
		# 		place=elements.place
		# 	)

		# 	events.append(event)

		# folktale = AnnotatedFolktale(
		# 	uri=uri,
		# 	nation=nation,
		# 	has_genre=genre,
		# 	title=title,
		# 	relationships=relationships,
		# 	agents=agents,
		# 	places=places,
		# 	objects=objects,
		# 	events=events
		# )
			# filename = re.sub(clean_regex, "", title)
			# filename = title_case_to_snake_case(filename)
			# save_annotated_folktale(folktale, f"{idx}_{filename}")

		# except Exception:
		# 	logger.exception(
        # 		"Error processing folktale | title={} | index={}",
        # 		title,
		# 		idx
    	# 	)
		# 	with open(f"{log_dir}/failed_indexes.log", "a", encoding="utf-8") as f:
		# 		f.write(f"{idx}\n")

if __name__ == "__main__":
	main()
