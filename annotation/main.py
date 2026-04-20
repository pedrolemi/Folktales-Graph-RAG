from schemas.event import Event, EventClass
from .tools.event_classifier import hierarchical_event_classification, extract_event_name, extract_event_agents
from utils.loader import load_csv, load_json_folder, save_structured_folktale
from .tools.place_extractor import extract_places
from .tools.agent_extractor import extract_agents
from .tools.genre_extractor import extract_genre
from .tools.event_chunking import extract_story_segments
from .tools.event_extractor import extract_event_elements
from .tools.object_extractor import extract_objects
from .tools.relationship_extractor import extract_relationships
from utils.regex_utils import clean_regex, title_case_to_snake_case
from utils.models import get_llm
from schemas.folktale import Folktale
from pandas import DataFrame
from loguru import logger
from tqdm import tqdm
import re
import os

def get_batch(df: DataFrame, start: int, size: int):
    if start < 0:
        raise ValueError("start must be >= 0")

    if size < 0:
        raise ValueError("size must be >= 0")

    return df.iloc[start:start + size]

def main():
	model = get_llm(0.7)

	data_dir = "./data"
	processed_dir = os.path.join(data_dir, "processed")
	folktales_path = os.path.join(processed_dir, "folk_tales_deduplicated.csv")
	folktales_df = load_csv(folktales_path)
	selected_folktales_df = get_batch(folktales_df, 0, 1)

	metadata_dir = "./metadata"
	structure_dir = os.path.join(metadata_dir, "structure")
	collections_dir = os.path.join(metadata_dir, "collections")
	entities_dir = os.path.join(metadata_dir, "entities")

	out_dir = "./out"
	os.makedirs(out_dir, exist_ok=True)

	structures = load_json_folder(structure_dir)
	collections = load_json_folder(collections_dir)
	entities = load_json_folder(entities_dir)

	n_folktales = len(selected_folktales_df)

	for i, row in tqdm(selected_folktales_df.iterrows(), total=n_folktales, desc="Folktales", leave=True):
		text = row["text"]
		url = row["source"]
		nation = row["nation"]
		title = row["title"]
		
		logger.debug(f"Starting annotation: '{title}' (idx={i})")

		genre = extract_genre(model, text, collections["genre"])

		objects = extract_objects(model, text, entities["object"])

		places = extract_places(model, text, entities["place"])

		agents = extract_agents(model, text, places, structures["role"], collections["trait"])

		relationships = extract_relationships(model, text, agents)
		
		story_segments = []
		story_segments = extract_story_segments(model, text)

		events = []
		for j, segment in tqdm(enumerate(story_segments), desc=f"Events ({title})", leave=False):
			objects_ids, place_id = extract_event_elements(model, title, segment, objects, places)

			event_type, final_thoughts = hierarchical_event_classification(
				model=model,
				event_index=j,
				story_segments=story_segments,
				taxonomy_tree=structures["function"],
				n_rounds=3,
				verbose=False
			)
			
			if event_type is None:
				event_type = EventClass.EVENT
			name = extract_event_name(model, event_type, segment, final_thoughts)
			event_agents = extract_event_agents(model, segment, final_thoughts, agents, title)
			
			# u = uuid.uuid4()
			# fixed = f"place_{u.hex}"

			event = Event(
				type=event_type,
				name=name,
				description=segment,
				agents=event_agents,
				objects=objects_ids,
				place=place_id,
				thoughts=final_thoughts
			)

			events.append(event)

		folktale = Folktale(
			url=url,
			nation=nation,
			title=title,
			genre=genre,
			agents=agents,
			relationships=relationships,
			places=places,
			objects=objects,
			events=events
		)

		filename = re.sub(clean_regex, "", title)
		filename = title_case_to_snake_case(filename)
		path = os.path.join(out_dir, f"{i}_{filename}")
		save_structured_folktale(folktale, path)


if __name__ == "__main__":
	main()
