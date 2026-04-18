from schemas.folktale import Folktale
from loguru import logger
from typing import Any
import pandas as pd
import json
import os

def load_json(path: str):
	if not path.endswith(".json"):
		path += ".json"

	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)

	return data

def save_json(path: str, data: dict[str, Any]):
	if not path.endswith(".json"):
		path += ".json"

	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=4)

def load_folder(dir: str, extension: str):
	files = {}
	
	for file_name in os.listdir(dir):
		if file_name.endswith(extension):
			key = os.path.splitext(file_name)[0]
			full_path = os.path.join(dir, file_name)

			if extension == ".json":
				files[key] = load_json(full_path)
			else:
				with open(full_path, "r", encoding="utf-8") as f:
					files[key] = f.read()

	return files

def load_json_folder(dir: str):
	return load_folder(dir, ".json")

def load_txt_folder(dir: str):
	return load_folder(dir, ".txt")

def save_structured_folktale(folktale: Folktale, path: str):
	folktale_json = folktale.model_dump(
		mode="json",
		exclude_none=True
	)

	save_json(path, folktale_json)

	logger.success(f"Annotated folktale saved sucessfully in {path}")

def load_csv(path: str):
	if not path.endswith(".csv"):
		path += ".csv"

	return pd.read_csv(path)

def save_csv(path: str, df: pd.DataFrame):
	if not path.endswith(".csv"):
		path += ".csv"

	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

	df.to_csv(path)
	