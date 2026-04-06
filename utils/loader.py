from models.folktale import AnnotatedFolktale
from matplotlib.figure import Figure
from loguru import logger
import pandas as pd
from typing import Any
import json
import os

def load_json(path: str):
	"""
	Carga un archivo JSON desde disco y lo devuelve como un objeto Python.
	:param path: Ruta al archivo JSON
	:return: Datos JSON parseados (dict o list)
	"""
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)

	return data

def save_json(path: str, data: dict[str, Any]):
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=4)

def load_json_folder(dir: str):
	"""
	Carga todos los archivos .json de un directorio en un diccionario.
	La clave es el nombre del archivo sin extensión.
	:param dir: Directorio que contiene los archivos JSON
	:return: Dict[str, Any]
	"""
	files = {}
	for file in os.listdir(dir):
		if file.endswith(".json"):
			filename = os.path.splitext(file)[0]

			path = os.path.join(dir, file)
			json = load_json(path)
			files[filename] = json
	return files

def load_txt_folder(dir: str):
	"""
	Carga todos los archivos .txt de un directorio en un diccionario.
	La clave es el nombre del archivo sin extensión.
	:param dir: Directorio que contiene los archivos de texto
	:return: Dict[str, str]
	"""
	files = {}
	for file in os.listdir(dir):
		if file.endswith(".txt"):
			filename = os.path.splitext(file)[0]

			path = os.path.join(dir, file)
			with open(path, "r", encoding="utf-8") as f:
				text = f.read()
			files[filename] = text
	return files

def save_structured_folktale(folktale: AnnotatedFolktale, dir: str, filename: str):
	os.makedirs(dir, exist_ok=True)

	folktale_json = folktale.model_dump(
		mode="json",
		exclude_none=True
	)

	output_file = filename + ".json"
	path = os.path.join(dir, output_file)

	save_json(path, folktale_json)

	logger.success(f"Annotated folktale saved sucessfully. Filename: {os.path.basename(path)}.")

data_dir = "./data"

def load_folktale_csv():
	file = "folk_tales_deduplicated.csv"
	path = os.path.join(data_dir, file)

	df = pd.read_csv(path)
	
	return df

out_dir = "./out"

def save_annotated_folktale(folktale: AnnotatedFolktale, filename: str):
	annotated_dir = os.path.join(out_dir, "annotated")

	save_structured_folktale(folktale, annotated_dir, filename)

def save_fig(fig: Figure, filename: str):
	fig.savefig(filename, dpi=300, bbox_inches="tight")
