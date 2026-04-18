from config import get_settings
from langchain_ollama import ChatOllama, OllamaEmbeddings

def get_llm(temperature: float):
	settings = get_settings()

	model = ChatOllama(
		base_url=settings.ollama_base_url,
		model=settings.ollama_model,
		num_gpu=-1,
		temperature=temperature,
		validate_model_on_init=True
	)
	return model

def get_embeddings():
	settings = get_settings()
	
	model = OllamaEmbeddings(
		base_url=settings.ollama_base_url,
		model=settings.ollama_embedding_model,
		num_gpu=-1,
		validate_model_on_init=True
	)

	return model
