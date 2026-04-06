import matplotlib.pyplot as plt
from models.folktale import GenreClass, AnnotatedFolktale
from collections import Counter
import pandas as pd

def plot_genre_distribution(folktales: list[AnnotatedFolktale]):
	"""
	Muestra la distribución de géneros de un conjunto de folktales en un gráfico de barras normalizado.

	Args:
		folktales (list[AnnotatedFolktale]): Lista de folktales anotados con información de género.

	"""
	labels = [genre.value for genre in GenreClass]
	
	genre_counts = Counter(folktale.has_genre for folktale in folktales)
	
	total = sum(genre_counts.values())

	percentages = [
		(genre_counts[genre] / total) * 100 if total > 0 else 0
		for genre in GenreClass
	]

	plt.figure(figsize=(8, 6))
	plt.bar(labels, percentages)
	plt.ylabel("Percentage (%)")
	plt.title("Genre Distribution (Normalized)")
	plt.xticks(rotation=30)
	plt.tight_layout()
	plt.show()

def plot_nation_counts(nation: pd.Series):    
    nation_counts = nation.value_counts()
    
    plt.figure(figsize=(10,6))
    nation_counts.plot(kind="bar", color="skyblue")
    plt.title("Number of Fairy Tales by Nation")
    plt.xlabel("Nation")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    