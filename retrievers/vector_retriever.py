from typing import Optional
from neo4j_manager import Neo4jManager
from utils.models import get_embeddings
from .base_retriever import BaseRetriever

class VectorRetriever(BaseRetriever):
    def __init__(self, neo4j_manager: Neo4jManager, index_name: str, return_projection: Optional[dict[str, str]] = None, extra_match: str = "", include_score: bool = True):
        super().__init__(neo4j_manager, index_name, return_projection, extra_match, include_score)
        self.embedding_gen = get_embeddings()

    def retrieve(self, query: str, top_k: int = 2) -> list[dict]:
        """
        Recupera chunks relevantes usando búsqueda vectorial.
        """
        # Generar embedding de la query
        query_embedding = self.embedding_gen.embed_query(query)

        return_clause = self._build_return_clause()

        # Buscar chunks similares
        cypher_query = f"""
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
        YIELD node, score

        {self.extra_match}

        RETURN {return_clause}
        ORDER BY score DESC
        """

        # print(cypher_query)

        params = {
            "index_name": self.index_name,
            "query_embedding": query_embedding,
            "top_k": top_k
        }

        return self.neo4j.execute_query(cypher_query, params)
